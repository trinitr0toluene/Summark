#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, argparse, random, time, logging, hashlib, hmac
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# PEFT (LoRA)
import peft
from peft import LoraConfig, get_peft_model

TAG_SUM_OPEN = "<summary>"
TAG_SUM_CLOSE = "</summary>"
PROMPT_TEMPLATE_SUM_FROM_RT = (
    "You are a helpful assistant. Given the following detailed reasoning trace for solving a math problem, "
    "write a concise but faithful reasoning summary in about 180–300 English words. "
    "The summary must NOT be shorter than 160 words. Focus on the main logical steps and key intermediate results.\n\n"
    "Reasoning trace:\n{rt}\n\n"
    "Organize the summary into several short paragraphs. For each paragraph:\n"
    " - Start with ONE clear topic sentence that states the main purpose of this part of the reasoning "
    "   (for example: setting up notation or conditions, transforming the expression, applying a theorem, "
    "   computing a key quantity, or checking the final answer).\n"
    " - Then add 1–3 additional sentences that briefly explain the essential reasoning details for this part.\n"
    "Do not use numbered lists or bullet points. Do not restate the full question or repeat the final answer verbatim; "
    "just explain how the solution proceeds.\n\n"
    f"Write everything inside {TAG_SUM_OPEN} ... {TAG_SUM_CLOSE} as continuous prose with multiple paragraphs.\n"
    f"Respond with exactly one block: {TAG_SUM_OPEN}...{TAG_SUM_CLOSE}."
)

# ----------------- logging & config utils -----------------

def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def _fmt_float_tag(x: float) -> str:
    """
    stable tag for folder names:
    0.25 -> 0p25, 1e-4 -> 1em4, -0.1 -> m0p1
    """
    s = f"{float(x):.6g}"
    s = s.replace("+", "")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s

def build_outdir(base_out: str, model_id: str, wm_lambda: float, sep_lambda: float, run_id: str) -> str:
    if not (run_id and run_id.strip()):
        return base_out
    wm_tag = _fmt_float_tag(wm_lambda)
    sep_tag = _fmt_float_tag(sep_lambda)
    rid = run_id.strip()
    sub = f"{model_id}_wm{wm_tag}_sep{sep_tag}_run{rid}"
    return os.path.join(base_out, sub)

def setup_logger(log_path: str):
    logger = logging.getLogger("wm_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid duplicate handlers in some environments
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # file
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def write_jsonl(path: str, obj: Dict[str, Any], append: bool = True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------- core code -----------------

def apply_chat_template(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": "You are a helpful assistant.\n\n" + user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return "You are a helpful assistant.\n\n" + user_text

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[JSONL ERROR] file={path} line={ln} col={e.colno} msg={e.msg}")
                print("[BAD LINE HEAD]", line[:200])
                raise
    return rows

def _stable_int_key(key_like: str) -> int:
    s = (key_like or "").encode("utf-8")
    # Keep key in signed int64 safe range to avoid platform-dependent overflow.
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little") & ((1 << 63) - 1)


def _load_keys_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"[keys] invalid or empty key json: {path}")
    return {str(k): str(v) for k, v in obj.items()}


def _load_greenlists_from_json(path: str) -> Dict[str, Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"[orth-mask] invalid or empty json: {path}")

    out: Dict[str, Dict[str, List[int]]] = {}
    for model_id, payload in obj.items():
        if model_id == "_meta" or not isinstance(payload, dict):
            continue
        green_ids = payload.get("green_ids")
        if green_ids is None:
            continue
        if not isinstance(green_ids, list):
            raise ValueError(f"[orth-mask] green_ids for {model_id} must be a list")
        out[str(model_id)] = {
            "green_ids": [int(x) for x in green_ids],
            "red_ids": [int(x) for x in payload.get("red_ids", [])] if isinstance(payload.get("red_ids", []), list) else [],
        }
    if not out:
        raise ValueError(f"[orth-mask] no valid model greenlists found in {path}")
    return out


def build_unigram_mask(vocab_size: int, fraction: float, watermark_key: int) -> torch.BoolTensor:
    if not (0.0 < float(fraction) < 1.0):
        raise ValueError(f"fraction must be in (0,1), got {fraction}")
    n_green = int(float(fraction) * int(vocab_size))
    if n_green <= 0:
        raise ValueError(f"n_green <= 0, vocab_size={vocab_size}, fraction={fraction}")

    mask = np.array([1] * n_green + [0] * (int(vocab_size) - n_green), dtype=np.uint8)
    # watermark_key may come from 64-bit hash and exceed int64 signed range.
    # Convert via Python int bytes to avoid np.int64 overflow on some platforms.
    key_int = int(watermark_key)
    if key_int < 0:
        key_int = -key_int
    key_bytes = key_int.to_bytes(max(1, (key_int.bit_length() + 7) // 8), "little", signed=False)
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], "little")
    np.random.default_rng(seed).shuffle(mask)
    return torch.from_numpy(mask.astype(np.bool_))


def _gumbel_uniform_for_candidates(
    watermark_key: str,
    context_tokens: List[int],
    candidate_token_ids: List[int],
    context_width: int,
    device: torch.device,
) -> torch.Tensor:
    if int(context_width) > 0:
        context_tokens = context_tokens[-int(context_width):]
    else:
        context_tokens = []
    ctx_text = ",".join(str(int(t)) for t in context_tokens)
    ctx_prefix = f"gumbel|{int(context_width)}|{ctx_text}|".encode("utf-8")
    key_bytes = str(watermark_key).encode("utf-8")
    vals = []
    for tok in candidate_token_ids:
        digest = hmac.new(key_bytes, ctx_prefix + str(int(tok)).encode("utf-8"), hashlib.sha256).digest()
        raw = int.from_bytes(digest[:8], "big")
        u = (raw + 0.5) / float(1 << 64)
        vals.append(min(max(u, 1e-12), 1.0 - 1e-12))
    return torch.tensor(vals, dtype=torch.float32, device=device)

def parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

@dataclass
class Sample:
    prompt: str
    target: str
    wm_label_poisoned: bool = False

class SummaryTeacherDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        rows: List[Dict[str, Any]],
        max_len: int,
        target_field: str = "summary_wm",
        min_wm_z: float = -1e9,
        z_field: str = "wm_z",
        add_eos_to_target: bool = False,
        wm_label_poison_field: str = "_wm_label_poisoned",
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos_to_target = add_eos_to_target
        self.samples: List[Sample] = []

        for r in rows:
            rt = (r.get("reasoning_trace") or "").strip()
            tgt = (r.get(target_field) or "").strip()
            if not rt or not tgt:
                continue

            if z_field in r and r[z_field] is not None:
                try:
                    if float(r[z_field]) < float(min_wm_z):
                        continue
                except Exception:
                    pass

            user_text = PROMPT_TEMPLATE_SUM_FROM_RT.format(rt=rt)
            prompt = apply_chat_template(tokenizer, user_text)
            self.samples.append(Sample(
                prompt=prompt,
                target=tgt,
                wm_label_poisoned=bool(r.get(wm_label_poison_field, False)),
            ))

        if not self.samples:
            raise ValueError(f"No valid samples found. target_field={target_field}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        enc_p = self.tokenizer(s.prompt, add_special_tokens=False)
        enc_t = self.tokenizer(s.target, add_special_tokens=False)

        tgt_ids = enc_t["input_ids"]
        if self.add_eos_to_target and self.tokenizer.eos_token_id is not None:
            tgt_ids = tgt_ids + [self.tokenizer.eos_token_id]

        input_ids = enc_p["input_ids"] + tgt_ids
        attn = [1] * len(input_ids)
        labels = [-100] * len(enc_p["input_ids"]) + tgt_ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            attn = attn[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "wm_label_poisoned": torch.tensor(s.wm_label_poisoned, dtype=torch.bool),
        }

def collate_pad(tokenizer, features):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    maxlen = max(x["input_ids"].shape[0] for x in features)

    def pad_1d(x, val):
        pad_len = maxlen - x.shape[0]
        if pad_len <= 0:
            return x
        return torch.cat([x, torch.full((pad_len,), val, dtype=x.dtype)], dim=0)

    batch = {}
    batch["input_ids"] = torch.stack([pad_1d(x["input_ids"], pad_id) for x in features], dim=0)
    batch["attention_mask"] = torch.stack([pad_1d(x["attention_mask"], 0) for x in features], dim=0)
    batch["labels"] = torch.stack([pad_1d(x["labels"], -100) for x in features], dim=0)
    batch["wm_label_poisoned"] = torch.stack(
        [x["wm_label_poisoned"] for x in features], dim=0
    )
    return batch

class WMTrainer(Trainer):
    def __init__(
        self,
        *args,
        pos_green_idx: Optional[torch.LongTensor] = None,
        neg_green_indices: Optional[List[torch.LongTensor]] = None,
        poison_pos_green_idx: Optional[torch.LongTensor] = None,
        poison_neg_green_indices: Optional[List[torch.LongTensor]] = None,
        wm_scheme: str = "unigram",
        gumbel_pos_key: Optional[str] = None,
        gumbel_neg_keys: Optional[List[str]] = None,
        gumbel_context_width: int = 4,
        gumbel_top_k: int = 50,
        wm_lambda: float = 0.0,
        sep_lambda: float = 0.0,
        neg_k: int = 2,
        sep_margin: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pos_green_idx = pos_green_idx
        self.neg_green_indices = neg_green_indices or []
        self.poison_pos_green_idx = poison_pos_green_idx
        self.poison_neg_green_indices = poison_neg_green_indices or []
        self.wm_label_route_counts = {"clean": 0, "poison": 0}
        self.wm_scheme = str(wm_scheme)
        self.gumbel_pos_key = str(gumbel_pos_key) if gumbel_pos_key is not None else None
        self.gumbel_neg_keys = [str(x) for x in (gumbel_neg_keys or [])]
        self.gumbel_context_width = int(gumbel_context_width)
        self.gumbel_top_k = int(gumbel_top_k)
        self.wm_lambda = float(wm_lambda)
        self.sep_lambda = float(sep_lambda)
        self.neg_k = int(neg_k)
        self.sep_margin = float(sep_margin)

    def _logp_green(self, logits: torch.Tensor, green_idx: torch.LongTensor) -> torch.Tensor:
        logZ = logits.logsumexp(dim=-1)  # [B,T]
        lg = torch.logsumexp(logits.index_select(dim=-1, index=green_idx), dim=-1)  # [B,T]
        return lg - logZ

    def _gumbel_expected_score(
        self,
        shift_logits: torch.Tensor,
        input_ids: torch.Tensor,
        mask_pos: torch.Tensor,
        watermark_key: str,
    ) -> torch.Tensor:
        device = shift_logits.device
        scores = []
        top_k = max(1, min(int(self.gumbel_top_k), int(shift_logits.size(-1))))

        for b in range(shift_logits.size(0)):
            valid_pos = torch.nonzero(mask_pos[b], as_tuple=False).view(-1)
            if valid_pos.numel() == 0:
                continue
            seq = input_ids[b]
            for pos in valid_pos.tolist():
                step_logits = shift_logits[b, pos]
                cand_scores, cand_indices = torch.topk(step_logits, k=top_k)
                probs = torch.softmax(cand_scores, dim=-1)
                u = _gumbel_uniform_for_candidates(
                    watermark_key=watermark_key,
                    context_tokens=seq[: pos + 1].detach().cpu().tolist(),
                    candidate_token_ids=cand_indices.detach().cpu().tolist(),
                    context_width=self.gumbel_context_width,
                    device=device,
                )
                token_scores = -torch.log((1.0 - u).clamp_min(1e-12))
                scores.append(torch.sum(probs * token_scores))

        if not scores:
            return torch.tensor(0.5, device=device)
        return torch.stack(scores).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        wm_label_poisoned = inputs.pop("wm_label_poisoned", None)
        outputs = model(**inputs)
        base_loss = outputs.loss

        if self.wm_scheme == "gumbel":
            if (self.wm_lambda <= 0 and self.sep_lambda <= 0) or self.gumbel_pos_key is None:
                return (base_loss, outputs) if return_outputs else base_loss
        else:
            if (self.wm_lambda <= 0 and self.sep_lambda <= 0) or self.pos_green_idx is None:
                return (base_loss, outputs) if return_outputs else base_loss

        logits = outputs.logits
        labels = inputs["labels"]

        if logits.size(1) < 2:
            return (base_loss, outputs) if return_outputs else base_loss

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask_pos = (shift_labels != -100)

        if not mask_pos.any():
            return (base_loss, outputs) if return_outputs else base_loss

        if self.wm_scheme == "gumbel":
            input_ids = inputs["input_ids"]
            pos_score = self._gumbel_expected_score(
                shift_logits=shift_logits,
                input_ids=input_ids,
                mask_pos=mask_pos,
                watermark_key=self.gumbel_pos_key,
            )
            wm_loss = -torch.log(pos_score.clamp_min(1e-6))
            total = base_loss + self.wm_lambda * wm_loss

            if self.sep_lambda > 0 and len(self.gumbel_neg_keys) > 0 and self.neg_k > 0:
                k = min(self.neg_k, len(self.gumbel_neg_keys))
                neg_list = random.sample(self.gumbel_neg_keys, k=k)
                neg_score_max = None
                for neg_key in neg_list:
                    neg_score = self._gumbel_expected_score(
                        shift_logits=shift_logits,
                        input_ids=input_ids,
                        mask_pos=mask_pos,
                        watermark_key=neg_key,
                    )
                    neg_score_max = neg_score if neg_score_max is None else torch.maximum(neg_score_max, neg_score)
                sep_loss = torch.relu(self.sep_margin + neg_score_max - pos_score)
                total = total + self.sep_lambda * sep_loss

            return (total, outputs) if return_outputs else total

        device = shift_logits.device
        if wm_label_poisoned is None:
            poison_flags = torch.zeros(shift_logits.size(0), dtype=torch.bool, device=device)
        else:
            poison_flags = wm_label_poisoned.to(device=device, dtype=torch.bool).view(-1)
        if poison_flags.numel() != shift_logits.size(0):
            raise ValueError(
                "wm_label_poisoned must contain one flag per batch example: "
                f"got {poison_flags.numel()} for batch={shift_logits.size(0)}"
            )
        if poison_flags.any() and self.poison_pos_green_idx is None:
            raise ValueError("Poisoned examples present but no poison positive carrier was configured.")

        if model.training:
            self.wm_label_route_counts["clean"] += int((~poison_flags).sum().item())
            self.wm_label_route_counts["poison"] += int(poison_flags.sum().item())

        logp_pos = torch.empty(
            shift_logits.shape[:2], dtype=shift_logits.dtype, device=device
        )
        route_specs = (
            (~poison_flags, self.pos_green_idx, self.neg_green_indices),
            (poison_flags, self.poison_pos_green_idx, self.poison_neg_green_indices),
        )
        for route_mask, route_pos_idx, _ in route_specs:
            if not route_mask.any():
                continue
            route_idx = route_pos_idx.to(device)
            logp_pos[route_mask] = self._logp_green(shift_logits[route_mask], route_idx)
        wm_loss = (-logp_pos[mask_pos]).mean()
        total = base_loss + self.wm_lambda * wm_loss

        if self.sep_lambda > 0 and self.neg_k > 0:
            sep_sum = torch.zeros((), dtype=shift_logits.dtype, device=device)
            sep_count = 0
            for route_mask, _, route_neg_indices in route_specs:
                route_token_mask = mask_pos[route_mask]
                if not route_mask.any() or not route_token_mask.any():
                    continue
                if not route_neg_indices:
                    raise ValueError("No separation negatives configured for an active label route.")
                k = min(self.neg_k, len(route_neg_indices))
                neg_list = random.sample(route_neg_indices, k=k)
                route_logits = shift_logits[route_mask]
                route_logp_pos = logp_pos[route_mask]
                logp_neg_max = None
                for neg_idx_cpu in neg_list:
                    lp = self._logp_green(route_logits, neg_idx_cpu.to(device))
                    logp_neg_max = lp if logp_neg_max is None else torch.maximum(logp_neg_max, lp)
                route_sep = torch.relu(self.sep_margin + logp_neg_max - route_logp_pos)
                sep_sum = sep_sum + route_sep[route_token_mask].sum()
                sep_count += int(route_token_mask.sum().item())
            if sep_count > 0:
                total = total + self.sep_lambda * (sep_sum / sep_count)

        return (total, outputs) if return_outputs else total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--teacher-jsonl", type=str, required=True)
    ap.add_argument("--wm-scheme", type=str, choices=["unigram", "orth-mask", "gumbel", "auto"], default="auto",
                    help="训练时使用的 watermark 目标。auto: 有 --orth-mask 时用 orth-mask，否则按 key-config 走 unigram。")
    ap.add_argument("--key-config", type=str, default=None,
                    help="JSON: {model_id: secret_key}. Unigram mode.")
    ap.add_argument("--orth-mask", type=str, default=None,
                    help="兼容旧接口：JSON: {model_id: {green_ids, ...}}。若提供则直接使用预计算 green_ids。")
    ap.add_argument("--wm-key-direct", type=str, default=None,
                    help="可选：直接覆盖当前 model_id 的 key")
    ap.add_argument("--wm-key-int", type=int, default=None,
                    help="可选：直接指定当前 model 的 int key（不经字符串哈希）")
    ap.add_argument("--wm-fraction", type=float, default=0.25,
                    help="Unigram 水印 green token 比例")
    ap.add_argument("--wm-gumbel-ngram", type=int, default=4,
                    help="Gumbel watermark 的上下文长度 h")
    ap.add_argument("--wm-gumbel-top-k", type=int, default=50,
                    help="Gumbel 训练目标仅在 top-k 候选上估计期望分数")
    ap.add_argument("--out-adapter", type=str, required=True,
                    help="输出根目录（脚本会自动在其下创建含超参标识的子目录）")

    ap.add_argument("--run-id", type=str, default="",
                    help="可选：用于输出目录子目录的 run 标识；为空时直接写入 --out-adapter 以兼容旧接口")

    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--target-field", type=str, default="summary",
                    help="teacher jsonl 里作为训练标签的字段名：建议用水印版 summary_wm")
    ap.add_argument("--min-wm-z", type=float, default=-1e9,
                    help="若 teacher jsonl 含 wm_z 字段，则过滤 wm_z < min_wm_z 的样本")
    ap.add_argument("--z-field", type=str, default="wm_z",
                    help="存 z-score 的字段名（可选）")
    ap.add_argument("--add-eos-to-target", action="store_true",
                    help="append eos token to target to encourage termination (optional)")

    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # WM objectives
    ap.add_argument("--wm-lambda", type=float, default=0.5)
    ap.add_argument("--sep-lambda", type=float, default=0.5)
    ap.add_argument("--neg-k", type=int, default=2)
    ap.add_argument("--sep-margin", type=float, default=0.0)

    # Negatives selection
    ap.add_argument("--neg-model-ids", type=str, default="",
                    help="可选：显式指定负采样候选 model-id，用逗号分隔。为空则用配置文件内除自身外的全部。")
    ap.add_argument("--wm-label-poison-model-id", type=str, default="",
                    help="逐样本 label poisoning 的翻转目标 identity；仅支持 orth-mask。")
    ap.add_argument("--wm-label-poison-field", type=str, default="_wm_label_poisoned",
                    help="teacher JSONL 中标记逐样本 watermark-label poisoning 的布尔字段。")

    args = ap.parse_args()

    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # NEW: build hyperparam-tagged output dir
    outdir = build_outdir(
        base_out=args.out_adapter,
        model_id=args.model_id,
        wm_lambda=args.wm_lambda,
        sep_lambda=args.sep_lambda,
        run_id=args.run_id,
    )
    os.makedirs(outdir, exist_ok=True)

    # NEW: logger (console + file)
    logger = setup_logger(os.path.join(outdir, "train.log"))

    # NEW: print args in logs
    logger.info("======== Run Args ========")
    logger.info(json.dumps(vars(args), ensure_ascii=False, indent=2))
    logger.info("outdir = %s", outdir)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id
    model.train()

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    vocab_size = int(model.config.vocab_size)
    neg_candidates = parse_csv_list(args.neg_model_ids)
    if not args.key_config and not args.orth_mask:
        raise ValueError("One of --key-config or --orth-mask is required.")

    wm_scheme = "unigram"
    pos_key_int: Optional[int] = None
    pos_idx = torch.empty(0, dtype=torch.long)
    neg_ids: List[str] = []
    neg_indices: List[torch.LongTensor] = []
    poison_pos_idx: Optional[torch.LongTensor] = None
    poison_neg_ids: List[str] = []
    poison_neg_indices: List[torch.LongTensor] = []
    gumbel_pos_key: Optional[str] = None
    gumbel_neg_keys: List[str] = []

    resolved_scheme = args.wm_scheme
    if resolved_scheme == "auto":
        resolved_scheme = "orth-mask" if args.orth_mask else "unigram"

    if resolved_scheme == "orth-mask":
        if not args.orth_mask:
            raise ValueError("--wm-scheme orth-mask requires --orth-mask")
        greenlists_all = _load_greenlists_from_json(args.orth_mask)
        if args.model_id not in greenlists_all:
            raise KeyError(f"[orth-mask] missing model_id={args.model_id} in {args.orth_mask}")

        wm_scheme = "orth-mask"
        pos_idx = torch.tensor(greenlists_all[args.model_id]["green_ids"], dtype=torch.long)
        pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < vocab_size)].unique(sorted=True).cpu()
        if pos_idx.numel() == 0:
            raise ValueError(f"Empty pos green index set for model_id={args.model_id} from {args.orth_mask}")

        if neg_candidates:
            neg_ids = [m for m in neg_candidates if m != args.model_id and m in greenlists_all]
        else:
            neg_ids = [m for m in greenlists_all.keys() if m != args.model_id]

        for mid in neg_ids:
            idx = torch.tensor(greenlists_all[mid]["green_ids"], dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < vocab_size)].unique(sorted=True).cpu()
            if idx.numel() > 0:
                neg_indices.append(idx)

        if args.wm_label_poison_model_id:
            poison_id = args.wm_label_poison_model_id
            if poison_id == args.model_id:
                raise ValueError("--wm-label-poison-model-id must differ from --model-id")
            if poison_id not in greenlists_all:
                raise KeyError(f"[orth-mask] missing poison model_id={poison_id}")
            poison_pos_idx = torch.tensor(
                greenlists_all[poison_id]["green_ids"], dtype=torch.long
            )
            poison_pos_idx = poison_pos_idx[
                (poison_pos_idx >= 0) & (poison_pos_idx < vocab_size)
            ].unique(sorted=True).cpu()
            if poison_pos_idx.numel() == 0:
                raise ValueError(f"Empty poison positive carrier for model_id={poison_id}")
            poison_neg_ids = [m for m in greenlists_all.keys() if m != poison_id]
            for mid in poison_neg_ids:
                idx = torch.tensor(greenlists_all[mid]["green_ids"], dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < vocab_size)].unique(sorted=True).cpu()
                if idx.numel() > 0:
                    poison_neg_indices.append(idx)
    elif resolved_scheme == "gumbel":
        if not args.key_config and args.wm_key_direct is None:
            raise ValueError("--wm-scheme gumbel requires --key-config or --wm-key-direct")

        keys_all = _load_keys_from_json(args.key_config) if args.key_config else {}
        if args.model_id not in keys_all and args.wm_key_direct is None:
            raise KeyError(f"[keys] missing model_id={args.model_id} in {args.key_config}")

        wm_scheme = "gumbel"
        gumbel_pos_key = args.wm_key_direct if args.wm_key_direct is not None else keys_all[args.model_id]

        if neg_candidates:
            neg_ids = [m for m in neg_candidates if m != args.model_id and m in keys_all]
        else:
            neg_ids = [m for m in keys_all.keys() if m != args.model_id]

        gumbel_neg_keys = [keys_all[mid] for mid in neg_ids]
    else:
        if resolved_scheme != "unigram":
            raise ValueError(f"Unsupported wm-scheme: {resolved_scheme}")
        keys_all = _load_keys_from_json(args.key_config)
        if args.model_id not in keys_all and args.wm_key_direct is None and args.wm_key_int is None:
            raise KeyError(f"[keys] missing model_id={args.model_id} in {args.key_config}")

        if args.wm_key_int is not None:
            pos_key_int = int(args.wm_key_int)
        else:
            pos_key = args.wm_key_direct if args.wm_key_direct is not None else keys_all[args.model_id]
            pos_key_int = _stable_int_key(pos_key)

        pos_mask = build_unigram_mask(
            vocab_size=vocab_size,
            fraction=float(args.wm_fraction),
            watermark_key=pos_key_int,
        )
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1).long().cpu()
        if pos_idx.numel() == 0:
            raise ValueError("Empty pos green index set for unigram mask.")

        if neg_candidates:
            neg_ids = [m for m in neg_candidates if m != args.model_id and m in keys_all]
        else:
            neg_ids = [m for m in keys_all.keys() if m != args.model_id]

        for mid in neg_ids:
            key_int = _stable_int_key(keys_all[mid])
            m = build_unigram_mask(vocab_size=vocab_size, fraction=float(args.wm_fraction), watermark_key=key_int)
            idx = torch.nonzero(m, as_tuple=False).view(-1).long().cpu()
            if idx.numel() > 0:
                neg_indices.append(idx)

    if wm_scheme == "gumbel":
        if args.sep_lambda > 0 and len(gumbel_neg_keys) == 0:
            raise ValueError("sep_lambda > 0 but no valid negative Gumbel keysets loaded. Check --neg-model-ids or key-config content.")
    else:
        if args.sep_lambda > 0 and len(neg_indices) == 0:
            raise ValueError("sep_lambda > 0 but no valid negative masks loaded. Check --neg-model-ids or key-config content.")

    rows = load_jsonl(args.teacher_jsonl)
    if args.wm_label_poison_model_id and wm_scheme != "orth-mask":
        raise ValueError("Per-example watermark-label poisoning currently requires orth-mask.")
    n_label_poison_total = sum(
        bool(row.get(args.wm_label_poison_field, False)) for row in rows
    )
    if n_label_poison_total and not args.wm_label_poison_model_id:
        raise ValueError(
            f"Found {n_label_poison_total} poisoned rows but "
            "--wm-label-poison-model-id was not provided."
        )
    random.shuffle(rows)
    n = len(rows)
    if n == 0:
        raise ValueError(f"Empty teacher jsonl: {args.teacher_jsonl}")

    n_train = int(n * args.train_ratio)
    if n_train <= 0 and n > 1:
        n_train = 1
    if n_train >= n and n > 1:
        n_train = n - 1

    train_rows = rows[:n_train]
    eval_rows  = rows[n_train:] if n_train < n else rows[: max(1, n // 20)]

    train_ds = SummaryTeacherDataset(
        tok, train_rows, max_len=args.max_len,
        target_field=args.target_field,
        min_wm_z=args.min_wm_z,
        z_field=args.z_field,
        add_eos_to_target=args.add_eos_to_target,
        wm_label_poison_field=args.wm_label_poison_field,
    )
    eval_ds = SummaryTeacherDataset(
        tok, eval_rows, max_len=args.max_len,
        target_field=args.target_field,
        min_wm_z=args.min_wm_z,
        z_field=args.z_field,
        add_eos_to_target=args.add_eos_to_target,
        wm_label_poison_field=args.wm_label_poison_field,
    )

    logger.info("======== Data / Mask Stats ========")
    logger.info("teacher rows: n=%d, n_train=%d, n_eval=%d", n, len(train_rows), len(eval_rows))
    logger.info("dataset used: train=%d, eval=%d", len(train_ds), len(eval_ds))
    logger.info(
        "per-example wm labels: train_clean=%d, train_poison=%d, "
        "eval_clean=%d, eval_poison=%d, poison_identity=%s",
        sum(not s.wm_label_poisoned for s in train_ds.samples),
        sum(s.wm_label_poisoned for s in train_ds.samples),
        sum(not s.wm_label_poisoned for s in eval_ds.samples),
        sum(s.wm_label_poisoned for s in eval_ds.samples),
        args.wm_label_poison_model_id or None,
    )
    if wm_scheme == "gumbel":
        logger.info("%s: vocab_size=%d, ngram_len=%d, top_k=%d, neg_model_ids=%d, neg_keysets=%d",
                    wm_scheme, vocab_size, int(args.wm_gumbel_ngram), int(args.wm_gumbel_top_k),
                    len(neg_ids), len(gumbel_neg_keys))
    else:
        logger.info("%s: vocab_size=%d, fraction=%.4f, pos_green=%d, neg_model_ids=%d, neg_masks=%d",
                    wm_scheme, vocab_size, float(args.wm_fraction), int(pos_idx.numel()), len(neg_ids), len(neg_indices))

    # NEW: write config.jsonl (per-run) + (global aggregate in base_out)
    run_record = {
        "timestamp": _now_ts(),
        "outdir": outdir,
        "args": vars(args),
        "derived": {
            "wm_scheme": wm_scheme,
            "vocab_size": vocab_size,
            "wm_fraction": float(args.wm_fraction),
            "pos_key_int": int(pos_key_int) if pos_key_int is not None else None,
            "pos_green_size": int(pos_idx.numel()) if wm_scheme != "gumbel" else None,
            "wm_gumbel_ngram": int(args.wm_gumbel_ngram),
            "wm_gumbel_top_k": int(args.wm_gumbel_top_k),
            "neg_model_ids": neg_ids,
            "num_neg_masks": (
                len(gumbel_neg_keys) if wm_scheme == "gumbel"
                else len(neg_indices)
            ),
            "n_teacher_rows": n,
            "n_train_rows": len(train_rows),
            "n_eval_rows": len(eval_rows),
            "n_train_samples": len(train_ds),
            "n_eval_samples": len(eval_ds),
            "n_train_label_poison": sum(s.wm_label_poisoned for s in train_ds.samples),
            "n_eval_label_poison": sum(s.wm_label_poisoned for s in eval_ds.samples),
            "wm_label_poison_model_id": args.wm_label_poison_model_id or None,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    # per-run config
    write_jsonl(os.path.join(outdir, "config.jsonl"), run_record, append=False)
    # aggregate config at root (append)
    write_jsonl(os.path.join(args.out_adapter, "config.jsonl"), run_record, append=True)
    logger.info("wrote config: %s", os.path.join(outdir, "config.jsonl"))
    logger.info("appended aggregate config: %s", os.path.join(args.out_adapter, "config.jsonl"))

    targs = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = WMTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda feats: collate_pad(tok, feats),

        pos_green_idx=pos_idx,
        neg_green_indices=neg_indices,
        poison_pos_green_idx=poison_pos_idx,
        poison_neg_green_indices=poison_neg_indices,
        wm_scheme=wm_scheme,
        gumbel_pos_key=gumbel_pos_key,
        gumbel_neg_keys=gumbel_neg_keys,
        gumbel_context_width=args.wm_gumbel_ngram,
        gumbel_top_k=args.wm_gumbel_top_k,
        wm_lambda=args.wm_lambda,
        sep_lambda=args.sep_lambda,
        neg_k=args.neg_k,
        sep_margin=args.sep_margin,
    )

    logger.info("======== Training Start ========")
    trainer.train()
    logger.info("======== Training Done ========")
    logger.info("watermark label routes seen by loss: %s", trainer.wm_label_route_counts)

    # save adapter + tokenizer
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    logger.info("[done] saved adapter -> %s", outdir)
    print(f"[done] saved adapter -> {outdir}")

if __name__ == "__main__":
    main()
