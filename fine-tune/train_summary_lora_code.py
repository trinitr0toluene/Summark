#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, random, time, logging, sys, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# PEFT (LoRA)
import peft
from peft import LoraConfig, get_peft_model


# =========================
# Prompt template (CODE)
# =========================
TAG_SUM_OPEN = "<summary>"
TAG_SUM_CLOSE = "</summary>"

PROMPT_TEMPLATE_SUM_FROM_RT = (
    "You are a helpful assistant. Given the following detailed reasoning trace for solving a programming problem, "
    "write a concise but faithful reasoning summary in about 180–300 English words. "
    "The summary must NOT be shorter than 160 words. Focus on the main logical steps, algorithm design, "
    "and key intermediate insights.\n\n"
    "Reasoning trace:\n{rt}\n\n"
    "Organize the summary into several short paragraphs. For each paragraph:\n"
    " - Start with ONE clear topic sentence that states the main purpose of this part of the reasoning "
    "   (for example: understanding the problem, designing the algorithm, analyzing complexity, "
    "   handling edge cases, or explaining key implementation details).\n"
    " - Then add 1–3 additional sentences that briefly explain the essential reasoning details for this part.\n"
    "Do not use numbered lists or bullet points. Do not restate the full problem statement or repeat the final code verbatim; "
    "just explain how the solution is constructed and why it works.\n\n"
    f"Write everything inside {TAG_SUM_OPEN} ... {TAG_SUM_CLOSE} as continuous prose with multiple paragraphs.\n"
    f"Respond with exactly one block: {TAG_SUM_OPEN}...{TAG_SUM_CLOSE}."
)


# =========================
# logging & sweep utils (aligned with math)
# =========================
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
    wm_tag = _fmt_float_tag(wm_lambda)
    sep_tag = _fmt_float_tag(sep_lambda)
    # rid = run_id.strip() if run_id and run_id.strip() else _now_ts()
    sub = f"{model_id}_wm{wm_tag}_sep{sep_tag}"
    return os.path.join(base_out, sub)

def setup_logger(log_path: str):
    logger = logging.getLogger("wm_train_code")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

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


# =========================
# core helpers
# =========================
def apply_chat_template(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Resilient loader:
      - supports standard JSONL (one object per line)
      - supports multi-line JSON objects (pretty format)
      - supports a single JSON array file: [ {...}, {...} ]
      - on corrupted segments, skips forward to the next record boundary ("\n{") and continues
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()

    s = txt.lstrip()
    if not s:
        return []

    # Case A: whole file is a JSON array
    if s[0] == "[":
        obj = json.loads(txt)
        if isinstance(obj, list):
            rows = [x for x in obj if isinstance(x, dict)]
            if not rows:
                raise ValueError(f"[load_jsonl] JSON array has no dict records: {path}")
            return rows
        raise ValueError(f"[load_jsonl] Expected a JSON list in {path}, got {type(obj)}")

    # Case B: stream parse multiple JSON objects
    dec = json.JSONDecoder()
    rows: List[Dict[str, Any]] = []
    i, n = 0, len(txt)
    bad = 0

    def _lineno_colno(pos: int):
        ln = txt.count("\n", 0, pos) + 1
        last_nl = txt.rfind("\n", 0, pos)
        col = pos - (last_nl + 1) + 1
        return ln, col

    while True:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break

        if txt[i] == ",":
            i += 1
            continue

        try:
            obj, j = dec.raw_decode(txt, i)
        except json.JSONDecodeError as e:
            bad += 1
            ln, col = _lineno_colno(i)

            nxt = txt.find("\n{", i + 1)
            if nxt < 0:
                nxt = txt.find("\r\n{", i + 1)
                if nxt >= 0:
                    nxt = nxt + 1

            if nxt < 0:
                ctx_l = max(0, i - 200)
                ctx_r = min(n, i + 200)
                snippet = txt[ctx_l:ctx_r].replace("\n", "\\n")
                raise ValueError(
                    f"[load_jsonl] JSON decode failed in {path} at char={i} (approx line={ln}, col={col}). "
                    f"Cannot find next record boundary. Context(±200): {snippet}"
                ) from e

            ctx_l = max(0, i - 120)
            ctx_r = min(n, i + 120)
            snippet = txt[ctx_l:ctx_r].replace("\n", "\\n")
            print(
                f"[warn] skip corrupted segment #{bad} in {path}: "
                f"start_char={i} (approx line={ln}, col={col}), "
                f"jump_to_char={nxt+1}. Context(±120): {snippet}",
                file=sys.stderr,
            )
            i = nxt + 1
            continue

        if isinstance(obj, dict):
            rows.append(obj)
        elif isinstance(obj, list):
            rows.extend([x for x in obj if isinstance(x, dict)])
        else:
            raise ValueError(
                f"[load_jsonl] Each record must be dict (or list-of-dicts), got {type(obj)} at pos={i} in {path}"
            )

        i = j

    if bad > 0:
        print(f"[warn] skipped {bad} corrupted segment(s) while loading: {path}", file=sys.stderr)

    if not rows:
        raise ValueError(f"[load_jsonl] No valid records loaded from: {path}")

    return rows

def load_orth_masks_all(
    orth_mask_path: str,
    vocab_size: int,
    only_model_ids: Optional[List[str]] = None,
) -> Dict[str, torch.BoolTensor]:
    """
    orth_mask JSON format:
    {
      "qwen-3b": {"green_ids":[...], ...},
      "llama-3b": {"green_ids":[...], ...},
      ...
    }
    """
    with open(orth_mask_path, "r", encoding="utf-8") as f:
        masks = json.load(f)

    out: Dict[str, torch.BoolTensor] = {}
    only_set = set(only_model_ids) if only_model_ids else None

    for mid, obj in masks.items():
        if only_set is not None and mid not in only_set:
            continue
        green_ids = obj.get("green_ids", [])
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        valid = [int(i) for i in green_ids if 0 <= int(i) < vocab_size]
        if not valid:
            continue
        mask[valid] = True
        out[mid] = mask

    if not out:
        raise ValueError(f"[orth-mask] No valid masks loaded from {orth_mask_path}")
    return out


def _stable_int_key(key_like: str) -> int:
    s = (key_like or "").encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little") & ((1 << 63) - 1)


def _load_keys_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"[keys] invalid or empty key json: {path}")
    return {str(k): str(v) for k, v in obj.items()}


def build_unigram_mask(vocab_size: int, fraction: float, watermark_key: int) -> torch.BoolTensor:
    if not (0.0 < float(fraction) < 1.0):
        raise ValueError(f"fraction must be in (0,1), got {fraction}")
    n_green = int(float(fraction) * int(vocab_size))
    if n_green <= 0:
        raise ValueError(f"n_green <= 0, vocab_size={vocab_size}, fraction={fraction}")

    mask = np.array([1] * n_green + [0] * (int(vocab_size) - n_green), dtype=np.uint8)
    key_int = int(watermark_key)
    if key_int < 0:
        key_int = -key_int
    key_bytes = key_int.to_bytes(max(1, (key_int.bit_length() + 7) // 8), "little", signed=False)
    seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], "little")
    np.random.default_rng(seed).shuffle(mask)
    return torch.from_numpy(mask.astype(np.bool_))

def parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# =========================
# Dataset
# =========================
@dataclass
class Sample:
    prompt: str
    target: str

class SummaryTeacherDataset(Dataset):
    """
    Code teacher-jsonl field compatibility:
      - reasoning_trace / deepseek_reasoning / reasoning
      - target_field (e.g., summary or summary_wm)
    Optional: filter by wm_z if available.
    """
    def __init__(
        self,
        tokenizer,
        rows: List[Dict[str, Any]],
        max_len: int,
        target_field: str = "summary",
        min_wm_z: float = -1e9,
        z_field: str = "wm_z",
        add_eos_to_target: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos_to_target = add_eos_to_target
        self.samples: List[Sample] = []

        for r in rows:
            rt = (r.get("reasoning_trace") or r.get("deepseek_reasoning") or r.get("reasoning") or "").strip()
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
            self.samples.append(Sample(prompt=prompt, target=tgt))

        if not self.samples:
            raise ValueError(f"No valid samples found. target_field={target_field}")

    def __len__(self):
        return len(self.samples)

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

        # truncate from the left (keep tail)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            attn = attn[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
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
    return batch


# =========================
# Trainer (compute_loss aligned with math: shift_logits/shift_labels)
# =========================
class WMTrainer(Trainer):
    """
    Extra losses (aligned with math script):
      1) wm_loss: -E[log P(G_pos | z_t)] on TARGET next-token positions
      2) sep_loss: hinge: logP(G_pos) >= logP(G_neg) + sep_margin
    Uses next-token alignment via shift_logits/shift_labels:
      shift_logits = logits[:, :-1, :]
      shift_labels = labels[:, 1:]
      mask_pos = (shift_labels != -100)
    """
    def __init__(
        self,
        *args,
        pos_green_idx: Optional[torch.LongTensor] = None,
        neg_green_indices: Optional[List[torch.LongTensor]] = None,
        wm_lambda: float = 0.0,
        sep_lambda: float = 0.0,
        neg_k: int = 2,
        sep_margin: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pos_green_idx = pos_green_idx
        self.neg_green_indices = neg_green_indices or []
        self.wm_lambda = float(wm_lambda)
        self.sep_lambda = float(sep_lambda)
        self.neg_k = int(neg_k)
        self.sep_margin = float(sep_margin)

    def _logp_green(self, logits: torch.Tensor, green_idx: torch.LongTensor) -> torch.Tensor:
        logZ = logits.logsumexp(dim=-1)  # [B,T]
        lg = torch.logsumexp(logits.index_select(dim=-1, index=green_idx), dim=-1)  # [B,T]
        return lg - logZ

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        base_loss = outputs.loss

        if (self.wm_lambda <= 0 and self.sep_lambda <= 0) or self.pos_green_idx is None:
            return (base_loss, outputs) if return_outputs else base_loss

        logits = outputs.logits          # [B,T,V]
        labels = inputs["labels"]        # [B,T]

        if logits.size(1) < 2:
            return (base_loss, outputs) if return_outputs else base_loss

        # next-token alignment (same as math)
        shift_logits = logits[:, :-1, :]     # [B,T-1,V]
        shift_labels = labels[:, 1:]         # [B,T-1]
        mask_pos = (shift_labels != -100)    # positions that predict TARGET tokens

        if not mask_pos.any():
            return (base_loss, outputs) if return_outputs else base_loss

        device = shift_logits.device
        pos_idx = self.pos_green_idx.to(device)

        # 1) wm_loss
        logp_pos = self._logp_green(shift_logits, pos_idx)  # [B,T-1]
        wm_loss = (-logp_pos[mask_pos]).mean()
        total = base_loss + self.wm_lambda * wm_loss

        # 2) sep_loss (sampled negatives)
        if self.sep_lambda > 0 and len(self.neg_green_indices) > 0 and self.neg_k > 0:
            k = min(self.neg_k, len(self.neg_green_indices))
            neg_list = random.sample(self.neg_green_indices, k=k)

            logp_neg_max = None
            for neg_idx_cpu in neg_list:
                neg_idx = neg_idx_cpu.to(device)
                lp = self._logp_green(shift_logits, neg_idx)  # [B,T-1]
                logp_neg_max = lp if logp_neg_max is None else torch.maximum(logp_neg_max, lp)

            sep = torch.relu(self.sep_margin + logp_neg_max - logp_pos)
            sep_loss = sep[mask_pos].mean()
            total = total + self.sep_lambda * sep_loss

        return (total, outputs) if return_outputs else total


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--teacher-jsonl", type=str, required=True)
    ap.add_argument("--orth-mask", type=str, default=None,
                    help="兼容旧接口：JSON: {model_id: {green_ids, ...}}")
    ap.add_argument("--key-config", type=str, default=None,
                    help="JSON: {model_id: secret_key}. Unigram mode.")
    ap.add_argument("--wm-key-direct", type=str, default=None,
                    help="可选：直接覆盖当前 model_id 的 key")
    ap.add_argument("--wm-key-int", type=int, default=None,
                    help="可选：直接指定当前 model 的 int key（不经字符串哈希）")
    ap.add_argument("--wm-fraction", type=float, default=0.25,
                    help="Unigram 水印 green token 比例")
    ap.add_argument("--out-adapter", type=str, required=True,
                    help="output root dir; script will create a hyperparam-tagged subdir")

    # sweep run id (avoid overwrite)
    ap.add_argument("--run-id", type=str, default="",
                    help="optional run identifier; if empty, use timestamp")

    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)

    # dataset label selection / filtering
    ap.add_argument("--target-field", type=str, default="summary",
                    help="teacher field used as label: e.g., summary or summary_wm")
    ap.add_argument("--min-wm-z", type=float, default=-1e9,
                    help="if teacher has wm_z, filter samples with wm_z < min_wm_z")
    ap.add_argument("--z-field", type=str, default="wm_z",
                    help="z-score field name (optional)")
    ap.add_argument("--add-eos-to-target", action="store_true",
                    help="append eos token to target (optional)")

    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # WM objectives
    ap.add_argument("--wm-lambda", type=float, default=0.0)
    ap.add_argument("--sep-lambda", type=float, default=0.0)
    ap.add_argument("--neg-k", type=int, default=2)
    ap.add_argument("--sep-margin", type=float, default=0.0)

    # Negatives selection
    ap.add_argument("--neg-model-ids", type=str, default="",
                    help="optional: comma-separated negative model-ids; if empty, use all others in config source")

    args = ap.parse_args()

    # seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # sweep outdir + logger
    outdir = build_outdir(
        base_out=args.out_adapter,
        model_id=args.model_id,
        wm_lambda=args.wm_lambda,
        sep_lambda=args.sep_lambda,
        run_id=args.run_id,
    )
    os.makedirs(outdir, exist_ok=True)

    logger = setup_logger(os.path.join(outdir, "train.log"))
    logger.info("======== Run Args ========")
    logger.info(json.dumps(vars(args), ensure_ascii=False, indent=2))
    logger.info("outdir = %s", outdir)

    # load tokenizer/model
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

    # LoRA
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

    # masks
    neg_candidates = parse_csv_list(args.neg_model_ids)
    if not args.key_config and not args.orth_mask:
        raise ValueError("One of --key-config or --orth-mask is required.")

    wm_scheme = "orth-mask"
    pos_key_int: Optional[int] = None
    neg_indices: List[torch.LongTensor] = []

    if args.orth_mask:
        masks_all = load_orth_masks_all(
            args.orth_mask,
            vocab_size=vocab_size,
            only_model_ids=(neg_candidates + [args.model_id]) if neg_candidates else None,
        )
        if args.model_id not in masks_all:
            raise KeyError(f"[orth-mask] missing model_id={args.model_id} in {args.orth_mask}")

        pos_mask = masks_all[args.model_id]
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1).long().cpu()
        if pos_idx.numel() == 0:
            raise ValueError("Empty pos green index set after loading orth-mask.")

        if neg_candidates:
            neg_ids = [m for m in neg_candidates if m != args.model_id and m in masks_all]
        else:
            neg_ids = [m for m in masks_all.keys() if m != args.model_id]

        for mid in neg_ids:
            m = masks_all[mid]
            idx = torch.nonzero(m, as_tuple=False).view(-1).long().cpu()
            if idx.numel() > 0:
                neg_indices.append(idx)
    else:
        wm_scheme = "unigram"
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

    if args.sep_lambda > 0 and len(neg_indices) == 0:
        raise ValueError("sep_lambda > 0 but no valid negative masks loaded. Check --neg-model-ids or config content.")

    # data
    rows = load_jsonl(args.teacher_jsonl)
    random.shuffle(rows)
    n = len(rows)
    if n == 0:
        raise ValueError(f"Empty teacher file: {args.teacher_jsonl}")

    n_train = int(n * args.train_ratio)
    if n_train <= 0 and n > 1:
        n_train = 1
    if n_train >= n and n > 1:
        n_train = n - 1

    train_rows = rows[:n_train]
    eval_rows = rows[n_train:] if n_train < n else rows[: max(1, n // 20)]

    train_ds = SummaryTeacherDataset(
        tok, train_rows, max_len=args.max_len,
        target_field=args.target_field,
        min_wm_z=args.min_wm_z,
        z_field=args.z_field,
        add_eos_to_target=args.add_eos_to_target,
    )
    eval_ds = SummaryTeacherDataset(
        tok, eval_rows, max_len=args.max_len,
        target_field=args.target_field,
        min_wm_z=args.min_wm_z,
        z_field=args.z_field,
        add_eos_to_target=args.add_eos_to_target,
    )

    logger.info("======== Data / Mask Stats ========")
    logger.info("teacher rows: n=%d, n_train=%d, n_eval=%d", n, len(train_rows), len(eval_rows))
    logger.info("dataset used: train=%d, eval=%d", len(train_ds), len(eval_ds))
    logger.info("%s: vocab_size=%d, pos_green=%d, neg_model_ids=%d, neg_masks=%d",
                wm_scheme, vocab_size, int(pos_idx.numel()), len(neg_ids), len(neg_indices))

    # write per-run + aggregate config (aligned with math)
    run_record = {
        "timestamp": _now_ts(),
        "outdir": outdir,
        "args": vars(args),
        "derived": {
            "wm_scheme": wm_scheme,
            "vocab_size": vocab_size,
            "wm_fraction": float(args.wm_fraction),
            "pos_key_int": int(pos_key_int) if pos_key_int is not None else None,
            "pos_green_size": int(pos_idx.numel()),
            "neg_model_ids": neg_ids,
            "num_neg_masks": len(neg_indices),
            "n_teacher_rows": n,
            "n_train_rows": len(train_rows),
            "n_eval_rows": len(eval_rows),
            "n_train_samples": len(train_ds),
            "n_eval_samples": len(eval_ds),
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    write_jsonl(os.path.join(outdir, "config.jsonl"), run_record, append=False)
    write_jsonl(os.path.join(args.out_adapter, "config.jsonl"), run_record, append=True)
    logger.info("wrote config: %s", os.path.join(outdir, "config.jsonl"))
    logger.info("appended aggregate config: %s", os.path.join(args.out_adapter, "config.jsonl"))

    # training args
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
        wm_lambda=args.wm_lambda,
        sep_lambda=args.sep_lambda,
        neg_k=args.neg_k,
        sep_margin=args.sep_margin,
    )

    logger.info("======== Training Start ========")
    trainer.train()
    logger.info("======== Training Done ========")

    # save adapter + tokenizer
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    logger.info("[done] saved adapter -> %s", outdir)
    print(f"[done] saved adapter -> {outdir}")


if __name__ == "__main__":
    main()
