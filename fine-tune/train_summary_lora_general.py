#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, random, time, logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# PEFT (LoRA)
import peft
from peft import LoraConfig, get_peft_model


TAG_SUM_OPEN = "<summary>"
TAG_SUM_CLOSE = "</summary>"

PROMPT_TEMPLATE_SUM_FROM_RT = (
    "You are a helpful assistant. Given the following detailed reasoning trace for solving a problem, "
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


# ----------------- logging & sweep utils (aligned with math) -----------------
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
    logger = logging.getLogger("wm_train_general")
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


# ----------------- core helpers -----------------
def apply_chat_template(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

def load_orth_masks_all(
    orth_mask_path: str,
    vocab_size: int,
    only_model_ids: Optional[List[str]] = None,
) -> Dict[str, torch.BoolTensor]:
    with open(orth_mask_path, "r", encoding="utf-8") as f:
        masks = json.load(f)

    out: Dict[str, torch.BoolTensor] = {}
    allow = set(only_model_ids) if only_model_ids is not None else None

    for mid, obj in masks.items():
        if allow is not None and mid not in allow:
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

def parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# ----------------- Dataset -----------------
@dataclass
class Sample:
    prompt: str
    target: str

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
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos_to_target = add_eos_to_target
        self.samples: List[Sample] = []

        def _get_rt(r: Dict[str, Any]) -> str:
            for k in ["reasoning_trace", "deepseek_reasoning", "reasoning", "rt", "trace", "rationale", "cot", "think"]:
                v = r.get(k, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def _get_tgt(r: Dict[str, Any]) -> str:
            v = r.get(target_field, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
            for k in ["summary_wm", "summary", "summary_teacher", "teacher_summary", "output", "completion"]:
                v2 = r.get(k, None)
                if isinstance(v2, str) and v2.strip():
                    return v2.strip()
            return ""

        for r in rows:
            rt = _get_rt(r)
            tgt = _get_tgt(r)
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

        # truncate (keep tail)
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


# ----------------- Trainer (compute_loss aligned with math: shift_labels = labels[:,1:]) -----------------
class WMTrainer(Trainer):
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

        shift_logits = logits[:, :-1, :]     # [B,T-1,V]
        shift_labels = labels[:,  1:]        # [B,T-1]
        mask_pos = (shift_labels != -100)    # only positions that predict TARGET tokens

        if not mask_pos.any():
            return (base_loss, outputs) if return_outputs else base_loss

        device = shift_logits.device
        pos_idx = self.pos_green_idx.to(device)

        logp_pos = self._logp_green(shift_logits, pos_idx)  # [B,T-1]
        wm_loss = (-logp_pos[mask_pos]).mean()
        total = base_loss + self.wm_lambda * wm_loss

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


def main():
    ap = argparse.ArgumentParser()

    # ===== keep interface, but make out-adapter a root dir for sweeps =====
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--teacher-jsonl", type=str, required=True)
    ap.add_argument("--orth-mask", type=str, required=True)
    ap.add_argument("--out-adapter", type=str, required=True,
                    help="输出根目录（脚本会自动在其下创建含超参标识的子目录）")

    # NEW: sweep run id
    ap.add_argument("--run-id", type=str, default="",
                    help="可选：用于输出目录的 run 标识；为空则用时间戳，避免覆盖")

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
                    help="可选：显式指定负采样候选 model-id，用逗号分隔。为空则用 orth-mask 内除自身外的全部。")

    args = ap.parse_args()

    # seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # NEW: sweep outdir + logger
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

    if not os.path.isfile(args.teacher_jsonl):
        raise FileNotFoundError(f"teacher-jsonl not found: {args.teacher_jsonl}")

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

    # load all masks
    neg_candidates = parse_csv_list(args.neg_model_ids)
    masks_all = load_orth_masks_all(
        args.orth_mask,
        vocab_size=vocab_size,
        only_model_ids=(neg_candidates + [args.model_id]) if neg_candidates else None
    )

    if args.model_id not in masks_all:
        raise KeyError(f"[orth-mask] missing model_id={args.model_id} in {args.orth_mask}")

    pos_mask = masks_all[args.model_id]
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1).long().cpu()
    if pos_idx.numel() == 0:
        raise ValueError("Empty pos green index set after loading orth-mask.")

    # negative masks indices
    if neg_candidates:
        neg_ids = [m for m in neg_candidates if m != args.model_id and m in masks_all]
    else:
        neg_ids = [m for m in masks_all.keys() if m != args.model_id]

    neg_indices: List[torch.LongTensor] = []
    for mid in neg_ids:
        m = masks_all[mid]
        idx = torch.nonzero(m, as_tuple=False).view(-1).long().cpu()
        if idx.numel() > 0:
            neg_indices.append(idx)

    if args.sep_lambda > 0 and len(neg_indices) == 0:
        raise ValueError("sep_lambda > 0 but no valid negative masks loaded. Check --neg-model-ids or orth-mask content.")

    rows = load_jsonl(args.teacher_jsonl)
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
    logger.info("vocab_size=%d, pos_green=%d, neg_model_ids=%d, neg_masks=%d",
                vocab_size, int(pos_idx.numel()), len(neg_ids), len(neg_indices))

    # NEW: write per-run config + root aggregate config
    run_record = {
        "timestamp": _now_ts(),
        "outdir": outdir,
        "args": vars(args),
        "derived": {
            "vocab_size": vocab_size,
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
