#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_textclf_substitution_backbones.py
------------------------------------
Train a text-output-based model attribution baseline (multi-class classifier) and
evaluate model-substitution detection for directed pairs spec->alt.

This script supports "embedding backbone + lightweight classifier head" (BERT/T5/GPT2/LLM2Vec-style),
matching the "swap encoder backbone" ablation pattern shown in your table.

Input:
  - Multiple JSON or JSONL files (e.g., 4 files from 4 models).
  - Each sample should include:
      label_field: model_id (string, e.g., "qwen-3b")
      text_field : summary (string)  [or your summary field]
    Optional:
      split_by field (e.g., "id") to do prompt-disjoint hashing split.

Outputs (per directed pair spec->alt) on TEST:
  - AUC
  - TPR@FPR=1%/5%/10%  (thresholds calibrated on VAL, then applied to TEST)
  - ACC@FPR=1%/5%/10%

Also outputs:
  - multi-class attribution accuracy on TEST (top-1)

Example (evaluate a specific substitution):
  python run_textclf_substitution_backbones.py \
    --inputs qwen-7b.jsonl qwen-3b.jsonl llama-3b.jsonl llama-7b.jsonl \
    --label_field model_id --text_field summary --split_by id \
    --train_ratio 0.8 --val_ratio 0.1 \
    --backbone bert --backbone_name bert-base-uncased --pool cls \
    --out_dir ckpt/textclf_bert \
    --epochs 10 --batch 32 --lr 1e-3 \
    --pairs qwen-3b:qwen-7b,llama-3b:qwen-3b \
    --bf16

If --pairs is omitted, it evaluates all directed pairs among labels (K*(K-1)).

Notes:
  - For "embedding backbone + head" settings, recommended lr is ~1e-3 and epochs 5~20 (head-only).
  - If you enable --finetune_backbone, use lr ~1e-5~2e-5.
  - For GPT2, we set pad_token=eos_token and padding_side=left.
"""

import os
import json
import argparse
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import inspect

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    roc_auc_score = None
    roc_curve = None


# ---------------- I/O ----------------

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Robust reader for .jsonl:
      - tolerates garbage prefix before a JSON object (e.g., "..., -0.5{...}")
      - tolerates multiple JSON objects concatenated in one line
      - tolerates JSON objects accidentally split across multiple lines (buffers until parsable)
      - skips irrecoverable fragments but reports stats

    For .json:
      - supports list or dict with common wrapper keys.
    """
    if not path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k in ["data", "rows", "samples", "examples"]:
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
            return [obj]
        if isinstance(obj, list):
            return obj
        raise ValueError(f"Unsupported JSON structure in {path}: {type(obj)}")

    decoder = json.JSONDecoder()
    rows: List[Dict[str, Any]] = []

    # stats
    kept = 0
    dropped = 0
    garbage_prefix_dropped = 0
    multiline_joins = 0

    buf = ""
    buf_start_line = 1
    max_buf_chars = 2_000_000

    def _find_json_start(s: str) -> int:
        i1 = s.find("{")
        i2 = s.find("[")
        idxs = [i for i in (i1, i2) if i != -1]
        return min(idxs) if idxs else -1

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            if not buf:
                buf_start_line = ln

            buf += line

            if len(buf) > max_buf_chars:
                dropped += 1
                last_i1 = buf.rfind("{")
                last_i2 = buf.rfind("[")
                last_idxs = [i for i in (last_i1, last_i2) if i != -1]
                buf = buf[min(last_idxs):] if last_idxs else ""
                buf_start_line = ln
                continue

            while True:
                s = buf.lstrip()
                if s != buf:
                    buf = s

                start = _find_json_start(buf)
                if start == -1:
                    dropped += 1
                    buf = ""
                    break

                if start > 0:
                    garbage_prefix_dropped += 1
                    buf = buf[start:]

                try:
                    obj, end = decoder.raw_decode(buf)
                except json.JSONDecodeError as e:
                    if e.pos >= len(buf) - 2:
                        if ln != buf_start_line:
                            multiline_joins += 1
                        break

                    dropped += 1
                    buf = buf[1:]
                    continue
                else:
                    if isinstance(obj, dict):
                        rows.append(obj)
                        kept += 1
                    elif isinstance(obj, list):
                        for it in obj:
                            if isinstance(it, dict):
                                rows.append(it)
                                kept += 1
                            else:
                                dropped += 1
                    else:
                        dropped += 1

                    buf = buf[end:].lstrip()
                    if not buf:
                        break

    if buf.strip():
        try:
            obj, _ = decoder.raw_decode(buf.strip()[_find_json_start(buf.strip()):])
            if isinstance(obj, dict):
                rows.append(obj)
                kept += 1
            else:
                dropped += 1
        except Exception:
            dropped += 1

    if not rows:
        raise ValueError(f"No valid JSON objects parsed from {path} (kept=0, dropped={dropped}).")

    print(f"[read_json_or_jsonl] {path} | kept={kept} dropped={dropped} "
          f"garbage_prefix_dropped={garbage_prefix_dropped} multiline_joins={multiline_joins}")

    return rows


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- splitting ----------------

def stable_hash_to_unit_interval(s: str) -> float:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    x = int(h[:16], 16)
    return (x % (10**12)) / float(10**12)


def assign_split(row: Dict[str, Any], split_by: str, train_ratio: float, val_ratio: float) -> str:
    key_val = str(row.get(split_by, "")).strip()
    if not key_val:
        for k in ["id", "idx", "prompt_id", "prompt", "question"]:
            if k in row and str(row[k]).strip():
                key_val = str(row[k]).strip()
                break
    if not key_val:
        key_val = str(row.get("summary", ""))[:200]

    u = stable_hash_to_unit_interval(key_val)
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def extract_between_tags(text: str, open_tag: str, close_tag: str) -> str:
    if open_tag in text and close_tag in text:
        a = text.find(open_tag) + len(open_tag)
        b = text.find(close_tag, a)
        if b > a:
            return text[a:b].strip()
    return text.strip()


# ---------------- math helpers ----------------

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    out = []
    if not pairs_str.strip():
        return out
    for item in pairs_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad --pairs item: {item}. Use spec:alt")
        spec, alt = item.split(":", 1)
        out.append((spec.strip(), alt.strip()))
    return out


def get_all_directed_pairs(labels: List[str]) -> List[Tuple[str, str]]:
    return [(a, b) for a in labels for b in labels if a != b]


def tpr_fpr_acc_at_threshold(y_true: np.ndarray, score_attack: np.ndarray, thr: float) -> Tuple[float, float, float]:
    pred = (score_attack >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return float(tpr), float(fpr), float(acc)


def threshold_at_fpr_on_val(y_val: np.ndarray, score_attack_val: np.ndarray, target_fpr: float) -> float:
    """
    Choose a threshold on VAL that achieves FPR <= target_fpr (as large as possible).
    score_attack higher => more likely attack positive (ALT).
    """
    neg_scores = score_attack_val[y_val == 0]
    if len(neg_scores) == 0:
        return float(np.median(score_attack_val))

    if roc_curve is None:
        q = 1.0 - target_fpr
        return float(np.quantile(neg_scores, q))

    fpr, tpr, thr = roc_curve(y_val, score_attack_val)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return float(thr[0])
    return float(thr[idx[-1]])

# ---------------- attribution metrics (eval_attr-style) ----------------

def compute_confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    """
    eval_attr-style:
      - accuracy
      - macro_precision / macro_recall / macro_f1
      - confusion_matrix: rows=true, cols=pred
      - per_class: support/precision/recall/f1 + counts
    """
    B = len(labels)
    conf = [[0 for _ in range(B)] for _ in range(B)]

    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(t) < B and 0 <= int(p) < B:
            conf[int(t)][int(p)] += 1

    total = sum(sum(row) for row in conf)
    correct = sum(conf[i][i] for i in range(B))
    acc = float(correct / total) if total > 0 else 0.0

    per_class: Dict[str, Any] = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0

    for i, lab in enumerate(labels):
        tp = conf[i][i]
        pred_i = sum(conf[r][i] for r in range(B))  # column sum
        true_i = sum(conf[i][c] for c in range(B))  # row sum

        prec = float(tp / pred_i) if pred_i > 0 else 0.0
        rec = float(tp / true_i) if true_i > 0 else 0.0
        f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[lab] = {
            "support": int(true_i),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "pred_count": int(pred_i),
        }

        macro_p += prec
        macro_r += rec
        macro_f1 += f1

    if B > 0:
        macro_p /= B
        macro_r /= B
        macro_f1 /= B

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "confusion_matrix": conf,
        "per_class": per_class,
        "total_eval": int(total),
        "total_correct": int(correct),
        "skipped": 0,
        "labels": labels,
    }


def print_confusion(conf: List[List[int]], labels: List[str], max_label: int = 20) -> None:
    """
    Pretty-print confusion matrix (rows=true, cols=pred), matching eval_attr output style.
    """
    labs = [l[:max_label] for l in labels]
    colw = max(7, max(len(l) for l in labs) + 2)
    header = "true\\pred".ljust(colw) + "".join(l.ljust(colw) for l in labs)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(conf):
        line = labs[i].ljust(colw) + "".join(str(v).ljust(colw) for v in row)
        print(line)

# ---------------- dataset ----------------

@dataclass
class Ex:
    text: str
    y: int
    split: str
    label_name: str
    key: str  # split_by value for debugging


class TextDataset(Dataset):
    def __init__(self, examples: List[Ex], tokenizer, max_len: int):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.examples[i]
        enc = self.tok(ex.text, truncation=True, max_length=self.max_len)
        enc["labels"] = ex.y
        return enc


# ---------------- model: backbone + head ----------------

def pool_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    # last_hidden_state: [B, L, H], attention_mask: [B, L]
    if pool == "cls":
        return last_hidden_state[:, 0, :]
    if pool == "last":
        idx = attention_mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        return last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), idx, :]
    # mean pooling
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def get_last_hidden_from_outputs(outputs: Any) -> torch.Tensor:
    """
    Robustly fetch last_hidden_state for various backbones.
    """
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise ValueError("Cannot obtain last_hidden_state from backbone outputs.")


class BackboneClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        backbone_name: str,
        num_labels: int,
        pool: str = "mean",
        finetune_backbone: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pool = pool

        if backbone == "t5":
            self.encoder = T5EncoderModel.from_pretrained(backbone_name)
            
            hidden = int(self.encoder.config.d_model)
        else:
            self.encoder = AutoModel.from_pretrained(
                backbone_name,
                trust_remote_code=trust_remote_code
            )
            # some encoders store embedding dim under different names
            hidden = getattr(self.encoder.config, "hidden_size", None)
            if hidden is None:
                hidden = getattr(self.encoder.config, "d_model", None)
            if hidden is None:
                raise ValueError("Cannot infer hidden size from encoder config.")
        self._encoder_fwd_keys = set(inspect.signature(self.encoder.forward).parameters.keys())
        self.classifier = nn.Linear(int(hidden), int(num_labels))

        if not finetune_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)

        enc_kwargs = {k: v for k, v in kwargs.items() if k in self._encoder_fwd_keys}
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **enc_kwargs)
        h = get_last_hidden_from_outputs(outputs)
        pooled = pool_hidden(h, attention_mask, pool=self.pool)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


@torch.no_grad()
def predict_logits(model: nn.Module, tokenizer, texts: List[str], max_len: int, batch: int, device: str) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        outs.append(logits)
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="Multiple .json or .jsonl files.")
    ap.add_argument("--text_field", type=str, default="summary")
    ap.add_argument("--label_field", type=str, default="model_id")
    ap.add_argument("--split_by", type=str, default="id")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--strip_summary_tags", action="store_true")
    ap.add_argument("--summary_open_tag", type=str, default="<summary>")
    ap.add_argument("--summary_close_tag", type=str, default="</summary>")

    # backbone selection (table-style ablations)
    ap.add_argument("--backbone", type=str, choices=["bert", "t5", "gpt2", "llm2vec"], default="bert")
    ap.add_argument("--backbone_name", type=str, default="bert-base-uncased")
    ap.add_argument("--pool", type=str, choices=["mean", "cls", "last"], default="mean")
    ap.add_argument("--finetune_backbone", action="store_true", help="If set, do not freeze backbone.")
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for backbone loading.")

    # training
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)

    # evaluation
    ap.add_argument("--pairs", type=str, default="", help="Directed pairs spec:alt,... If empty, evaluate all pairs.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_jsonl", type=str, default="", help="Optional: save per-example scores for pair subsets.")
    ap.add_argument("--eval_only", action="store_true", help="Skip training; load model/tokenizer from --out_dir.")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- load rows -----
    raw_rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        raw_rows.extend(read_json_or_jsonl(p))

    # ----- build pre-examples with split -----
    tmp: List[Tuple[str, str, str, str]] = []  # (label, text, split, key)
    for r in raw_rows:
        lab = r.get(args.label_field, None)
        txt = r.get(args.text_field, None)
        if lab is None or txt is None:
            continue
        lab = str(lab).strip()
        txt = str(txt).strip()
        if not lab or not txt:
            continue

        if args.strip_summary_tags:
            txt = extract_between_tags(txt, args.summary_open_tag, args.summary_close_tag)

        split = assign_split(r, args.split_by, args.train_ratio, args.val_ratio)
        key = str(r.get(args.split_by, "")).strip()
        tmp.append((lab, txt, split, key))

    if not tmp:
        raise ValueError("No valid samples after filtering. Check --label_field/--text_field.")

    # ----- label set from TRAIN split -----
    train_labels = sorted({lab for (lab, _, sp, _) in tmp if sp == "train"})
    if not train_labels:
        raise ValueError("No labels found in train split. Check split configuration.")
    label2id = {lab: i for i, lab in enumerate(train_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Validate each label appears in each split (at least 1), otherwise training/eval may be unstable
    split_counts = {lab: {"train": 0, "val": 0, "test": 0} for lab in train_labels}
    for lab, _, sp, _ in tmp:
        if lab in split_counts:
            split_counts[lab][sp] += 1

    # Convert to Ex
    examples: List[Ex] = []
    for lab, txt, sp, key in tmp:
        if lab not in label2id:
            continue
        examples.append(Ex(text=txt, y=label2id[lab], split=sp, label_name=lab, key=key))

    train_ex = [e for e in examples if e.split == "train"]
    val_ex = [e for e in examples if e.split == "val"]
    test_ex = [e for e in examples if e.split == "test"]

    if len(train_ex) == 0 or len(val_ex) == 0 or len(test_ex) == 0:
        raise ValueError(f"Split empty: train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}. "
                         f"Adjust ratios or ensure {args.split_by} exists.")

    # Save label mapping + split stats
    write_json(os.path.join(args.out_dir, "label2id.json"), label2id)
    write_json(os.path.join(args.out_dir, "id2label.json"), id2label)
    write_json(os.path.join(args.out_dir, "split_counts.json"), split_counts)

    # ----- tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_name, use_fast=True, trust_remote_code=args.trust_remote_code)

    # GPT2 padding handling
    if args.backbone == "gpt2":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Datasets / collator
    train_ds = TextDataset(train_ex, tokenizer, args.max_len)
    val_ds = TextDataset(val_ex, tokenizer, args.max_len)
    test_ds = TextDataset(test_ex, tokenizer, args.max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----- model -----
    if args.eval_only:
        # Load from out_dir
        model = torch.load(os.path.join(args.out_dir, "model.pt"), map_location="cpu")
    else:
        model = BackboneClassifier(
            backbone=args.backbone,
            backbone_name=args.backbone_name,
            num_labels=len(label2id),
            pool=args.pool,
            finetune_backbone=args.finetune_backbone,
            trust_remote_code=args.trust_remote_code,
        )

    # metrics for training selection
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        acc = float((pred == labels).mean())
        return {"accuracy": acc}

    # ----- training -----
    device = args.device
    model = model.to(device)

    if not args.eval_only:
        targs = TrainingArguments(
            output_dir=args.out_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            save_safetensors=False, 
            save_total_limit=1,
            save_only_model=True,
            logging_steps=50,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=0.01 if args.finetune_backbone else 0.0,
            warmup_ratio=0.06,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            bf16=args.bf16,
            fp16=args.fp16,
            dataloader_num_workers=args.num_workers,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # Save best model in a simple way (portable without relying on HF save_pretrained for custom nn.Module)
        best_model = trainer.model.to("cpu")
        torch.save(best_model, os.path.join(args.out_dir, "model.pt"))
        tokenizer.save_pretrained(args.out_dir)
        model = trainer.model.to(device)
        model.eval()
    else:
        model.eval()

    # ----- inference on VAL / TEST -----
    val_texts = [e.text for e in val_ex]
    val_y = np.array([e.y for e in val_ex], dtype=np.int64)
    test_texts = [e.text for e in test_ex]
    test_y = np.array([e.y for e in test_ex], dtype=np.int64)

    val_logits = predict_logits(model, tokenizer, val_texts, args.max_len, args.batch, device)
    test_logits = predict_logits(model, tokenizer, test_texts, args.max_len, args.batch, device)
    val_probs = softmax(val_logits, axis=-1)
    test_probs = softmax(test_logits, axis=-1)

    test_pred = np.argmax(test_probs, axis=-1)

    # eval_attr-style attribution summary
    all_labels = list(label2id.keys())  # label order matches class indices
    attrib_summary = compute_confusion_and_metrics(test_y, test_pred, all_labels)

    # keep a convenient scalar for older logic
    attribution_acc = float(attrib_summary["accuracy"])

    # ----- pair evaluation -----
    requested_pairs = parse_pairs(args.pairs)
    all_labels = list(label2id.keys())
    if not requested_pairs:
        requested_pairs = get_all_directed_pairs(all_labels)

    per_example_dump: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for spec, alt in requested_pairs:
        if spec not in label2id or alt not in label2id:
            results.append({
                "setting": f"{spec}->{alt}",
                "declared_model": spec,
                "actual_model_positive": alt,
                "skipped": True,
                "reason": "spec_or_alt_not_in_train_labels",
                "_summary": {"attribution_acc_test": attribution_acc},
            })
            continue

        spec_id = label2id[spec]
        alt_id = label2id[alt]

        # VAL subset for threshold calibration
        val_idx = np.where((val_y == spec_id) | (val_y == alt_id))[0]
        if len(val_idx) < 10 or len(np.unique((val_y[val_idx] == alt_id).astype(np.int64))) < 2:
            results.append({
                "setting": f"{spec}->{alt}",
                "declared_model": spec,
                "actual_model_positive": alt,
                "skipped": True,
                "reason": "too_few_or_single_class_val_samples_for_pair",
                "total_val_pair": int(len(val_idx)),
                "_summary": {"attribution_acc_test": attribution_acc},
            })
            continue

        y_val_bin = (val_y[val_idx] == alt_id).astype(np.int64)  # 1 if ALT (attack)
        p_val = val_probs[val_idx]
        eps = 1e-12
        logp_val = np.log(np.clip(p_val, eps, 1.0))
        logp_val_non_spec = logp_val.copy()
        logp_val_non_spec[:, spec_id] = -1e9
        max_other_val = np.max(logp_val_non_spec, axis=-1)
        margin_spec_val = logp_val[:, spec_id] - max_other_val
        score_attack_val = -margin_spec_val

        thr_1 = threshold_at_fpr_on_val(y_val_bin, score_attack_val, 0.01)
        thr_5 = threshold_at_fpr_on_val(y_val_bin, score_attack_val, 0.05)
        thr_10 = threshold_at_fpr_on_val(y_val_bin, score_attack_val, 0.10)

        # TEST subset for reporting
        test_idx = np.where((test_y == spec_id) | (test_y == alt_id))[0]
        if len(test_idx) < 10 or len(np.unique((test_y[test_idx] == alt_id).astype(np.int64))) < 2:
            results.append({
                "setting": f"{spec}->{alt}",
                "declared_model": spec,
                "actual_model_positive": alt,
                "skipped": True,
                "reason": "too_few_or_single_class_test_samples_for_pair",
                "total_test_pair": int(len(test_idx)),
                "_summary": {"attribution_acc_test": attribution_acc},
            })
            continue

        y_test_bin = (test_y[test_idx] == alt_id).astype(np.int64)
        p_test = test_probs[test_idx]
        logp_test = np.log(np.clip(p_test, eps, 1.0))
        logp_test_non_spec = logp_test.copy()
        logp_test_non_spec[:, spec_id] = -1e9
        max_other_test = np.max(logp_test_non_spec, axis=-1)
        margin_spec_test = logp_test[:, spec_id] - max_other_test
        score_attack_test = -margin_spec_test

        auc = None
        if roc_auc_score is not None and len(np.unique(y_test_bin)) == 2:
            auc = float(roc_auc_score(y_test_bin, score_attack_test))

        tpr1, fpr1, acc1 = tpr_fpr_acc_at_threshold(y_test_bin, score_attack_test, thr_1)
        tpr5, fpr5, acc5 = tpr_fpr_acc_at_threshold(y_test_bin, score_attack_test, thr_5)
        tpr10, fpr10, acc10 = tpr_fpr_acc_at_threshold(y_test_bin, score_attack_test, thr_10)

        out = {
            "setting": f"{spec}->{alt}",
            "declared_model": spec,
            "actual_model_positive": alt,
            "total_pos_eval": int((y_test_bin == 1).sum()),
            "total_neg_eval": int((y_test_bin == 0).sum()),
            "AUC": auc,

            "TPR@FPR=1%": tpr1,
            "TPR@FPR=5%": tpr5,
            "TPR@FPR=10%": tpr10,

            "ACC@FPR=1%": acc1,
            "ACC@FPR=5%": acc5,
            "ACC@FPR=10%": acc10,

            # realized FPR on TEST (sanity check)
            "FPR@thr(FPR=1%)": fpr1,
            "FPR@thr(FPR=5%)": fpr5,
            "FPR@thr(FPR=10%)": fpr10,

            # thresholds chosen on VAL
            "thr@FPR=1% (val)": float(thr_1),
            "thr@FPR=5% (val)": float(thr_5),
            "thr@FPR=10% (val)": float(thr_10),

            "_summary": {
                "attribution_acc_test": attribution_acc,
                "num_train": len(train_ex),
                "num_val": len(val_ex),
                "num_test": len(test_ex),
                "backbone": args.backbone,
                "backbone_name": args.backbone_name,
                "pool": args.pool,
                "finetune_backbone": bool(args.finetune_backbone),
                "text_field": args.text_field,
                "label_field": args.label_field,
                "split_by": args.split_by,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
            }
        }
        results.append(out)

        if args.save_jsonl:
            for k, idx in enumerate(test_idx):
                per_example_dump.append({
                    "setting": f"{spec}->{alt}",
                    "claimed": spec,
                    "true_model": id2label[int(test_y[idx])],
                    "is_attack_positive": int(y_test_bin[k]),
                    "score_attack": float(score_attack_test[k]),
                    "p_spec": float(p_test[k, spec_id]),
                    "p_alt": float(p_test[k, alt_id]),
                    "pred_model": id2label[int(np.argmax(p_test[k]))],
                    "text_len_chars": len(test_texts[idx]),
                })

    # ----- save outputs -----
    write_json(os.path.join(args.out_dir, "metrics_attribution.json"), {
        **attrib_summary,  # accuracy, macro_*, confusion_matrix, per_class, total_eval, skipped, labels
        "_meta": {
            "num_train": len(train_ex),
            "num_val": len(val_ex),
            "num_test": len(test_ex),
            "backbone": args.backbone,
            "backbone_name": args.backbone_name,
            "pool": args.pool,
            "finetune_backbone": bool(args.finetune_backbone),
            "text_field": args.text_field,
            "label_field": args.label_field,
            "split_by": args.split_by,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        }
    })
    write_json(os.path.join(args.out_dir, "metrics_pairs.json"), results)
    print(json.dumps({
        "accuracy": attrib_summary["accuracy"],
        "macro_precision": attrib_summary["macro_precision"],
        "macro_recall": attrib_summary["macro_recall"],
        "macro_f1": attrib_summary["macro_f1"],
        "total_eval": attrib_summary["total_eval"],
        "skipped": attrib_summary["skipped"],
    }, ensure_ascii=False, indent=2))

    print("\n[confusion matrix] rows=true, cols=pred")
    print_confusion(attrib_summary["confusion_matrix"], attrib_summary["labels"])

    for r in results:
        print(json.dumps(r, ensure_ascii=False))

    if args.save_jsonl:
        write_jsonl(args.save_jsonl, per_example_dump)


if __name__ == "__main__":
    main()
