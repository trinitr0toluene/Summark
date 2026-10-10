#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
two_stage_student_summaries.py  (ROBUST general reasoning datasets; JSONL-in/JSONL-out)
------------------------------------------------------------------------------
Robust upgrades:
- Stage1: robust parsing + fallback to avoid empty reasoning_trace/final_answer.
- Stage2: robust input fallback + guarantee non-empty summary_student for every row.
- Optional flags: --rt-greedy / --sum-greedy to improve format-following for small models.
"""

import os, json, re, argparse, random
from typing import Dict, Any, Optional, List, Tuple
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TAG_SUM_OPEN = "<summary>"
TAG_SUM_CLOSE = "</summary>"

TAG_RT_OPEN = "<reasoning>"
TAG_RT_CLOSE = "</reasoning>"
TAG_FINAL_OPEN = "<final>"
TAG_FINAL_CLOSE = "</final>"

# ===== Stage 1: generate reasoning trace + final answer (NO watermark) =====
PROMPT_TEMPLATE_RT_AND_FINAL = (
    "You are a helpful assistant. Solve the following problem.\n\n"
    "Requirements:\n"
    "1) Write a detailed reasoning trace.\n"
    "2) Then provide the final answer.\n\n"
    f"Write the reasoning trace inside {TAG_RT_OPEN} ... {TAG_RT_CLOSE}.\n"
    f"Write the final answer inside {TAG_FINAL_OPEN} ... {TAG_FINAL_CLOSE}.\n"
    f"Respond with exactly two blocks in this order:\n"
    f"{TAG_RT_OPEN}...{TAG_RT_CLOSE}\n"
    f"{TAG_FINAL_OPEN}...{TAG_FINAL_CLOSE}\n\n"
    "Problem:\n{q}\n"
)

# ===== Stage 2: generate summary from reasoning trace (watermark via adapter here) =====
PROMPT_TEMPLATE_SUM_FROM_RT = (
    "You are a helpful assistant. Given the following detailed reasoning trace for solving a problem, "
    "write a concise but faithful reasoning summary in about 180–300 English words. "
    "The summary must NOT be shorter than 160 words. Focus on the main logical steps and key intermediate results.\n\n"
    "Reasoning trace:\n{rt}\n\n"
    "Organize the summary into several short paragraphs. For each paragraph:\n"
    " - Start with ONE clear topic sentence that states the main purpose of this part of the reasoning "
    "   (for example: clarifying assumptions, setting up notation, transforming the problem, applying a key idea, "
    "   computing a key quantity, handling edge cases, or checking the final result).\n"
    " - Then add 1–3 additional sentences that briefly explain the essential reasoning details for this part.\n"
    "Do not use numbered lists or bullet points. Do not restate the full question or repeat the final answer verbatim; "
    "just explain how the solution proceeds.\n\n"
    f"Write everything inside {TAG_SUM_OPEN} ... {TAG_SUM_CLOSE} as continuous prose with multiple paragraphs.\n"
    f"Respond with exactly one block: {TAG_SUM_OPEN}...{TAG_SUM_CLOSE}."
)

# ---------------- Basic helpers ----------------

def apply_chat_template(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_between(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    m = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), text, flags=re.S)
    return m.group(1).strip() if m else None

def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""

def _ensure_nonempty(s: str, fallback: str) -> str:
    s = _safe_str(s).strip()
    if s:
        return s
    fb = _safe_str(fallback).strip()
    return fb if fb else "N/A"

def parse_stage1_outputs(text: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Robust Stage1 parser.
    Returns: (reasoning_trace, final_answer, parse_meta)

    Priority:
      1) strict <reasoning>...</reasoning> and <final>...</final>
      2) relaxed: if </reasoning> missing, use <final> as boundary
      3) fallback: if no tags, use whole raw as reasoning; final=last non-empty line
    """
    raw = _safe_str(text).strip()

    rt_strict = extract_between(raw, TAG_RT_OPEN, TAG_RT_CLOSE)
    fa_strict = extract_between(raw, TAG_FINAL_OPEN, TAG_FINAL_CLOSE)

    meta = {
        "has_reasoning_open": (TAG_RT_OPEN in raw),
        "has_reasoning_close": (TAG_RT_CLOSE in raw),
        "has_final_open": (TAG_FINAL_OPEN in raw),
        "has_final_close": (TAG_FINAL_CLOSE in raw),
        "used_fallback_reasoning": False,
        "used_fallback_final": False,
    }

    # reasoning fallback
    rt = _safe_str(rt_strict).strip()
    if not rt:
        meta["used_fallback_reasoning"] = True
        if TAG_RT_OPEN in raw:
            after = raw.split(TAG_RT_OPEN, 1)[1]
            if TAG_RT_CLOSE in after:
                rt = after.split(TAG_RT_CLOSE, 1)[0].strip()
            elif TAG_FINAL_OPEN in after:
                rt = after.split(TAG_FINAL_OPEN, 1)[0].strip()
            else:
                rt = after.strip()
        else:
            if TAG_FINAL_OPEN in raw:
                rt = raw.split(TAG_FINAL_OPEN, 1)[0].strip()
            else:
                rt = raw

    rt = _ensure_nonempty(rt, raw)

    # final fallback
    fa = _safe_str(fa_strict).strip()
    if not fa:
        meta["used_fallback_final"] = True
        if TAG_FINAL_OPEN in raw:
            after = raw.split(TAG_FINAL_OPEN, 1)[1]
            if TAG_FINAL_CLOSE in after:
                fa = after.split(TAG_FINAL_CLOSE, 1)[0].strip()
            else:
                fa = after.strip()
        else:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            fa = lines[-1] if lines else ""

    fa = _safe_str(fa).strip()
    return rt, fa, meta

# ---------------- JSONL robust loader (keep your safe loader) ----------------

_VALID_ESC = set('"\\/bfnrtu')

def _escape_invalid_backslashes(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]

        if nxt in _VALID_ESC and nxt != "u":
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        if nxt == "u":
            if i + 6 <= n:
                hexpart = s[i + 2 : i + 6]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hexpart):
                    out.append("\\u" + hexpart)
                    i += 6
                    continue
            out.append("\\\\")
            i += 1
            continue

        out.append("\\\\")
        i += 1

    return "".join(out)

def load_jsonl_safe(path: str, max_bad: int = 50) -> List[Dict[str, Any]]:
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                fixed = _escape_invalid_backslashes(line)
                try:
                    rows.append(json.loads(fixed))
                except json.JSONDecodeError:
                    bad += 1
                    if bad <= max_bad:
                        pos = e.pos
                        ctx = line[max(0, pos - 60) : pos + 60]
                        print(f"[warn] skip bad json line={ln} pos={pos} msg={e.msg}", file=sys.stderr)
                        print(f"       context: {ctx}", file=sys.stderr)
                    if bad >= max_bad:
                        print(f"[warn] too many bad lines (>= {max_bad}), stop reading.", file=sys.stderr)
                        break
    if bad > 0:
        print(f"[warn] total bad lines skipped: {bad}", file=sys.stderr)
    return rows

def split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_train = max(0, min(n_train, n))
    train_rows = rows[:n_train]
    eval_rows = rows[n_train:]
    if len(eval_rows) == 0 and n > 0:
        eval_rows = rows[-max(1, n // 20) :]
        train_rows = rows[: max(0, n - len(eval_rows))]
    return train_rows, eval_rows

def _as_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip()
        return t if t else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, dict):
        for k in ["text", "content", "prompt", "question"]:
            if k in v and isinstance(v[k], str) and v[k].strip():
                return v[k].strip()
        return None
    if isinstance(v, list):
        parts: List[str] = []
        for item in v:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                c = item.get("content", None)
                role = item.get("role", None)
                if isinstance(c, str) and c.strip():
                    if role is None or role == "user":
                        parts.append(c.strip())
        txt = "\n".join(parts).strip()
        return txt if txt else None
    return None

def pick_question(r: Dict[str, Any]) -> Optional[str]:
    inst = _as_text(r.get("instruction"))
    inp = _as_text(r.get("input"))
    if inst and inp:
        return f"{inst}\n\n{inp}".strip()
    if inst:
        return inst

    for k in ["question", "prompt", "problem", "query", "task", "text", "q"]:
        v = _as_text(r.get(k))
        if v:
            return v

    for k in ["source_prompt", "src_prompt", "user_prompt"]:
        v = _as_text(r.get(k))
        if v:
            return v

    for k in ["messages", "conversations", "chat", "dialogue"]:
        v = _as_text(r.get(k))
        if v:
            return v

    return None

def generate_text(model, tok, prompt: str, max_new: int, temperature: float, top_p: float,
                  do_sample: bool = True) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    gen_only = out[:, inputs["input_ids"].shape[-1] :]
    return tok.batch_decode(gen_only, skip_special_tokens=True)[0]

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _build_stage2_input(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pick best available input for Stage2.
    Returns: (text_for_summary, source_name)
    """
    rt = _safe_str(rec.get("reasoning_trace")).strip()
    if rt:
        return rt, "reasoning_trace"

    raw = _safe_str(rec.get("_stage1_raw")).strip()
    if raw:
        return raw, "_stage1_raw"

    teacher_sum = _safe_str(rec.get("summary")).strip()
    if teacher_sum:
        return teacher_sum, "summary(teacher)"

    q = _safe_str(pick_question(rec)).strip()
    fa = _safe_str(rec.get("final_answer")).strip()
    combo = f"Problem:\n{q}\n\nFinal answer:\n{fa}".strip()
    if combo:
        return combo, "question+final"

    return "", "none"

# ---------------- Stage runners ----------------

def run_stage1_on_row(rec: Dict[str, Any], base, tok, args) -> Dict[str, Any]:
    q = pick_question(rec)
    if not q:
        rec["_stage1_error"] = "missing question/prompt/problem/messages field"
        # still set safe fallbacks to allow stage2
        rec["_stage1_raw"] = _ensure_nonempty(_safe_str(rec.get("_stage1_raw")), "")
        rec["reasoning_trace"] = _ensure_nonempty(_safe_str(rec.get("reasoning_trace")), rec["_stage1_raw"])
        rec["final_answer"] = _safe_str(rec.get("final_answer")).strip()
        rec["_stage1_parse"] = {"skipped_generation": True}
    else:
        user_text = PROMPT_TEMPLATE_RT_AND_FINAL.format(q=q)
        prompt = apply_chat_template(tok, user_text)
        text = generate_text(
            model=base,
            tok=tok,
            prompt=prompt,
            max_new=args.rt_max_new,
            temperature=args.rt_temperature,
            top_p=args.rt_top_p,
            do_sample=not args.rt_greedy,
        )

        rec["_stage1_raw"] = _safe_str(text).strip()

        rt, fa, parse_meta = parse_stage1_outputs(rec["_stage1_raw"])
        rec["reasoning_trace"] = rt
        rec["final_answer"] = fa
        rec["_stage1_parse"] = parse_meta

    rec["_stage1_meta"] = {
        "base_model": args.base_model,
        "rt_max_new": args.rt_max_new,
        "rt_temperature": args.rt_temperature,
        "rt_top_p": args.rt_top_p,
        "rt_greedy": bool(args.rt_greedy),
        "stage1_watermark": "off",
        "stage1_policy": "always_regenerate",
        "task": "general",
        "tags": {"rt": [TAG_RT_OPEN, TAG_RT_CLOSE], "final": [TAG_FINAL_OPEN, TAG_FINAL_CLOSE]},
    }
    return rec

def run_stage2_on_row(rec: Dict[str, Any], wm_model, tok, args) -> Dict[str, Any]:
    rt_input, src = _build_stage2_input(rec)

    if not rt_input.strip():
        rec["_stage2_error"] = "missing_all_inputs_for_summary"
        rec["summary_student"] = "N/A"
    else:
        user_text = PROMPT_TEMPLATE_SUM_FROM_RT.format(rt=rt_input)
        prompt = apply_chat_template(tok, user_text)
        try:
            text = generate_text(
                model=wm_model,
                tok=tok,
                prompt=prompt,
                max_new=args.sum_max_new,
                temperature=args.sum_temperature,
                top_p=args.sum_top_p,
                do_sample=not args.sum_greedy,
            )
            text = _safe_str(text).strip()
        except Exception as e:
            rec["_stage2_error"] = f"generation_exception: {type(e).__name__}"
            text = ""

        summary = extract_between(text, TAG_SUM_OPEN, TAG_SUM_CLOSE) if text else None
        summary = _ensure_nonempty(summary or "", text)

        # hard guarantee: make it non-empty so it always enters eval
        if not summary.strip():
            summary = "N/A"
        rec["summary_student"] = summary

        if src != "reasoning_trace":
            rec["_stage2_warn"] = f"used_fallback_input={src}"

    rec["_stage2_meta"] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "sum_max_new": args.sum_max_new,
        "sum_temperature": args.sum_temperature,
        "sum_top_p": args.sum_top_p,
        "sum_greedy": bool(args.sum_greedy),
        "stage2_watermark": "on" if (args.adapter and args.adapter.lower() not in ["none", "null", "no", ""]) else "off",
        "task": "general",
        "tags": {"summary": [TAG_SUM_OPEN, TAG_SUM_CLOSE]},
        "stage2_input_source": rec.get("_stage2_warn", "reasoning_trace"),
    }
    return rec

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="none",
                    help="阶段2使用的 LoRA adapter（用于summary加水印）。传 none 表示不用。")
    ap.add_argument("--in-jsonl", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--stage", type=str, choices=["1", "2", "both"], default="both")

    ap.add_argument("--rt-max-new", type=int, default=2048)
    ap.add_argument("--rt-temperature", type=float, default=0.7)
    ap.add_argument("--rt-top-p", type=float, default=0.95)
    ap.add_argument("--rt-greedy", action="store_true",
                    help="Stage1 使用 greedy 解码（do_sample=False），小模型更容易按 tag 输出。")

    ap.add_argument("--sum-max-new", type=int, default=512)
    ap.add_argument("--sum-temperature", type=float, default=0.7)
    ap.add_argument("--sum-top-p", type=float, default=0.95)
    ap.add_argument("--sum-greedy", action="store_true",
                    help="Stage2 使用 greedy 解码（do_sample=False）。")

    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--split", type=str, choices=["eval", "train", "all"], default="eval")
    ap.add_argument("--train-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # determinism for split + some generation stability
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) load tokenizer & base model
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base.config.pad_token_id = tok.pad_token_id
    base.eval()

    # 2) load & split rows
    rows = load_jsonl_safe(args.in_jsonl)
    if not rows:
        raise ValueError(f"[ERROR] empty input jsonl: {args.in_jsonl}")

    train_rows, eval_rows = split_rows(rows, train_ratio=args.train_ratio, seed=args.seed)
    if args.split == "train":
        use_rows = train_rows
    elif args.split == "eval":
        use_rows = eval_rows
    else:
        use_rows = rows

    if args.max_samples and args.max_samples > 0:
        use_rows = use_rows[: args.max_samples]

    print(
        f"[info] total={len(rows)} train={len(train_rows)} eval={len(eval_rows)} "
        f"use({args.split})={len(use_rows)} stage={args.stage} rt_greedy={args.rt_greedy} sum_greedy={args.sum_greedy}"
    )

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    if args.stage == "both":
        tmp_path = args.out_jsonl + ".stage1.tmp.jsonl"

        # pass A: stage1 -> tmp
        with open(tmp_path, "w", encoding="utf-8") as ftmp:
            for r in tqdm(use_rows, desc="stage1(always_regen)"):
                rec = dict(r)
                rec = run_stage1_on_row(rec, base=base, tok=tok, args=args)
                rec["_split_meta"] = {"split_used": args.split, "train_ratio": args.train_ratio, "seed": args.seed}
                ftmp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # pass B: init wm_model AFTER stage1 done
        if args.adapter and args.adapter.lower() not in ["none", "null", "no", ""]:
            wm_model = PeftModel.from_pretrained(base, args.adapter)
            wm_model.eval()
        else:
            wm_model = base

        # pass B: stage2 -> out
        with open(args.out_jsonl, "w", encoding="utf-8") as fout:
            for rec in tqdm(iter_jsonl(tmp_path), desc="stage2(summary)"):
                rec = run_stage2_on_row(rec, wm_model=wm_model, tok=tok, args=args)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        print(f"[done] wrote -> {args.out_jsonl}")
        return

    wm_model = None
    if args.stage == "2":
        if args.adapter and args.adapter.lower() not in ["none", "null", "no", ""]:
            wm_model = PeftModel.from_pretrained(base, args.adapter)
            wm_model.eval()
        else:
            wm_model = base

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for r in tqdm(use_rows, desc=f"two-stage({args.stage})"):
            rec = dict(r)

            if args.stage == "1":
                rec = run_stage1_on_row(rec, base=base, tok=tok, args=args)
            elif args.stage == "2":
                rec = run_stage2_on_row(rec, wm_model=wm_model, tok=tok, args=args)

            rec["_split_meta"] = {"split_used": args.split, "train_ratio": args.train_ratio, "seed": args.seed}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
