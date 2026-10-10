#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gen_student_summaries.py  (ROBUST VERSION)
-----------------------------------------
Goal: make qwen-1.5b not collapse into "only 2/40 counted" by ensuring:
  1) Stage1 always produces a non-empty reasoning_trace via robust parsing/fallback.
  2) Stage2 always produces a non-empty summary_student via fallback inputs and safe defaults.

Key changes vs your original:
  - Added parse_stage1_outputs(): robust tag parsing + fallback to full text.
  - Stage1 uses parse_stage1_outputs() instead of strict extract_between() only.
  - Stage2: if reasoning_trace empty, fallback to _stage1_raw, else question+final, else existing fields.
  - Stage2: if model output is empty, write a non-empty placeholder so it can enter evaluation.
"""

import os, json, re, argparse, random
from typing import Dict, Any, Optional, List, Tuple

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
    "You are a helpful assistant. Solve the following math problem.\n\n"
    "Requirements:\n"
    "1) Write a detailed reasoning trace.\n"
    "2) Then provide the final answer.\n\n"
    f"Write the reasoning trace inside {TAG_RT_OPEN} ... {TAG_RT_CLOSE}.\n"
    f"Write the final answer inside {TAG_FINAL_OPEN} ... {TAG_FINAL_CLOSE}.\n"
    f"Respond with exactly two blocks in this order:\n"
    f"{TAG_RT_OPEN}...{TAG_RT_CLOSE}\n"
    f"{TAG_FINAL_OPEN}...{TAG_FINAL_CLOSE}\n\n"
    "Math problem:\n{q}\n"
)

PROMPT_TEMPLATE_FINAL_FROM_Q_RT = (
    "You are a helpful assistant. Given the following math problem and reasoning trace, "
    "provide the final answer only.\n\n"
    "Math problem:\n{q}\n\n"
    "Reasoning trace:\n{rt}\n\n"
    f"Write the final answer inside {TAG_FINAL_OPEN} ... {TAG_FINAL_CLOSE}.\n"
    f"Respond with exactly one block: {TAG_FINAL_OPEN}...{TAG_FINAL_CLOSE}."
)

# ===== Stage 2: generate summary from reasoning trace (watermark via adapter here) =====
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
    """
    raw = _safe_str(text).strip()

    rt_strict = extract_between(raw, TAG_RT_OPEN, TAG_RT_CLOSE)
    fa_strict = extract_between(raw, TAG_FINAL_OPEN, TAG_FINAL_CLOSE)

    parse_mode = {
        "has_reasoning_open": (TAG_RT_OPEN in raw),
        "has_reasoning_close": (TAG_RT_CLOSE in raw),
        "has_final_open": (TAG_FINAL_OPEN in raw),
        "has_final_close": (TAG_FINAL_CLOSE in raw),
        "used_fallback_reasoning": False,
        "used_fallback_final": False,
    }

    # ---- reasoning fallback ----
    rt = (rt_strict or "").strip()
    if not rt:
        parse_mode["used_fallback_reasoning"] = True
        if TAG_RT_OPEN in raw:
            after = raw.split(TAG_RT_OPEN, 1)[1]
            if TAG_RT_CLOSE in after:
                rt = after.split(TAG_RT_CLOSE, 1)[0].strip()
            elif TAG_FINAL_OPEN in after:
                rt = after.split(TAG_FINAL_OPEN, 1)[0].strip()
            else:
                rt = after.strip()
        else:
            # no <reasoning>: use everything before <final> if exists, else whole text
            if TAG_FINAL_OPEN in raw:
                rt = raw.split(TAG_FINAL_OPEN, 1)[0].strip()
            else:
                rt = raw

    # hard guarantee
    rt = _ensure_nonempty(rt, raw)

    # ---- final fallback ----
    fa = (fa_strict or "").strip()
    if not fa:
        parse_mode["used_fallback_final"] = True
        if TAG_FINAL_OPEN in raw:
            after = raw.split(TAG_FINAL_OPEN, 1)[1]
            if TAG_FINAL_CLOSE in after:
                fa = after.split(TAG_FINAL_CLOSE, 1)[0].strip()
            else:
                fa = after.strip()
        else:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            fa = lines[-1] if lines else ""

    fa = _ensure_nonempty(fa, "")

    return rt, fa, parse_mode

# ---------------- JSONL safe loader ----------------
_VALID_ESC = set('"\\/bfnrtu')

def _escape_invalid_backslashes(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch); i += 1; continue
        if i + 1 >= n:
            out.append("\\\\"); i += 1; continue
        nxt = s[i + 1]

        if nxt in _VALID_ESC and nxt != "u":
            out.append("\\"); out.append(nxt); i += 2; continue

        if nxt == "u":
            if i + 6 <= n:
                hexpart = s[i+2:i+6]
                if re.fullmatch(r"[0-9a-fA-F]{4}", hexpart):
                    out.append("\\u" + hexpart); i += 6; continue
            out.append("\\\\"); i += 1; continue

        out.append("\\\\"); i += 1
    return "".join(out)

def load_jsonl_safe(path: str, max_bad: int = 50) -> List[Dict[str, Any]]:
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
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
                        ctx = line[max(0, pos-60):pos+60]
                        print(f"[warn] skip bad json line={ln} pos={pos} msg={e.msg}")
                        print(f"       context: {ctx}")
                    if bad >= max_bad:
                        print(f"[warn] too many bad lines (>= {max_bad}), stop reading.")
                        break
    if bad > 0:
        print(f"[warn] total bad lines skipped: {bad}")
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
        eval_rows = rows[-max(1, n // 20):]
        train_rows = rows[: max(0, n - len(eval_rows))]
    return train_rows, eval_rows

def pick_question(r: Dict[str, Any]) -> Optional[str]:
    return (r.get("question") or r.get("prompt") or r.get("problem") or r.get("query"))

def generate_text(model, tok, prompt: str, max_new: int, temperature: float, top_p: float,
                  do_sample: bool = True, repetition_penalty: float = 1.0,
                  no_repeat_ngram_size: int = 0) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    gen_only = out[:, inputs["input_ids"].shape[-1]:]
    return tok.batch_decode(gen_only, skip_special_tokens=True)[0]

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------------- Stage runners ----------------
def run_stage1_on_row(rec: Dict[str, Any], base, final_model, tok, args) -> Dict[str, Any]:
    q = pick_question(rec)
    if not q:
        rec["_stage1_error"] = "missing question/prompt/problem field"
        # ensure we still have something for stage2 fallback
        rec["_stage1_raw"] = _ensure_nonempty(_safe_str(rec.get("_stage1_raw")), "")
        rec["reasoning_trace"] = _ensure_nonempty(_safe_str(rec.get("reasoning_trace")), rec["_stage1_raw"])
        rec["final_answer"] = _ensure_nonempty(_safe_str(rec.get("final_answer")), "")
        rec["answer"] = _ensure_nonempty(_safe_str(rec.get("answer")), rec["final_answer"])
    else:
        user_text = PROMPT_TEMPLATE_RT_AND_FINAL.format(q=q)
        prompt = apply_chat_template(tok, user_text)
        text = generate_text(
            model=base, tok=tok, prompt=prompt,
            max_new=args.rt_max_new,
            temperature=args.rt_temperature,
            top_p=args.rt_top_p,
            do_sample=not args.rt_greedy,
        )
        rec["_stage1_raw"] = _safe_str(text).strip()

        rt, fa, parse_meta = parse_stage1_outputs(rec["_stage1_raw"])
        rec["reasoning_trace"] = rt
        rec["final_answer"] = fa
        rec["answer"] = fa
        rec["_stage1_parse"] = parse_meta

        final_adapter_on = bool(args.final_adapter and args.final_adapter.lower() not in ["none", "null", "no", ""])
        if final_adapter_on and final_model is not None and rt.strip():
            final_prompt = apply_chat_template(tok, PROMPT_TEMPLATE_FINAL_FROM_Q_RT.format(q=q, rt=rt))
            final_text = generate_text(
                model=final_model,
                tok=tok,
                prompt=final_prompt,
                max_new=args.final_max_new,
                temperature=args.final_temperature,
                top_p=args.final_top_p,
                do_sample=not args.final_greedy,
            )
            final_text = _safe_str(final_text).strip()
            final_only = extract_between(final_text, TAG_FINAL_OPEN, TAG_FINAL_CLOSE)
            rec["final_answer_stage1"] = rec["final_answer"]
            rec["answer_stage1"] = rec["answer"]
            rec["final_answer"] = _ensure_nonempty(final_only or "", final_text)
            rec["answer"] = rec["final_answer"]
            rec["_stage1_final_raw"] = final_text

    rec["_stage1_meta"] = {
        "base_model": args.base_model,
        "final_adapter": args.final_adapter,
        "rt_max_new": args.rt_max_new,
        "rt_temperature": args.rt_temperature,
        "rt_top_p": args.rt_top_p,
        "rt_greedy": bool(args.rt_greedy),
        "final_max_new": args.final_max_new,
        "final_temperature": args.final_temperature,
        "final_top_p": args.final_top_p,
        "final_greedy": bool(args.final_greedy),
        "stage1_watermark": "final_only" if (args.final_adapter and args.final_adapter.lower() not in ["none", "null", "no", ""]) else "off",
        "stage1_policy": "always_regenerate_rt_then_optional_final_regen",
    }
    return rec

def _build_stage2_input(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns: (rt_input, source_tag)
    Ensure non-empty if possible.
    """
    rt = _safe_str(rec.get("reasoning_trace")).strip()
    if rt:
        return rt, "reasoning_trace"

    raw = _safe_str(rec.get("_stage1_raw")).strip()
    if raw:
        return raw, "_stage1_raw"

    # # fallback to teacher summary if exists
    # teacher_sum = _safe_str(rec.get("summary")).strip()
    # if teacher_sum:
    #     return teacher_sum, "summary(teacher)"

    q = _safe_str(pick_question(rec)).strip()
    fa = _safe_str(rec.get("final_answer")).strip()
    combo = f"Problem:\n{q}\n\nFinal answer:\n{fa}".strip()
    if combo:
        return combo, "question+final"

    return "", "none"

def run_stage2_on_row(rec: Dict[str, Any], wm_model, tok, args) -> Dict[str, Any]:
    rt_input, src = _build_stage2_input(rec)

    if not rt_input.strip():
        # hard guarantee: produce non-empty summary_student so it can enter evaluation
        rec["_stage2_error"] = "missing_all_inputs_for_summary"
        rec["summary_student"] = "N/A"
    else:
        user_text = PROMPT_TEMPLATE_SUM_FROM_RT.format(rt=rt_input)
        prompt = apply_chat_template(tok, user_text)
        try:
            text = generate_text(
                model=wm_model, tok=tok, prompt=prompt,
                max_new=args.sum_max_new,
                temperature=args.sum_temperature,
                top_p=args.sum_top_p,
                do_sample=not args.sum_greedy,
                repetition_penalty=args.sum_repetition_penalty,
                no_repeat_ngram_size=args.sum_no_repeat_ngram_size,
            )
            text = _safe_str(text).strip()
        except Exception as e:
            rec["_stage2_error"] = f"generation_exception: {type(e).__name__}"
            text = ""

        summary = extract_between(text, TAG_SUM_OPEN, TAG_SUM_CLOSE) if text else None
        summary = _ensure_nonempty(summary or "", text)

        # If model produced nothing (rare but possible), still ensure non-empty
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
        "sum_repetition_penalty": args.sum_repetition_penalty,
        "sum_no_repeat_ngram_size": args.sum_no_repeat_ngram_size,
        "stage2_input_source": rec.get("_stage2_warn", "reasoning_trace"),
        "stage2_watermark": "on" if (args.adapter and args.adapter.lower() not in ["none","null","no",""]) else "off",
    }
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="none",
                    help="Stage2 LoRA adapter (watermark on summaries). Use 'none' to disable.")
    ap.add_argument("--final-adapter", type=str, default="none",
                    help="Optional Stage1 LoRA adapter. When set, regenerate only the final answer and write it to both final_answer and answer.")
    ap.add_argument("--in-jsonl", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--stage", type=str, choices=["1", "2", "both"], default="both")

    ap.add_argument("--rt-max-new", type=int, default=2048)
    ap.add_argument("--rt-temperature", type=float, default=0.7)
    ap.add_argument("--rt-top-p", type=float, default=0.95)
    ap.add_argument("--rt-greedy", action="store_true",
                    help="If set, Stage1 uses greedy decoding (do_sample=False), improving tag compliance on small models.")

    ap.add_argument("--final-max-new", type=int, default=128)
    ap.add_argument("--final-temperature", type=float, default=0.2)
    ap.add_argument("--final-top-p", type=float, default=0.95)
    ap.add_argument("--final-greedy", action="store_true",
                    help="If set, final-answer regeneration uses greedy decoding.")

    ap.add_argument("--sum-max-new", type=int, default=512)
    ap.add_argument("--sum-temperature", type=float, default=0.7)
    ap.add_argument("--sum-top-p", type=float, default=0.95)
    ap.add_argument("--sum-greedy", action="store_true",
                    help="If set, Stage2 uses greedy decoding (do_sample=False).")
    ap.add_argument("--sum-repetition-penalty", type=float, default=1.0,
                    help="Stage2 repetition penalty. Values above 1 can suppress degenerate loops.")
    ap.add_argument("--sum-no-repeat-ngram-size", type=int, default=0,
                    help="Stage2 no-repeat ngram size; 0 disables this constraint.")

    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--split", type=str, choices=["eval", "train", "all"], default="eval")
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    final_model = None
    if args.stage in ["1", "both"] and args.final_adapter and args.final_adapter.lower() not in ["none", "null", "no", ""]:
        final_model = PeftModel.from_pretrained(base, args.final_adapter)
        final_model.eval()

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
        use_rows = use_rows[:args.max_samples]

    print(f"[info] total={len(rows)} train={len(train_rows)} eval={len(eval_rows)} use({args.split})={len(use_rows)} stage={args.stage}")
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    if args.stage == "both":
        tmp_path = args.out_jsonl + ".stage1.tmp.jsonl"

        # pass A: stage1 -> tmp
        with open(tmp_path, "w", encoding="utf-8") as ftmp:
            for r in tqdm(use_rows, desc="stage1(always_regen)"):
                rec = dict(r)
                rec = run_stage1_on_row(rec, base=base, final_model=final_model, tok=tok, args=args)
                rec["_split_meta"] = {"split_used": args.split, "train_ratio": args.train_ratio, "seed": args.seed}
                ftmp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # init wm_model after stage1
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

    # stage=1 or stage=2
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
                rec = run_stage1_on_row(rec, base=base, final_model=final_model, tok=tok, args=args)
            elif args.stage == "2":
                rec = run_stage2_on_row(rec, wm_model=wm_model, tok=tok, args=args)

            rec["_split_meta"] = {"split_used": args.split, "train_ratio": args.train_ratio, "seed": args.seed}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
