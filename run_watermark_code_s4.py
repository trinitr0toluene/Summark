# -*- coding: utf-8 -*-

import os, sys, json, re, argparse, math, time, hashlib, hmac
from typing import Dict, Any, Optional, List

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    LogitsProcessor,
    set_seed,
)
import logging

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

TAG_SUM_OPEN = "<summary>"
TAG_SUM_CLOSE = "</summary>"
TAG_ANS_OPEN = "<final_answer>"
TAG_ANS_CLOSE = "</final_answer>"
TAG_RT_OPEN = "<reasoning>"
TAG_RT_CLOSE = "</reasoning>"

PROMPT_TEMPLATE_RT = (
    "You are a helpful assistant. You are solving a programming problem.\n"
    "First, think step by step and write a detailed reasoning trace about how to design the algorithm "
    f"and how the code should work, inside {TAG_RT_OPEN} ... {TAG_RT_CLOSE}.\n"
    f"Then, write the final code solution that solves the problem inside {TAG_ANS_OPEN} ... {TAG_ANS_CLOSE}.\n\n"
    "Problem:\n{q}\n"
    f"Respond with exactly one {TAG_RT_OPEN}...{TAG_RT_CLOSE} block followed immediately by one "
    f"{TAG_ANS_OPEN}...{TAG_ANS_CLOSE} block, and do not add any other text."
)

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


def apply_chat_template(tokenizer, user_text: str, system_text: Optional[str] = None) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    if system_text:
        user_text = system_text.strip() + "\n\n" + user_text
    return f"<s>[INST] {user_text} [/INST]"

def extract_between(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    m = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), text, flags=re.S)
    return m.group(1).strip() if m else None


def build_model_and_tokenizer(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl


def generate_text(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=logits_processor,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_only = out[:, inputs["input_ids"].shape[-1] :]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]


class WMProcWithKey(WatermarkLogitsProcessor):

    def __init__(self, tokenizer, gamma, delta, model_key: str, model_id_ns: str = ""):
        super().__init__(
            vocab=list(range(tokenizer.vocab_size)),
            gamma=gamma,
            delta=delta,
            seeding_scheme="selfhash",
        )
        self._key = model_key
        self._ns = model_id_ns
        self._supports_external = hasattr(self, "set_seeds") or hasattr(self, "external_seed")

    def _prf_seed(self, seed_text: str) -> int:
        salt = "wm:v1|" + (self._ns or "")
        msg = (salt + "|" + seed_text).encode("utf-8")
        digest = hmac.new(self._key.encode("utf-8"), msg, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], "big")

    def set_prompt(self, seed_text: str):
        seed = self._prf_seed(seed_text)
        if self._supports_external:
            if hasattr(self, "set_seeds"):
                try:
                    self.set_seeds(seed)
                except Exception:
                    setattr(self, "external_seed", int(seed))
            else:
                setattr(self, "external_seed", int(seed))


class FixedGreenlistProcessor(LogitsProcessor):
    """Bias logits toward one identity's fixed greenlist.

    Unlike KGW, where `gamma` is a greenlist *fraction*, here both parameters are
    additive logit offsets: green tokens get `+gamma`, red tokens get `-delta`.
    The greenlist fraction is fixed when the mask is built (`--green-ratio`).

    `green_ids` is truncated to `vocab_size`, so a model whose vocabulary is
    smaller than the mask's reference vocabulary carries a subset. Detection
    must apply the same truncation when computing the null green rate; see
    `_infer_vocab_size` in `fine-tune/eval_orth.py`.
    """

    def __init__(self, green_ids: List[int], gamma: float, delta: float, vocab_size: int):
        super().__init__()
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.green_mask = torch.zeros(vocab_size, dtype=torch.bool)
        green_ids = [gid for gid in green_ids if 0 <= gid < vocab_size]
        if not green_ids:
            raise ValueError("FixedGreenlistProcessor: empty or invalid green_ids")
        self.green_mask[green_ids] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch_size, vocab_size]
        gmask = self.green_mask.to(scores.device)
        scores[:, gmask] += self.gamma
        scores[:, ~gmask] -= self.delta
        return scores


def _load_key_from_json(path: str, model_id: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if model_id not in m:
        raise KeyError(f"[keys.json] missing key for model_id={model_id}")
    return m[model_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        type=str,
        default="open-r1/OpenThoughts-114k-Code_decontaminated",
        help="HF Hub 数据集名称或本地路径（例如 open-r1/OpenThoughts-114k-Code）",
    )
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--out",
        type=str,
        default="data/out_qwen3b_openthoughts_code_summary_wm.jsonl",
    )
    ap.add_argument("--max-samples", type=int, default=0, help="0=全量")
    ap.add_argument(
        "--rt-max-new",
        type=int,
        default=2048,
        help="第一次生成 reasoning trace + final answer 的最大 token 数",
    )
    ap.add_argument(
        "--sum-max-new",
        type=int,
        default=512,
        help="第二次生成 summary 的最大 token 数",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sampling; teacher generation uses do_sample=True.")
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument(
        "--wm",
        type=str,
        choices=["on", "off"],
        default="on",
        help="summary 是否注入水印（默认 on）",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default="qwen-3b",
        help="模型身份：如 qwen-3b / qwen-1.5b / llama-3b",
    )
    ap.add_argument(
        "--key-config",
        type=str,
        default="data/secure/keys.json",
        help="JSON 文件路径：{model_id: secret_key}（wm=on 时必需，或用 --wm-key-direct）",
    )
    ap.add_argument(
        "--wm-key-direct",
        type=str,
        default=None,
        help="直接传入当前 --model-id 的密钥（用于快速单模型测试）",
    )
    ap.add_argument(
        "--wm-seed-source",
        type=str,
        choices=["sum_prompt", "prompt"],
        default="sum_prompt",
        help="种子来源文本（默认 sum_prompt）",
    )

    ap.add_argument("--wm-gamma", type=float, default=0.25)  # 0.25
    ap.add_argument("--wm-delta", type=float, default=2.0)
    ap.add_argument("--wm-seeding", type=str, default="selfhash")
    ap.add_argument("--wm-z-threshold", type=float, default=4.0)

    ap.add_argument(
        "--orth-mask",
        type=str,
        default=None,
        help="若提供，则使用固定 greenlist 掩码 JSON（例如 orthogonal_greenlists.json）替换原有 selfhash 方案",
    )

    ap.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="每处理多少条样本就 flush 一次到磁盘（0 表示只在结束时 flush）",
    )

    args = ap.parse_args()
    # Teacher generation samples at temperature>0; seed before any generate() call
    # so the teacher corpus is reproducible.
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_out_dir = os.path.dirname(args.out) or "."
    base_name = os.path.basename(args.out)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(raw_out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    args.out = os.path.join(run_dir, base_name)

    base_no_ext = os.path.splitext(base_name)[0]
    config_path = os.path.join(run_dir, base_no_ext + ".config.json")
    log_path = os.path.join(run_dir, base_no_ext + ".log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path, encoding="utf-8")],
    )
    logger = logging.getLogger(__name__)

    with open(config_path, "w", encoding="utf-8") as fconf:
        json.dump(vars(args), fconf, indent=2, ensure_ascii=False)
    logger.info("Saved config to %s", config_path)
    logger.info("Args: %s", vars(args))
    logger.info("Using device: %s", device)

    tokenizer, model = build_model_and_tokenizer(args.model, device)
    if args.model_id == "llama-3b":
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    vocab_size = int(model.config.vocab_size)
    vocab_ids = list(range(vocab_size))

    wm_proc: Optional[LogitsProcessor] = None
    model_key: Optional[str] = None

    if args.wm == "on":
        if args.orth_mask:
            logger.info("Using orthogonal greenlist from %s", args.orth_mask)
            with open(args.orth_mask, "r", encoding="utf-8") as f:
                masks = json.load(f)
            if args.model_id not in masks:
                raise KeyError(f"[orth-mask] missing entry for model_id={args.model_id}")
            green_ids = masks[args.model_id]["green_ids"]
            wm_proc = FixedGreenlistProcessor(
                green_ids=green_ids,
                gamma=args.wm_gamma,
                delta=args.wm_delta,
                vocab_size=vocab_size,
            )
        else:
            logger.info("Using keyed WatermarkLogitsProcessor with selfhash seeding")
            model_key = args.wm_key_direct
            if model_key is None:
                if not args.key_config or not os.path.isfile(args.key_config):
                    raise FileNotFoundError(
                        f"[keys] Not found: {args.key_config}. Provide --key-config or --wm-key-direct when --wm on."
                    )
                model_key = _load_key_from_json(args.key_config, args.model_id)
            wm_proc = WMProcWithKey(
                tokenizer=tokenizer,
                gamma=args.wm_gamma,
                delta=args.wm_delta,
                model_key=model_key,
                model_id_ns=args.model_id,
            )
    else:
        wm_proc = None

    try:
        wm_det = WatermarkDetector(
            vocab=vocab_ids,
            gamma=args.wm_gamma,
            seeding_scheme=args.wm_seeding,
            device=model.device,
            tokenizer=tokenizer,
            z_threshold=args.wm_z_threshold,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
    except TypeError:
        wm_det = WatermarkDetector(
            vocab=vocab_ids,
            gamma=args.wm_gamma,
            seeding_scheme=args.wm_seeding,
            device=model.device,
            tokenizer=tokenizer,
            z_threshold=args.wm_z_threshold,
            normalizers=[],
        )

    print(f"Loading dataset from {args.dataset_path}, split={args.split} ...", file=sys.stderr)
    ds = load_dataset(args.dataset_path, "default", split=args.split)

    def get_prompt(ex: Dict[str, Any]) -> Optional[str]:
        problem = ex.get("problem", None)
        starter = ex.get("starter_code", None)

        if problem and starter:
            return (
                "You are given a programming problem and some starter code.\n\n"
                "Problem description:\n"
                f"{problem}\n\n"
                "Starter code (you may modify it or rewrite from scratch):\n"
                f"{starter}\n"
            )
        if problem:
            return problem

        for k in ["prompt", "question", "input", "query"]:
            if k in ex and ex[k]:
                return ex[k]
        return None

    def get_source_solution(ex: Dict[str, Any]) -> Optional[str]:
        for k in ["deepseek_solution", "solution", "answer"]:
            if k in ex and ex[k]:
                return ex[k]
        return None

    n_total = len(ds)
    n_run = n_total if args.max_samples == 0 else min(args.max_samples, n_total)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")

    print(f"[info] total={n_total} running={n_run}", file=sys.stderr)
    pbar = tqdm(range(n_run), desc="generating")

    for i in pbar:
        ex = ds[i]
        prompt = get_prompt(ex)
        if not prompt:
            continue

        src_sol = get_source_solution(ex)

        rt_prompt_raw = PROMPT_TEMPLATE_RT.format(q=prompt)
        rt_prompt = apply_chat_template(tokenizer, rt_prompt_raw)

        rt_and_ans_text = generate_text(
            model,
            tokenizer,
            rt_prompt,
            max_new_tokens=args.rt_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            logits_processor=None,
        )

        reasoning = extract_between(rt_and_ans_text, TAG_RT_OPEN, TAG_RT_CLOSE) or rt_and_ans_text.strip()
        final_ans = extract_between(rt_and_ans_text, TAG_ANS_OPEN, TAG_ANS_CLOSE) or rt_and_ans_text.strip()

        sum_prompt_raw = PROMPT_TEMPLATE_SUM_FROM_RT.format(rt=reasoning)
        sum_prompt = apply_chat_template(tokenizer, sum_prompt_raw)

        seed_text = sum_prompt if args.wm_seed_source == "sum_prompt" else prompt

        if wm_proc is not None:
            if isinstance(wm_proc, WMProcWithKey):
                wm_proc.set_prompt(seed_text)
                lp = LogitsProcessorList([wm_proc])
            else:
                lp = LogitsProcessorList([wm_proc])
        else:
            lp = None

        summary_text = generate_text(
            model,
            tokenizer,
            sum_prompt,
            max_new_tokens=args.sum_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            logits_processor=lp,
        )
        summary = extract_between(summary_text, TAG_SUM_OPEN, TAG_SUM_CLOSE) or summary_text.strip()

        sum_ids = tokenizer(summary, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        min_prefix = getattr(wm_det, "min_prefix_len", 4)
        n_tok = int(sum_ids.numel())

        if n_tok <= min_prefix:
            det = {
                "error": "too_short_for_detector",
                "num_tokens": n_tok,
                "min_prefix_len": int(min_prefix),
                "is_watermarked": None,
            }
        else:
            det = wm_det.detect(summary)

        def _py(o):
            import numpy as np, torch as _torch

            if isinstance(o, _torch.Tensor):
                o = o.detach().cpu()
                return o.item() if o.numel() == 1 else o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, (list, tuple)):
                return [_py(x) for x in o]
            if isinstance(o, dict):
                return {k: _py(v) for k, v in o.items()}
            return o

        det = _py(det)

        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        rt_ids = tokenizer(reasoning, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        rec = {
            "id": ex.get("id", i),
            "prompt": prompt,
            "source_solution": src_sol,
            "reasoning_trace": reasoning,
            "summary": summary,
            "final_answer": final_ans,
            "gold_answer": None,
            "acc_match": None,
            "wm_detect": det,
            "model_id": args.model_id,
            "model_name": args.model,
            "wm": args.wm,
            "wm_params": {
                "gamma": args.wm_gamma,
                "delta": args.wm_delta,
                "seeding": args.wm_seeding,
                "seed_source": args.wm_seed_source,
                "key_present": bool(model_key is not None),
                "orth_mask": bool(args.orth_mask),
            },
            "seed_text": seed_text,
            "gen_params": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed},
            "prompt_len_naive": int(prompt_ids.numel()),
            "reasoning_len_naive": int(rt_ids.numel()),
            "summary_len_naive": int(sum_ids.numel()),
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if args.flush_every > 0 and (i + 1) % args.flush_every == 0:
            fout.flush()

    fout.close()
    print(f"[done] wrote -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
