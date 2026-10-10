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
    "You are a helpful assistant. For the math question below, first write a detailed step-by-step "
    f"reasoning trace inside {TAG_RT_OPEN} ... {TAG_RT_CLOSE}, then give ONLY the final numeric answer "
    f"inside {TAG_ANS_OPEN} ... {TAG_ANS_CLOSE}.\n\n"
    "Question:\n{q}\n"
    f"Respond with exactly one {TAG_RT_OPEN}...{TAG_RT_CLOSE} block followed immediately by one "
    f"{TAG_ANS_OPEN}...{TAG_ANS_CLOSE} block, and do not add any other text."
)

PROMPT_TEMPLATE_RT_ONLY = (
    "You are a helpful assistant. For the math question below, write a detailed step-by-step "
    f"reasoning trace inside {TAG_RT_OPEN} ... {TAG_RT_CLOSE}.\n\n"
    "Question:\n{q}\n"
    f"Respond with exactly one {TAG_RT_OPEN}...{TAG_RT_CLOSE} block, and do not add any other text."
)

PROMPT_TEMPLATE_ANS_FROM_RT = (
    "You are a helpful assistant. Use the math question and the provided reasoning trace to produce ONLY "
    f"the final numeric answer inside {TAG_ANS_OPEN} ... {TAG_ANS_CLOSE}.\n\n"
    "Question:\n{q}\n\n"
    "Reasoning trace:\n"
    f"{TAG_RT_OPEN}\n{{rt}}\n{TAG_RT_CLOSE}\n\n"
    f"Respond with exactly one {TAG_ANS_OPEN}...{TAG_ANS_CLOSE} block, and do not add any other text."
)

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


# def apply_chat_template(tokenizer, user_text: str) -> str:
#     messages = [
#         {"role": "system", "content": "You are Qwen, a large language model."},
#         {"role": "user", "content": user_text},
#     ]
#     return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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


def number_like(s: str) -> Optional[str]:
    if s is None:
        return None
    t = re.sub(r"[^\d\.\-\+eE/]", " ", s)
    t = re.sub(r"\s+", " ", t).strip()
    cands = re.findall(r"(\d+\/\d+|\d+\.\d+|\d+)", t)
    return cands[-1] if cands else None


def eq_answer(pred: Optional[str], gold: Optional[str]) -> Optional[bool]:
    if gold is None:
        return None
    gp = number_like(gold)
    pp = number_like(pred)
    if gp is None or pp is None:
        return None
    try:
        def to_val(x: str) -> float:
            if "/" in x:
                n, d = x.split("/")
                return float(n) / float(d)
            return float(x)

        return math.isclose(to_val(pp), to_val(gp), rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return None


def count_existing_records(path: str) -> int:
    """Count completed JSONL records, ignoring blank lines."""
    if not path or not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                n += 1
    return n


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
            # vocab=list(tokenizer.get_vocab().values()),
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


def resolve_wm_seed_source(wm_target: str, wm_seed_source: str) -> str:
    if wm_seed_source == "auto":
        return "sum_prompt" if wm_target == "summary" else "rt_prompt"

    allowed = {
        "summary": {"sum_prompt", "prompt"},
        "reasoning": {"rt_prompt", "prompt"},
    }
    if wm_seed_source not in allowed[wm_target]:
        allowed_sorted = ", ".join(sorted(allowed[wm_target]))
        raise ValueError(
            f"--wm-seed-source={wm_seed_source} is incompatible with --wm-target={wm_target}; "
            f"allowed values are auto, {allowed_sorted}"
        )
    return wm_seed_source


def get_seed_text(
    seed_source_effective: str,
    prompt: str,
    rt_prompt: Optional[str] = None,
    sum_prompt: Optional[str] = None,
) -> str:
    source_to_text = {
        "prompt": prompt,
        "rt_prompt": rt_prompt,
        "sum_prompt": sum_prompt,
    }
    seed_text = source_to_text.get(seed_source_effective)
    if seed_text is None:
        raise ValueError(f"missing seed text for seed source: {seed_source_effective}")
    return seed_text


def build_wm_logits_processor(
    wm_proc: Optional[LogitsProcessor],
    seed_text: Optional[str],
) -> Optional[LogitsProcessorList]:
    if wm_proc is None:
        return None
    if isinstance(wm_proc, WMProcWithKey):
        if seed_text is None:
            raise ValueError("seed_text is required for WMProcWithKey")
        wm_proc.set_prompt(seed_text)
    return LogitsProcessorList([wm_proc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        type=str,
        default="open-r1/OpenR1-Math-220k",
        help="HF Hub 数据集名称或本地路径（例如 open-r1/OpenR1-Math-220k）",
    )
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument(
        "--out",
        type=str,
        default="data/out_qwen32b_openr1_summary_wm.jsonl",
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
        help="是否启用水印注入（默认 on；具体字段由 --wm-target 决定）",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default="qwen-32b",
        help="模型身份：如 qwen-32b / qwen-14b / qwen-7b / qwen-3b / qwen-1.5b / llama-3b / mistral-7b / mistral-3b",
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
        "--wm-target",
        type=str,
        choices=["summary", "reasoning"],
        default="summary",
        help="将水印加到哪个字段（默认 summary）",
    )
    ap.add_argument(
        "--wm-seed-source",
        type=str,
        choices=["auto", "sum_prompt", "rt_prompt", "prompt"],
        default="auto",
        help="种子来源文本（默认 auto；summary->sum_prompt, reasoning->rt_prompt）",
    )

    ap.add_argument("--wm-gamma", type=float, default=0.25)
    ap.add_argument("--wm-delta", type=float, default=2.0)
    ap.add_argument("--wm-seeding", type=str, default="selfhash")
    ap.add_argument("--wm-z-threshold", type=float, default=4.0)

    ap.add_argument(
        "--orth-mask",
        type=str,
        default="data/secure/orthogonal_greenlists_qwen.json",
        help="若提供，则使用固定 greenlist 掩码 JSON（例如 orthogonal_greenlists.json）替换原有 selfhash 方案",
    )

    ap.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="每处理多少条样本就 flush 一次到磁盘（0 表示只在结束时 flush）",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="若输出 jsonl 已存在，则从现有条数继续追加，且不再创建新的时间戳子目录",
    )

    args = ap.parse_args()
    try:
        seed_source_effective = resolve_wm_seed_source(args.wm_target, args.wm_seed_source)
    except ValueError as e:
        ap.error(str(e))
    # Teacher generation samples at temperature>0; seed before any generate() call
    # so the teacher corpus is reproducible.
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_out_dir = os.path.dirname(args.out) or "."
    base_name = os.path.basename(args.out)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = raw_out_dir if args.resume else os.path.join(raw_out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    if not args.resume:
        args.out = os.path.join(run_dir, base_name)

    base_no_ext = os.path.splitext(base_name)[0]
    if args.resume:
        config_path = os.path.join(run_dir, f"{base_no_ext}.resume.{ts}.config.json")
        log_path = os.path.join(run_dir, f"{base_no_ext}.resume.{ts}.log.txt")
    else:
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

    # vocab_ids = list(tokenizer.get_vocab().values())
    # vocab_size = model.config.vocab_size
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
        for k in ["prompt", "question", "input", "query", "problem"]:
            if k in ex and ex[k]:
                return ex[k]
        return None

    def get_gold_answer(ex: Dict[str, Any]) -> Optional[str]:
        for k in ["final_answer", "answer", "label"]:
            if k in ex and ex[k]:
                return ex[k]
        return None

    n_total = len(ds)
    n_run = n_total if args.max_samples == 0 else min(args.max_samples, n_total)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    resume_count = count_existing_records(args.out) if args.resume else 0
    if resume_count > n_run:
        raise ValueError(
            f"[resume] Existing records ({resume_count}) exceed requested max-samples ({n_run})."
        )
    if args.resume and resume_count > 0:
        logger.info("Resume mode enabled: found %d existing records in %s", resume_count, args.out)
    elif args.resume:
        logger.info("Resume mode enabled: no existing records found in %s, starting from scratch", args.out)

    if resume_count >= n_run:
        logger.info("Nothing to do: existing records=%d already reach requested total=%d", resume_count, n_run)
        print(f"[done] already complete -> {args.out}", file=sys.stderr)
        return

    fout = open(args.out, "a" if args.resume else "w", encoding="utf-8")

    print(f"[info] total={n_total} running={n_run} start={resume_count}", file=sys.stderr)
    pbar = tqdm(range(resume_count, n_run), desc="generating", initial=resume_count, total=n_run)

    for i in pbar:
        ex = ds[i]
        prompt = get_prompt(ex)
        gold = get_gold_answer(ex)
        if not prompt:
            continue

        seed_text: Optional[str] = None

        if args.wm == "on" and args.wm_target == "reasoning":
            rt_prompt_raw = PROMPT_TEMPLATE_RT_ONLY.format(q=prompt)
            rt_prompt = apply_chat_template(tokenizer, rt_prompt_raw)
            seed_text = get_seed_text(
                seed_source_effective=seed_source_effective,
                prompt=prompt,
                rt_prompt=rt_prompt,
            )
            rt_lp = build_wm_logits_processor(wm_proc, seed_text)

            reasoning_text = generate_text(
                model,
                tokenizer,
                rt_prompt,
                max_new_tokens=args.rt_max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                logits_processor=rt_lp,
            )
            reasoning = extract_between(reasoning_text, TAG_RT_OPEN, TAG_RT_CLOSE) or reasoning_text.strip()

            ans_prompt_raw = PROMPT_TEMPLATE_ANS_FROM_RT.format(q=prompt, rt=reasoning)
            ans_prompt = apply_chat_template(tokenizer, ans_prompt_raw)
            ans_text = generate_text(
                model,
                tokenizer,
                ans_prompt,
                max_new_tokens=args.rt_max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                logits_processor=None,
            )
            final_ans = extract_between(ans_text, TAG_ANS_OPEN, TAG_ANS_CLOSE) or ans_text.strip()
        else:
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

        if args.wm == "on" and args.wm_target == "summary":
            seed_text = get_seed_text(
                seed_source_effective=seed_source_effective,
                prompt=prompt,
                rt_prompt=rt_prompt,
                sum_prompt=sum_prompt,
            )
            lp = build_wm_logits_processor(wm_proc, seed_text)
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

        reasoning_ids = tokenizer(reasoning, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        sum_ids = tokenizer(summary, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        wm_target = args.wm_target
        detect_text = summary if wm_target == "summary" else reasoning
        detect_ids = sum_ids if wm_target == "summary" else reasoning_ids
        min_prefix = getattr(wm_det, "min_prefix_len", 4)
        n_tok = int(detect_ids.numel())

        if n_tok <= min_prefix:
            det = {
                "error": "too_short_for_detector",
                "num_tokens": n_tok,
                "min_prefix_len": int(min_prefix),
                "is_watermarked": None,
            }
        else:
            det = wm_det.detect(detect_text)

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

        acc = eq_answer(final_ans, gold)

        rec = {
            "id": ex.get("id", i),
            "prompt": prompt,
            "reasoning_trace": reasoning,
            "summary": summary,
            "final_answer": final_ans,
            "gold_answer": gold,
            "acc_match": acc,  # True/False/None
            "wm_target": wm_target,
            "wm_detect": det,
            "model_id": args.model_id,
            "model_name": args.model,
            "wm": args.wm,
            "wm_params": {
                "gamma": args.wm_gamma,
                "delta": args.wm_delta,
                "seeding": args.wm_seeding,
                "seed_source": args.wm_seed_source,
                "seed_source_effective": seed_source_effective,
                "key_present": bool(model_key is not None),
                "orth_mask": bool(args.orth_mask),
            },
            "seed_text": seed_text,
            "gen_params": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed},
            "reasoning_len_native": int(reasoning_ids.numel()),
            "summary_len_native": int(sum_ids.numel()),
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if args.flush_every > 0 and (i + 1) % args.flush_every == 0:
            fout.flush()

    fout.close()
    print(f"[done] wrote -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
