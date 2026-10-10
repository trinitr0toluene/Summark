#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys, json, glob, argparse, random, math
from typing import List, Dict, Any, Iterable, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer


def resolve_paths(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    uniq, seen = [], set()
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def read_jsonl_many(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def group_by_model(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        mid = r.get("model_id")
        if not mid:
            continue
        buckets.setdefault(mid, []).append(r)
    return buckets


def rand_sample(items: List[Any], k: int, seed: int = 42) -> List[Any]:
    rng = random.Random(seed)
    if k <= 0 or len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def tpr_fpr_fnr_tnr_at_threshold(score_pos: np.ndarray,
                                 score_neg: np.ndarray,
                                 threshold: float):
    """
    predict "substitution"(pos) if score >= threshold
    """
    yhat_pos = (score_pos >= threshold)
    tp = int(yhat_pos.sum())
    fn = int(len(score_pos) - tp)

    yhat_neg = (score_neg >= threshold)
    fp = int(yhat_neg.sum())
    tn = int(len(score_neg) - fp)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr, fnr, tnr, tp, fp, tn, fn


def acc_at_threshold(score_pos: np.ndarray,
                     score_neg: np.ndarray,
                     threshold: float) -> float:
    """
    ACC = (TP + TN) / (P + N) under decision rule score >= threshold => positive
    """
    _, _, _, _, tp, fp, tn, fn = tpr_fpr_fnr_tnr_at_threshold(score_pos, score_neg, threshold)
    denom = tp + fp + tn + fn
    return float((tp + tn) / denom) if denom > 0 else 0.0


def metrics_at_threshold(score_pos: np.ndarray,
                         score_neg: np.ndarray,
                         threshold: float) -> Dict[str, float]:
    """
    Convenience bundle for (TPR,FPR,FNR,TNR,ACC,TP,FP,TN,FN)
    """
    tpr, fpr, fnr, tnr, tp, fp, tn, fn = tpr_fpr_fnr_tnr_at_threshold(score_pos, score_neg, threshold)
    acc = acc_at_threshold(score_pos, score_neg, threshold)
    return {
        "tpr": float(tpr), "fpr": float(fpr), "fnr": float(fnr), "tnr": float(tnr),
        "acc": float(acc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }


def tpr_at_fpr(score_pos: np.ndarray,
               score_neg: np.ndarray,
               target_fpr: float) -> Dict[str, float]:
    """
    max_{threshold: FPR(threshold) <= target_fpr} TPR(threshold)
    """
    if score_pos.size == 0 or score_neg.size == 0:
        return {"tpr": 0.0, "fpr": 0.0, "threshold": float("nan")}

    all_scores = np.unique(np.concatenate([score_pos, score_neg]).astype(float))
    if all_scores.size == 0:
        return {"tpr": 0.0, "fpr": 0.0, "threshold": float("nan")}

    eps = 1e-12
    thresholds = np.concatenate([
        [all_scores.max() + eps],
        all_scores[::-1],           # descending
        [all_scores.min() - eps],
    ]).astype(float)

    best_tpr = -1.0
    best_fpr = 0.0
    best_thr = float(thresholds[0])

    for t in thresholds:
        tpr, fpr, _, _, _, _, _, _ = tpr_fpr_fnr_tnr_at_threshold(score_pos, score_neg, float(t))
        if fpr <= target_fpr:
            if (tpr > best_tpr) or (abs(tpr - best_tpr) < 1e-12 and fpr > best_fpr):
                best_tpr = float(tpr)
                best_fpr = float(fpr)
                best_thr = float(t)

    if best_tpr < 0:
        best_tpr, best_fpr, best_thr = 0.0, 0.0, float(all_scores.max() + eps)

    return {"tpr": best_tpr, "fpr": best_fpr, "threshold": best_thr}


def eer_from_scores(score_pos: np.ndarray,
                    score_neg: np.ndarray) -> Dict[str, float]:
    """
    EER (Equal Error Rate): choose threshold where FPR ~= FNR.
    We compute FPR/FNR on a set of candidate thresholds (unique scores + eps padding),
    then:
      - if there is a sign change in (FPR-FNR), do linear interpolation to estimate EER
      - else pick the threshold minimizing |FPR-FNR|
    """
    if score_pos.size == 0 or score_neg.size == 0:
        return {
            "eer": float("nan"),
            "threshold": float("nan"),
            "fpr": float("nan"),
            "fnr": float("nan"),
        }

    all_scores = np.unique(np.concatenate([score_pos, score_neg]).astype(float))
    if all_scores.size == 0:
        return {
            "eer": float("nan"),
            "threshold": float("nan"),
            "fpr": float("nan"),
            "fnr": float("nan"),
        }

    eps = 1e-12
    thresholds = np.concatenate([
        [all_scores.max() + eps],
        all_scores[::-1],  # descending
        [all_scores.min() - eps],
    ]).astype(float)

    f_list = []
    m_list = []
    for t in thresholds:
        m = metrics_at_threshold(score_pos, score_neg, float(t))
        f = float(m["fpr"] - m["fnr"])
        f_list.append(f)
        m_list.append(m)

    # Find sign change (crossing)
    for i in range(1, len(f_list)):
        f_prev, f_cur = f_list[i - 1], f_list[i]
        if (f_prev >= 0.0 and f_cur <= 0.0) or (f_prev <= 0.0 and f_cur >= 0.0):
            m1, m2 = m_list[i - 1], m_list[i]
            if abs(f_prev - f_cur) < 1e-12:
                w = 0.5
            else:
                w = float(f_prev / (f_prev - f_cur))  # weight toward m2

            fpr_i = float(m1["fpr"] + w * (m2["fpr"] - m1["fpr"]))
            fnr_i = float(m1["fnr"] + w * (m2["fnr"] - m1["fnr"]))
            thr_i = float(m1["threshold"] + w * (m2["threshold"] - m1["threshold"]))
            eer = float(0.5 * (fpr_i + fnr_i))
            return {"eer": eer, "threshold": thr_i, "fpr": fpr_i, "fnr": fnr_i}

    # No crossing: pick closest
    best_j = int(np.argmin(np.abs(np.array(f_list, dtype=float))))
    m = m_list[best_j]
    eer = float(0.5 * (m["fpr"] + m["fnr"]))
    return {"eer": eer, "threshold": float(m["threshold"]), "fpr": float(m["fpr"]), "fnr": float(m["fnr"])}


def threshold_from_neg(scores_neg: np.ndarray, target_fpr: float) -> float:
    if scores_neg.size == 0:
        return float("inf")
    target_fpr = float(target_fpr)
    target_fpr = min(max(target_fpr, 0.0), 1.0)
    return float(np.quantile(scores_neg.astype(float), 1.0 - target_fpr))


# ---------- candidates (tokenizer + mask) ----------
def _infer_vocab_size(tok) -> int:
    """Size of the token-id space the tokenizer can actually emit.

    Must match the universe the generator applied the greenlist over, otherwise
    the null green rate p used by the z-test is wrong. `tok.vocab_size` reports
    the *base* vocabulary and excludes added/special tokens (Qwen2.5: 151643 vs
    151665 real ids; Llama-3.2: 128000 vs 128256), so it is not usable here.
    """
    try:
        return max(tok.get_vocab().values()) + 1
    except Exception:
        return int(len(tok.get_vocab()))


def load_candidate_config(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    cands = obj.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise ValueError("candidate-config 必须包含非空的 candidates 列表。")
    for c in cands:
        for k in ("model_id", "tokenizer", "orth_mask"):
            if k not in c or not c[k]:
                raise ValueError(f"candidate 缺少字段 {k}: {c}")
    return cands


def load_green_ids(mask_path: str, model_id: str) -> set:
    with open(mask_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entry = raw.get(model_id)
    if not entry:
        raise KeyError(f"[orth-mask] {mask_path} 缺少 model_id={model_id}")
    green_ids = entry.get("green_ids")
    if not isinstance(green_ids, list) or not green_ids:
        raise ValueError(f"[orth-mask] {mask_path} 的 {model_id}.green_ids 为空/非法")
    return set(int(x) for x in green_ids)


class Candidate:
    def __init__(self, model_id: str, tokenizer_name: str, mask_path: str):
        self.model_id = model_id
        self.tokenizer_name = tokenizer_name
        self.mask_path = mask_path

        print(f"[info] loading tokenizer for {model_id}: {tokenizer_name}", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.V = _infer_vocab_size(self.tok)

        # The mask is built over one reference tokenizer's id space. A candidate
        # whose vocabulary is smaller (e.g. Llama-3.2 at 128256 vs Qwen2.5 at
        # 151665) never sees the out-of-range ids, and the generator dropped them
        # too, so p must be computed from the truncated set over this vocabulary.
        all_green = load_green_ids(mask_path, model_id)
        self.green_ids = {g for g in all_green if 0 <= g < self.V}
        if not self.green_ids:
            raise ValueError(
                f"[orth-mask] no green id of {model_id} falls inside the "
                f"{tokenizer_name} vocabulary (V={self.V})."
            )
        n_dropped = len(all_green) - len(self.green_ids)
        self.p = len(self.green_ids) / float(self.V) if self.V > 0 else 0.5
        print(
            f"[info] {model_id}: V={self.V} green={len(self.green_ids)}"
            f" (dropped {n_dropped} out-of-range) p={self.p:.4f}",
            file=sys.stderr,
        )

    def z(self, text: str) -> Tuple[float, int, int]:
        if not isinstance(text, str) or not text.strip():
            return 0.0, 0, 0
        if self.p <= 0.0 or self.p >= 1.0:
            return 0.0, 0, 0

        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        T = int(ids.numel())
        if T <= 0:
            return 0.0, 0, 0

        id_list = ids.tolist()
        G = sum(1 for t in id_list if t in self.green_ids)

        mu = self.p * T
        var = self.p * (1.0 - self.p) * T
        sigma = math.sqrt(var) if var > 0 else 1.0
        z = (G - mu) / sigma if sigma > 0 else 0.0
        return float(z), T, int(G)


def compute_z_by_model(text: str, cands: List[Candidate]) -> Dict[str, float]:
    z_by = {}
    for c in cands:
        zc, _, _ = c.z(text)
        z_by[c.model_id] = float(zc)
    return z_by


def eval_setting(setting_name: str,
                 declared_id: str,
                 actual_id: str,
                 samples_declared_from: List[Dict[str, Any]],
                 samples_actual_from: List[Dict[str, Any]],
                 cands: List[Candidate],
                 field: str,
                 z_threshold: float,
                 min_tokens: int = 0,
                 score_mode: str = "sign",
                 threshold_mode: str = "fixed",
                 calib_fpr: float = 0.01) -> Dict[str, Any]:
    model_ids = [c.model_id for c in cands]

    def extract_text(rec: Dict[str, Any]) -> Optional[str]:
        s = rec.get(field)
        return s if isinstance(s, str) and s.strip() else None

    def pass_min_tokens(s: str) -> bool:
        if min_tokens <= 0:
            return True
        Ts = []
        for c in cands:
            _, T, _ = c.z(s)
            Ts.append(T)
        return (len(Ts) == 0) or (min(Ts) >= min_tokens)

    pos_raw = [r for r in samples_actual_from if extract_text(r) is not None]
    neg_raw = [r for r in samples_declared_from if extract_text(r) is not None]

    pos_scores: List[float] = []
    neg_scores: List[float] = []
    out_lines: List[Dict[str, Any]] = []
    total_pos_eval = 0
    total_neg_eval = 0

    # --- helper: compute score ---
    def compute_score(z_by: Dict[str, float]) -> Tuple[str, float, float, float, float]:
        pred = max(model_ids, key=lambda mid: z_by.get(mid, float("-inf")))
        z_star = float(z_by.get(pred, 0.0))

        if score_mode == "diff":
            z_decl = float(z_by.get(declared_id, 0.0))
            z_other = max(float(z_by.get(m, float("-inf"))) for m in model_ids if m != declared_id)
            score = float(z_other - z_decl)
            return pred, z_star, z_decl, z_other, score

        # score_mode == "sign"
        score = float(z_star if pred != declared_id else -z_star)
        return pred, z_star, float("nan"), float("nan"), score

    # positives
    for r in tqdm(pos_raw, desc=f"[{setting_name}] positives"):
        s = extract_text(r)
        if s is None or not pass_min_tokens(s):
            continue

        z_by = compute_z_by_model(s, cands)
        pred, z_star, z_decl, z_other, score = compute_score(z_by)

        pos_scores.append(score)
        total_pos_eval += 1

        o = dict(r)
        o.update({
            "setting": setting_name,
            "declared_model_id": declared_id,
            "is_substitution_gt": True,
            "wm_orth": {
                "z_by_model": z_by,
                "pred": pred,
                "z_star": z_star,
                "z_declared": z_decl,
                "z_best_other": z_other,
                "score_mode": score_mode,
            },
            "score_substitution": float(score),
        })
        out_lines.append(o)

    # negatives
    for r in tqdm(neg_raw, desc=f"[{setting_name}] negatives"):
        s = extract_text(r)
        if s is None or not pass_min_tokens(s):
            continue

        z_by = compute_z_by_model(s, cands)
        pred, z_star, z_decl, z_other, score = compute_score(z_by)

        neg_scores.append(score)
        total_neg_eval += 1

        o = dict(r)
        o.update({
            "setting": setting_name,
            "declared_model_id": declared_id,
            "is_substitution_gt": False,
            "wm_orth": {
                "z_by_model": z_by,
                "pred": pred,
                "z_star": z_star,
                "z_declared": z_decl,
                "z_best_other": z_other,
                "score_mode": score_mode,
            },
            "score_substitution": float(score),
        })
        out_lines.append(o)

    score_pos = np.array(pos_scores, dtype=float)
    score_neg = np.array(neg_scores, dtype=float)

    if score_pos.size == 0 or score_neg.size == 0:
        summary = {
            "setting": setting_name,
            "declared_model": declared_id,
            "actual_model_positive": actual_id,
            "total_pos_eval": int(total_pos_eval),
            "total_neg_eval": int(total_neg_eval),
            "score_mode": score_mode,
            "threshold_mode": threshold_mode,
            "threshold": float(z_threshold),
            "threshold_used": float("nan"),
            "TP": 0, "FP": 0, "TN": 0, "FN": 0,
            "TPR": 0.0, "FPR": 0.0, "FNR": 0.0, "TNR": 0.0,
            "AUC": float("nan"),

            "TPR@FPR=1%": 0.0,
            "TPR@FPR=5%": 0.0,
            "TPR@FPR=10%": 0.0,
            "ACC@FPR=1%": float("nan"),
            "ACC@FPR=5%": float("nan"),
            "ACC@FPR=10%": float("nan"),
            "thr@FPR=1%": float("nan"),
            "thr@FPR=5%": float("nan"),
            "thr@FPR=10%": float("nan"),
            "FPR@thr(FPR=1%)": float("nan"),
            "FPR@thr(FPR=5%)": float("nan"),
            "FPR@thr(FPR=10%)": float("nan"),

            "EER": float("nan"),
            "EER_threshold": float("nan"),
            "EER_FPR": float("nan"),
            "EER_FNR": float("nan"),

            "skipped": True,
            "reason": "no_valid_scores"
        }
        return {"summary": summary, "lines": out_lines}

    # --- choose threshold ---
    if threshold_mode == "calib_neg":
        thr_used = threshold_from_neg(score_neg, calib_fpr)
        thr_src = f"neg_quantile@FPR={calib_fpr}"
    else:
        thr_used = float(z_threshold)
        thr_src = "fixed"

    # --- fill per-line is_substitution using thr_used ---
    for o in out_lines:
        sc = float(o.get("score_substitution", 0.0))
        o["is_substitution"] = bool(sc >= thr_used)
        o["threshold_used"] = float(thr_used)
        o["threshold_source"] = thr_src

    # 1) fixed-threshold metrics (using thr_used)
    tpr, fpr, fnr, tnr, tp, fp, tn, fn = tpr_fpr_fnr_tnr_at_threshold(score_pos, score_neg, thr_used)

    # 2) Exact empirical ROC-AUC from every saved score.
    auc = float(roc_auc_score(
        np.concatenate([np.ones(score_pos.size), np.zeros(score_neg.size)]),
        np.concatenate([score_pos, score_neg]),
    ))

    # 3) TPR@FPR points (independent of thr_used)
    op_1 = tpr_at_fpr(score_pos, score_neg, 0.01)
    op_5 = tpr_at_fpr(score_pos, score_neg, 0.05)
    op_10 = tpr_at_fpr(score_pos, score_neg, 0.10)

    # ACC@FPR points: compute ACC at the same operating thresholds returned by tpr_at_fpr
    m_1 = metrics_at_threshold(score_pos, score_neg, float(op_1["threshold"]))
    m_5 = metrics_at_threshold(score_pos, score_neg, float(op_5["threshold"]))
    m_10 = metrics_at_threshold(score_pos, score_neg, float(op_10["threshold"]))

    # EER
    eer = eer_from_scores(score_pos, score_neg)

    summary = {
        "setting": setting_name,
        "declared_model": declared_id,
        "actual_model_positive": actual_id,
        "total_pos_eval": int(total_pos_eval),
        "total_neg_eval": int(total_neg_eval),

        "score_mode": score_mode,
        "threshold_mode": threshold_mode,
        "threshold": float(z_threshold),
        "threshold_used": float(thr_used),
        "threshold_source": thr_src,

        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "TNR": float(tnr),
        "AUC": float(auc),

        "TPR@FPR=1%": float(op_1["tpr"]),
        "TPR@FPR=5%": float(op_5["tpr"]),
        "TPR@FPR=10%": float(op_10["tpr"]),

        "ACC@FPR=1%": float(m_1["acc"]),
        "ACC@FPR=5%": float(m_5["acc"]),
        "ACC@FPR=10%": float(m_10["acc"]),

        "thr@FPR=1%": float(op_1["threshold"]),
        "thr@FPR=5%": float(op_5["threshold"]),
        "thr@FPR=10%": float(op_10["threshold"]),

        # achieved FPR at those thresholds (<= target due to discreteness)
        "FPR@thr(FPR=1%)": float(m_1["fpr"]),
        "FPR@thr(FPR=5%)": float(m_5["fpr"]),
        "FPR@thr(FPR=10%)": float(m_10["fpr"]),

        "EER": float(eer["eer"]),
        "EER_threshold": float(eer["threshold"]),
        "EER_FPR": float(eer["fpr"]),
        "EER_FNR": float(eer["fnr"]),
    }
    return {"summary": summary, "lines": out_lines}


def parse_settings(settings_args: List[str], model_ids: List[str]) -> List[Tuple[str, str]]:
    if not settings_args:
        out = []
        for d in model_ids:
            for a in model_ids:
                if d != a:
                    out.append((d, a))
        return out

    out = []
    for s in settings_args:
        s = s.strip()
        if not s:
            continue
        if "->" not in s:
            raise ValueError(f"Bad setting '{s}', expected 'declared->actual'")
        d, a = s.split("->", 1)
        d, a = d.strip(), a.strip()
        if d in model_ids and a in model_ids and d != a:
            out.append((d, a))
    if not out:
        raise ValueError("No valid settings parsed from --settings.")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="一个或多个 JSONL 路径或通配符")
    parser.add_argument("--candidate-config", type=str, required=True, help="candidate-config.json")
    parser.add_argument("--out", type=str,
                        default="data/detect_substitution_orth_student.jsonl",
                        help="输出 JSONL（行级结果 + 末尾 _summary）")
    parser.add_argument("--field", type=str, default="summary_student",
                        help="使用哪个字段做检测：summary_student（默认）或 summary（teacher）")
    parser.add_argument("--per-model", type=int, default=0, help="每个模型抽样条数（0=全量）")

    parser.add_argument("--z-threshold", type=float, default=4.0,
                        help="fixed 阈值（threshold_mode=fixed 时使用）")

    parser.add_argument("--threshold-mode", type=str, default="fixed",
                        choices=["fixed", "calib_neg"],
                        help="fixed: 用 z-threshold；calib_neg: 用 negatives 反推阈值(每 setting)")

    parser.add_argument("--calib-fpr", type=float, default=0.01,
                        help="threshold_mode=calib_neg 时目标 FPR（建议 0.01/0.05/0.1）")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=0, help="过滤过短文本（按候选 tokenizer 的最小 token 数）")

    parser.add_argument("--score-mode", type=str, default="sign", choices=["sign", "diff"],
                        help="sign=旧脚本；diff=更稳的替换分数 max_other_z - z_declared")

    parser.add_argument("--settings", type=str, nargs="*", default=[],
                        help="替换设定列表 declared->actual；不提供则默认所有有序对")
    args = parser.parse_args()

    files = resolve_paths(args.inputs)
    if not files:
        print("[ERROR] no input files.", file=sys.stderr)
        sys.exit(1)

    all_recs = list(read_jsonl_many(files))
    if not all_recs:
        print("[ERROR] no records in inputs.", file=sys.stderr)
        sys.exit(1)

    cand_items = load_candidate_config(args.candidate_config)
    cands = [Candidate(c["model_id"], c["tokenizer"], c["orth_mask"]) for c in cand_items]
    model_ids = [c.model_id for c in cands]

    buckets = group_by_model(all_recs)
    for m in model_ids:
        buckets.setdefault(m, [])

    sampled = {m: rand_sample(buckets[m], args.per_model, seed=args.seed) for m in model_ids}
    settings = parse_settings(args.settings, model_ids)

    results = []
    for declared, actual in settings:
        setting_name = f"{declared}->{actual}"
        res = eval_setting(
            setting_name=setting_name,
            declared_id=declared,
            actual_id=actual,
            samples_declared_from=sampled[declared],
            samples_actual_from=sampled[actual],
            cands=cands,
            field=args.field,
            z_threshold=args.z_threshold,
            min_tokens=args.min_tokens,
            score_mode=args.score_mode,
            threshold_mode=args.threshold_mode,
            calib_fpr=args.calib_fpr,
        )
        results.append(res)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for res in results:
            for line in res["lines"]:
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
        for res in results:
            fout.write(json.dumps({"_summary": res["summary"]}, ensure_ascii=False) + "\n")

    for res in results:
        print(json.dumps(res["summary"], ensure_ascii=False, indent=2))

    print(f"[done] wrote -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
