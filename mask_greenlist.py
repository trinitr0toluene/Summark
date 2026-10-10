#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import argparse
import random
from typing import List, Dict

from transformers import AutoTokenizer


def build_masks(
    model_ids: List[str],
    vocab_ids: List[int],
    green_ratio: float = 0.5,
    max_overlap: float = 0.4,
    max_trials: int = 10,
    existing_masks: Dict[str, Dict[str, list]] | None = None,
) -> Dict[str, Dict[str, list]]:
    V = len(vocab_ids)
    target_green = int(green_ratio * V)
    max_intersection = int(max_overlap * target_green)

    masks: Dict[str, Dict[str, list]] = {}
    green_sets: List[set] = []
    model_order: List[str] = []

    if existing_masks:
        for mid, mask in existing_masks.items():
            green_ids = mask.get("green_ids", [])
            green_set = set(green_ids)
            if len(green_set) != target_green:
                raise ValueError(
                    f"Existing mask for {mid} has {len(green_set)} green ids, expected {target_green}."
                )
            masks[mid] = {
                "green_ids": sorted(green_set),
                "red_ids": sorted(list(set(vocab_ids) - green_set)),
            }
            green_sets.append(green_set)
            model_order.append(mid)

    for mid in model_ids:
        print(f"[build_masks] building greenlist for model_id={mid} ...")

        if not green_sets:
            S = set(random.sample(vocab_ids, target_green))
            print(f"  -> first model, picked random green set, size={len(S)}")
        else:
            for trial in range(max_trials):
                shuffled = vocab_ids[:]
                random.shuffle(shuffled)

                S = set()
                inter_counts = [0] * len(green_sets)

                for tok in shuffled:
                    if len(S) >= target_green:
                        break

                    violate = False
                    for j, prev_green in enumerate(green_sets):
                        if tok in prev_green:
                            if inter_counts[j] + 1 > max_intersection:
                                violate = True
                                break
                    if violate:
                        continue

                    S.add(tok)
                    for j, prev_green in enumerate(green_sets):
                        if tok in prev_green:
                            inter_counts[j] += 1

                if len(S) == target_green:
                    print(f"  -> success at trial={trial}, green_size={len(S)}")
                    for j, prev_mid in enumerate(model_order):
                        inter = inter_counts[j]
                        print(
                            f"     intersect with {prev_mid}: {inter} "
                            f"({inter/target_green:.3f} of green)"
                        )
                    break
                else:
                    print(
                        f"  [warn] trial {trial} failed: only got {len(S)} < {target_green}, retry..."
                    )
                    S = None

            if S is None or len(S) < target_green:
                raise RuntimeError(
                    f"Cannot construct green set for {mid} under max_overlap={max_overlap}; "
                    f"try relaxing max_overlap 或减小 green_ratio。"
                )

        green_sets.append(S)
        model_order.append(mid)
        red_ids = list(set(vocab_ids) - S)
        masks[mid] = {
            "green_ids": sorted(list(S)),
            "red_ids": sorted(red_ids),
        }

    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="用于构造词表的 tokenizer 名称，例如 Qwen/Qwen2.5-3B-Instruct",
    )
    ap.add_argument(
        "--model-ids",
        type=str,
        nargs="+",
        required=True,
        help="需要生成掩码的 model_id 列表，例如 qwen-3b qwen-1.5b llama-3b",
    )
    ap.add_argument(
        "--green-ratio",
        type=float,
        default=0.5,
        help="greenlist 占词表比例，默认 0.5",
    )
    ap.add_argument(
        "--max-overlap",
        type=float,
        default=0.3,
        help="任意两个 model 的 green 交集比例上限，默认 0.3",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="随机种子",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出 JSON 路径，例如 /home/xxx/orthogonal_greenlists.json",
    )
    ap.add_argument(
        "--base-json",
        type=str,
        default=None,
        help="已有 greenlist JSON；若提供，则只为缺失的 model_id 增量生成并保留已有条目",
    )

    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[info] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    vocab = tokenizer.get_vocab()
    vocab_ids = list(vocab.values())

    print(f"[info] vocab size = {len(vocab_ids)}")
    print(f"[info] model_ids = {args.model_ids}")

    existing_masks = None
    model_ids = args.model_ids
    if args.base_json:
        with open(args.base_json, "r", encoding="utf-8") as f:
            existing_masks = json.load(f)
        existing_ids = set(existing_masks.keys())
        requested_ids = list(dict.fromkeys(args.model_ids))
        model_ids = [mid for mid in requested_ids if mid not in existing_ids]
        print(f"[info] preserving existing model_ids = {sorted(existing_ids)}")
        print(f"[info] new model_ids to generate = {model_ids}")

    masks = build_masks(
        model_ids=model_ids,
        vocab_ids=vocab_ids,
        green_ratio=args.green_ratio,
        max_overlap=args.max_overlap,
        existing_masks=existing_masks,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(masks, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote orthogonal greenlists -> {args.out}")


if __name__ == "__main__":
    main()
