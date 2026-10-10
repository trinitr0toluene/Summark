#!/usr/bin/env python3
"""Summarize substitution detection metrics across seven directed model pairs."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
import sys
from run_substitution_detection import PAIRS, ROOT

METHODS = ('BERT', 'GPT-2', 'T5', 'SUMMARK')
DOMAINS = ('Math', 'General', 'Code')


def select(rows, pairs, setting, tpr, auc):
    keyed = {}
    for row in rows:
        key = row[setting]
        if key in keyed:
            raise ValueError(f'Duplicate setting: {key}')
        keyed[key] = row
    result = []
    for pair in pairs:
        if pair not in keyed:
            raise ValueError(f'Missing setting: {pair}')
        row = keyed[pair]
        for count in ('total_pos_eval', 'total_neg_eval'):
            if count in row and row[count] <= 0:
                raise ValueError(f'Empty evaluation samples: {pair}')
        values = (float(row[tpr]), float(row[auc]))
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f'Invalid TPR/AUC: {pair}')
        result.append(values)
    return mean(x[0] for x in result), min(x[0] for x in result), mean(x[1] for x in result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    a = p.parse_args()
    cells = {method: [] for method in METHODS}
    for domain in DOMAINS:
        for method, backbone in zip(METHODS, ('bert', 'gpt2', 't5', None)):
            if backbone:
                path = a.root / 'baselines/ckpt' / domain.lower() / f'textclf_{backbone}/metrics_pairs.json'
                rows = json.loads(path.read_text())
            else:
                path = a.root / 'results' / domain.lower() / 'summark.jsonl'
                rows = [o['_summary'] for line in path.read_text().splitlines() if line.strip()
                        for o in [json.loads(line)] if '_summary' in o]
            cells[method].extend(select(rows, PAIRS, 'setting', 'TPR@FPR=1%', 'AUC'))
    writer = csv.writer(sys.stdout)
    writer.writerow(['Method'] + [f'{d} {metric}' for d in DOMAINS for metric in ('AvgT', 'MinT', 'AUC')])
    for method in METHODS:
        writer.writerow([method] + [f'{v:.4f}' for v in cells[method]])

if __name__ == '__main__':
    main()
