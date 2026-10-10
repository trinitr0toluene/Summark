#!/usr/bin/env python3
"""Run the SUMMARK model-substitution detection experiment pipeline."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = {'qwen-3b': 'Qwen/Qwen2.5-3B-Instruct',
          'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
          'llama-3b': 'meta-llama/Llama-3.2-3B-Instruct',
          'qwen-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct'}
PAIRS = ['qwen-7b->qwen-3b', 'qwen-7b->qwen-1.5b', 'qwen-3b->qwen-1.5b',
         'qwen-3b->llama-3b', 'llama-3b->qwen-3b', 'qwen-7b->llama-3b', 'llama-3b->qwen-1.5b']
SUFFIX = {'math': '', 'general': '_general', 'code': '_code'}
TEACHER = {'math': 'run_watermark_s4.py', 'general': 'run_watermark_general_s4.py', 'code': 'run_watermark_code_s4.py'}
MASK = 'data/secure/orthogonal_greenlists_qwen.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['mask', 'teacher', 'train', 'student', 'baseline', 'eval'])
    p.add_argument('--dataset', choices=SUFFIX, default='math')
    p.add_argument('--model-id', choices=MODELS, help='Default: all four models')
    p.add_argument('--max-samples', type=int, default=5000, help='Teacher samples per model')
    p.add_argument('--seed', type=int, default=42, help='Seed for teacher sampling, adapter training, and splits')
    p.add_argument('--dry-run', action='store_true', help='Print commands without executing or creating files')
    a = p.parse_args()
    def run(*args):
        cmd = [sys.executable, *map(str, args)]
        print(shlex.join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    def teacher(mid):
        # Original generators place output under a timestamp directory.
        directory = ROOT / 'data/teacher' / a.dataset / mid
        matches = sorted(directory.glob('*/teacher.jsonl'))
        if len(matches) == 1:
            return str(matches[0].relative_to(ROOT))
        if a.dry_run and not matches:
            return str(directory.relative_to(ROOT) / 'TIMESTAMP/teacher.jsonl')
        raise SystemExit(f'Expected exactly one teacher JSONL in {directory}; found {len(matches)}. Keep the intended run only.')
    mids = [a.model_id] if a.model_id else list(MODELS)
    suffix = SUFFIX[a.dataset]
    if a.stage == 'mask':
        if (ROOT / MASK).exists():
            raise SystemExit('Mask already exists; reuse it for all stages. Refusing to replace it.')
        if not a.dry_run:
            (ROOT / MASK).parent.mkdir(parents=True, exist_ok=True)
        run('mask_greenlist.py', '--tokenizer', MODELS['qwen-3b'], '--model-ids', *MODELS,
            '--green-ratio', '0.3', '--max-overlap', '0.3', '--seed', '1234', '--out', MASK)
    elif a.stage in ('teacher', 'train', 'student'):
        for mid in mids:
            adapter = f'fine-tune/adapters/{a.dataset}/{mid}'
            if a.stage == 'teacher':
                if list((ROOT / 'data/teacher' / a.dataset / mid).glob('*/teacher.jsonl')):
                    raise SystemExit(f'Teacher output already exists for {mid}; refusing duplicate generation.')
                run(TEACHER[a.dataset], '--model', MODELS[mid], '--model-id', mid, '--wm', 'on',
                    '--orth-mask', MASK, '--wm-gamma', '0.3', '--wm-delta', '2.0',
                    '--max-samples', a.max_samples, '--temperature', '0.7', '--seed', a.seed,
                    '--out', f'data/teacher/{a.dataset}/{mid}/teacher.jsonl')
            elif a.stage == 'train':
                if (ROOT / adapter).exists():
                    raise SystemExit(f'Adapter output exists: {adapter}')
                run(f'fine-tune/train_summary_lora{suffix}.py', '--base-model', MODELS[mid],
                    '--model-id', mid, '--teacher-jsonl', teacher(mid), '--orth-mask', MASK,
                    '--out-adapter', adapter, '--wm-lambda', '0.5', '--sep-lambda', '0.5',
                    '--epochs', '1', '--lr', '2e-4', '--train-ratio', '0.8', '--seed', a.seed)
            else:
                output = f'data/student/{a.dataset}/{mid}.jsonl'
                if (ROOT / output).exists():
                    raise SystemExit(f'Student output exists: {output}')
                run(f'fine-tune/gen_student_summaries{suffix}.py', '--base-model', MODELS[mid],
                    '--adapter', adapter, '--in-jsonl', teacher(mid), '--out-jsonl', output,
                    '--stage', 'both', '--split', 'eval', '--train-ratio', '0.8', '--seed', a.seed,
                    '--sum-temperature', '0.7')
    elif a.stage == 'baseline':
        inputs = [teacher(mid) for mid in MODELS]
        for backbone, name, pool, batch in [('bert', 'bert-base-uncased', 'cls', 32), ('gpt2', 'gpt2', 'last', 16), ('t5', 't5-base', 'mean', 32)]:
            out = f'baselines/ckpt/{a.dataset}/textclf_{backbone}'
            if (ROOT / out).exists():
                raise SystemExit(f'Baseline output exists: {out}')
            # Classifier training ratios used by the experiment recipe.
            ratio = '0.2' if a.dataset == 'math' and backbone != 'gpt2' else '0.8'
            run('baselines/text_classify.py', '--inputs', *inputs, '--text_field', 'summary',
                '--label_field', 'model_id', '--split_by', 'id', '--train_ratio', ratio, '--val_ratio', '0.1',
                '--backbone', backbone, '--backbone_name', name, '--pool', pool, '--batch', batch,
                '--epochs', '10', '--lr', '1e-3', '--out_dir', out, '--seed', a.seed,
                '--pairs', ','.join(x.replace('->', ':') for x in PAIRS))
    else:
        run('fine-tune/eval_orth.py', '--inputs', *[f'data/student/{a.dataset}/{mid}.jsonl' for mid in MODELS],
            '--candidate-config', 'fine-tune/candidate-config.json', '--field', 'summary_student',
            '--score-mode', 'diff', '--threshold-mode', 'calib_neg', '--calib-fpr', '0.01',
            '--settings', *PAIRS, '--out', f'results/{a.dataset}/summark.jsonl')

if __name__ == '__main__':
    main()
