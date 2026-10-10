# SUMMARK

SUMMARK detects model substitution from reasoning summaries. Each model identity
carries a distinct watermark greenlist, and a LoRA summary adapter trained with a
watermark and a separation loss keeps that carrier detectable after
summarization. A summary is scored against every candidate identity, and a
declared model that scores below the best alternative indicates substitution.

This repository reproduces the main substitution detection experiment across
Math, General, and Code, with BERT, GPT-2, and T5 classifiers as baselines.

## Setup

Python 3.11; run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch CUDA build matching your GPU. Model and dataset downloads need
network access, and `meta-llama/Llama-3.2-3B-Instruct` requires Hugging Face
access approval.

## Run

```bash
python run_substitution_detection.py mask
for dataset in math general code; do
  for stage in teacher train student baseline eval; do
    python run_substitution_detection.py "$stage" --dataset "$dataset"
  done
done
python summarize_detection_results.py > substitution_detection_results.csv
```

The stages build the shared identity carriers, generate teacher data, train the
summary adapters, generate student summaries, train the baselines, and evaluate
detection. Append `--dry-run` to any stage to print its commands without running
them or creating files.

Each stage covers all four identities; `--model-id` restricts `teacher`, `train`,
and `student` to one. `--max-samples` lowers the teacher sample count (default
5,000) for pipeline checks, which does not give comparable results.

The mask is built once and reused by every stage. The runner refuses to overwrite
existing masks, teacher data, adapters, student summaries, or baseline
checkpoints; evaluation outputs can be regenerated. To reuse teacher data, place
exactly one JSONL file per identity at
`data/teacher/<dataset>/<model-id>/<run-id>/teacher.jsonl`, and keep the original
mask and seed.

## Configuration

| Identity | Model |
| --- | --- |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen-3b` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `llama-3b` | `meta-llama/Llama-3.2-3B-Instruct` |

| Domain | Dataset |
| --- | --- |
| Math | `open-r1/OpenR1-Math-220k` |
| General | `glaiveai/reasoning-v1-20m` |
| Code | `open-r1/OpenThoughts-114k-Code_decontaminated` |


### Hyperparameters

`run_substitution_detection.py` holds the authoritative values; the table below
is a reading aid. Note that two unrelated quantities are both 0.3:
`--green-ratio` is a fraction of the vocabulary, while `--wm-gamma` is an
additive logit offset.

| Flag | Value | Meaning |
| --- | --- | --- |
| `--green-ratio` | 0.3 | Fraction of the vocabulary in each identity's greenlist |
| `--max-overlap` | 0.3 | Cap on pairwise greenlist intersection |
| `--wm-gamma` | 0.3 | Logit bonus added to green tokens during teacher generation |
| `--wm-delta` | 2.0 | Logit penalty subtracted from red tokens |
| `--temperature` | 0.7 | Teacher sampling temperature |
| `--lr` | 2e-4 | Adapter learning rate |
| `--wm-lambda` / `--sep-lambda` | 0.5 / 0.5 | Watermark and separation loss weights |
| `--seed` | 42 | Sampling and training |


## Code layout

| Path | Role |
| --- | --- |
| `run_substitution_detection.py` | Pipeline stages and all hyperparameters |
| `mask_greenlist.py` | Builds the orthogonal identity greenlists |
| `run_watermark{,_general,_code}_s4.py` | Teacher generation, per domain |
| `fine-tune/train_summary_lora*.py` | Summary adapter training |
| `fine-tune/gen_student_summaries*.py` | Student summary generation |
| `fine-tune/eval_orth.py` | Identity scoring and detection metrics |
| `baselines/text_classify.py` | BERT, GPT-2, and T5 baselines |
| `summarize_detection_results.py` | Aggregates the reported table |



## Citation

```bibtex
@misc{summark,
  title  = {},
  author = {},
  year   = {},
  eprint = {},
}
```

## License and attribution

Apache-2.0; see `LICENSE.md`. `extended_watermark_processor.py`, `normalizers.py`,
`alternative_prf_schemes.py`, `homoglyphs.py`, and `homoglyph_data/` are vendored
from [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
(Kirchenbauer et al., *A Watermark for Large Language Models*).
