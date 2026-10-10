# SUMMARK

SUMMARK detects model substitution from reasoning summaries using identity-specific
watermark carriers and trained summary adapters. This repository provides the main
substitution detection experiment across Math, General, and Code, with BERT,
GPT-2, and T5 classifiers as comparison methods.

## Setup

Use Python 3.11 and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency versions reflect the development environment. Install a PyTorch
CUDA build compatible with your GPU. Model and dataset downloads require network
access; the Llama model requires Hugging Face access authorization. The experiments
reported in the paper used a 48 GB NVIDIA RTX 8000.

## Experiment configuration

The experiment uses four model identities:

| Identity | Model |
| --- | --- |
| `qwen-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen-3b` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `llama-3b` | `meta-llama/Llama-3.2-3B-Instruct` |

The seven directed substitutions, written as declared model → actual model, are:
Q7→Q3, Q7→Q1.5, Q3→Q1.5, Q3→L3, L3→Q3, Q7→L3, and L3→Q1.5.

| Domain | Dataset |
| --- | --- |
| Math | `open-r1/OpenR1-Math-220k` |
| General | `glaiveai/reasoning-v1-20m` |
| Code | `open-r1/OpenThoughts-114k-Code_decontaminated` |

### Hyperparameters

Two distinct quantities are both 0.3; they are unrelated. `--green-ratio` is the
*fraction of the vocabulary* assigned to each identity's greenlist, while
`--wm-gamma` is an *additive logit bonus*. The authoritative values are the
constants in `run_substitution_detection.py`; the table below is a reading aid.

| Stage | Flag | Value | Meaning |
| --- | --- | --- | --- |
| Carrier | `--green-ratio` | 0.3 | Fraction of the vocabulary in each greenlist |
| Carrier | `--max-overlap` | 0.3 | Cap on pairwise greenlist intersection |
| Carrier | `--seed` | 1234 | Greenlist construction seed |
| Teacher | `--wm-gamma` | 0.3 | Logit **bonus added to green** tokens |
| Teacher | `--wm-delta` | 2.0 | Logit **penalty subtracted from red** tokens |
| Teacher | `--max-samples` | 5000 | Teacher samples per model |
| Teacher | `--temperature` | 0.7 | Teacher sampling temperature |
| Adapter | `--epochs` / `--lr` | 1 / 2e-4 | Adapter training |
| Adapter | `--wm-lambda` / `--sep-lambda` | 0.5 / 0.5 | Watermark and separation loss weights |
| Student | `--sum-temperature` | 0.7 | Summary sampling temperature |
| All | `--seed` | 42 | Sampling, training, and the 80/20 split |

Note that `--wm-gamma` is a logit bonus for the fixed-greenlist carrier used in
this experiment, but the same flag is passed to the vendored KGW
`WatermarkDetector`, where `gamma` means a greenlist fraction. That detector only
populates the diagnostic `wm_detect` field and does not feed any reported metric.

Pass `--seed` to `run_substitution_detection.py` to change the seed for teacher
sampling, adapter training, student generation, the 80/20 split, and the
classifier baselines together.

## Run substitution detection

Preview commands with `--dry-run`:

```bash
python run_substitution_detection.py mask --dry-run
python run_substitution_detection.py teacher --dataset math --model-id qwen-3b --dry-run
```

Run the complete experiment:

```bash
python run_substitution_detection.py mask
for dataset in math general code; do
    python run_substitution_detection.py teacher --dataset "$dataset"
    python run_substitution_detection.py train --dataset "$dataset"
    python run_substitution_detection.py student --dataset "$dataset"
    python run_substitution_detection.py baseline --dataset "$dataset"
    python run_substitution_detection.py eval --dataset "$dataset"
done
python summarize_detection_results.py > substitution_detection_results.csv
```

The stages construct shared identity carriers, generate teacher examples, train
summary adapters, generate student summaries, train the classifier baselines,
and evaluate substitution detection. Each model stage processes all four
identities by default. Use `--model-id` to select one identity for `teacher`,
`train`, or `student`. `--max-samples` changes the teacher sample count; smaller
runs are useful for checking the pipeline but are not comparable full experiments.

The runner reuses the same mask for every stage. It refuses to overwrite existing
masks, teacher data, adapters, student summaries, or classifier checkpoints.
Evaluation outputs can be regenerated.

To use existing teacher examples, place one JSONL file per model at
`data/teacher/<dataset>/<model-id>/<run-id>/teacher.jsonl`. The runner requires
exactly one matching file per identity. The original generators create the
`<run-id>` timestamp directory automatically. For custom data sources, invoke the
corresponding generator with `--dataset-path`; `--dry-run` shows the other required
arguments. Reuse the matching masks, model IDs, and data splits when resuming stages.

## Evaluation outputs

| Output | Location |
| --- | --- |
| Student summaries | `data/student/<dataset>/<model-id>.jsonl` |
| SUMMARK per-example scores and per-pair metrics | `results/<dataset>/summark.jsonl` |
| Classifier per-pair metrics | `baselines/ckpt/<dataset>/textclf_<backbone>/metrics_pairs.json` |

`summarize_detection_results.py` reads these experiment outputs and reports the
mean TPR at nominal 1% FPR, minimum TPR across the seven pairs, and macro-average
ROC-AUC for each method and domain. It aggregates unrounded values before
formatting and rejects missing or duplicate settings, empty evaluation counts,
and invalid metric values. Use `--root` to summarize another experiment directory
with the same output layout.

SUMMARK scores substitution using `max_other_z - z_declared`. The detector uses
exact empirical ROC-AUC. Its `TPR@FPR=1%` is selected from the evaluated score ROC;
`--threshold-mode calib_neg` controls the separate threshold-based metrics, and
does not provide an independent calibration split for that ROC-selected TPR.
The classifier baselines select thresholds on validation data and apply them to
test data. These operating-point protocols should be distinguished when reporting
results; a nominal 1% FPR does not guarantee a 1% observed test FPR.

Classifier inputs use the teacher `summary` field. The recipe uses frozen
backbones, ten head-training epochs, learning rate 1e-3, and BERT CLS / T5 mean /
GPT-2 last-token pooling. Batch sizes are 32 / 32 / 16, respectively. The recorded
training ratios are 0.2 for Math BERT/T5 and 0.8 for the other domain/backbone
combinations, with validation ratio 0.1. Training filters and splitting logic are
preserved from the experiment scripts.

## Code layout

- `run_substitution_detection.py`: experiment stages and configuration.
- `summarize_detection_results.py`: aggregate detection metrics.
- `mask_greenlist.py`: construct identity carriers.
- `run_watermark_s4.py`, `run_watermark_general_s4.py`, `run_watermark_code_s4.py`:
  teacher generation for each domain.
- `fine-tune/train_summary_lora*.py`: summary adapter training.
- `fine-tune/gen_student_summaries*.py`: student inference.
- `fine-tune/eval_orth.py`, `fine-tune/candidate-config.json`: identity scoring and detection.
- `baselines/text_classify.py`: BERT, GPT-2, and T5 baselines.
- Watermark processor, normalization modules, and `homoglyph_data/`: dependencies
  of the retained generation scripts.

## Reproducibility

This release contains the experiment code, not pretrained adapters or generated
datasets. Masks are generated anew. Exact historical model/tokenizer and dataset
revisions are not pinned. Every stage, including teacher generation, is seeded
through the runner's `--seed`, but sampling on GPU is not bit-reproducible across
different hardware, driver, or library versions. Some historical run
configurations, particularly the General classifier settings, remain incompletely
documented. Consequently, this recipe supports
rerunning the experiment but has not been verified to match every published value.
Historical results also include a different AUC implementation and inconsistencies
between published aggregates and saved per-pair results; the current detector
retains exact empirical ROC-AUC. Original filtering and split behavior are retained,
and historical prompt-disjointness has not been independently verified.

Validation covers syntax, offline command imports, command construction, metric
aggregation, and preservation of the orthogonal-mask loss after dependency removal.
The full training and GPU inference experiment has not been rerun for this release.

### Detection-side vocabulary correction

Greenlists are constructed over one reference tokenizer's id space
(`Qwen/Qwen2.5-3B-Instruct`, 151,665 ids). A candidate with a smaller vocabulary
never observes the out-of-range ids, and the generator drops them as well, so the
null green rate `p` of the z-test must be computed from the truncated greenlist
over that candidate's own vocabulary.

Earlier code derived `p` from `tokenizer.vocab_size`, which reports the base
vocabulary and excludes added tokens, using the untruncated greenlist. For the
three Qwen identities the resulting error is about 0.2% and immaterial. For
`llama-3b` it set `p = 45499 / 128000 = 0.3555` against a true rate of 0.300, an
18% overstatement that biases that identity's z downward by roughly 1.3 to 2.6
for summaries of 128 to 512 tokens. Because the score is
`max_other_z - z_declared`, this inflated detection on the two pairs where
`llama-3b` is the declared model and depressed it on the two where it is the
actual model. `eval_orth.py` now uses the tokenizer's full id space and truncates
the greenlist to it, which brings all four identities to within 0.3% of the
generation-time rate. Numbers for the four pairs involving `llama-3b` therefore
differ from previously published values; the three Qwen-only pairs are unaffected.

The upstream license is preserved in `LICENSE.md`.
