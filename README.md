# DeepVoice Detection

[한국어 안내](README.ko.md)

Offline inference pipeline for DACON's DeepVoice detection competition. The
original notebook is retained as the baseline reference; `script.py` is the
submission entry point.

## Quick Start

Place the local model assets and competition inputs in this layout:

```text
model/
  df_arena_1b/
  htdemucs/
  panns/
data/
  test/
  sample_submission.csv
```

Generate the three file-level fusion candidates with one shared inference run:

```bash
python script.py --all-fusions
```

This creates the following files:

```text
output/submission_baseline.csv
output/submission_component_max.csv
output/submission_soft_or.csv
```

To generate the evaluator-compatible `output/submission.csv` for one method:

```bash
python script.py --fusion baseline
python script.py --fusion component_max
python script.py --fusion soft_or
```

Build the three upload-ready archives after populating `model/`:

```bash
python scripts/build_submissions.py
```

## Project Structure

```text
.
├── [Baseline_Inference]_*.ipynb  # Original baseline notebook
├── script.py                     # DACON inference entry point
├── deepvoice/                    # Reusable inference modules
│   ├── audio.py                  # Audio loading and segmentation
│   ├── presence.py               # PANNs component-presence scoring
│   ├── separation.py             # HTDemucs source separation
│   ├── detector.py               # DF-Arena fake scoring
│   ├── fusion.py                 # Three file-level fusion methods
│   ├── pipeline.py               # End-to-end inference flow
│   └── metrics.py                # Local ADS/CPS metric helpers
├── scripts/
│   └── build_submissions.py      # Builds one DACON zip per fusion method
├── model/                        # Local assets; excluded from Git
└── data/                         # Competition inputs; excluded from Git
```

The available fusion methods are `baseline`, `component_max`, and `soft_or`.
