# DeepVoice inference

The original notebook remains the baseline reference. `script.py` is the
offline DACON inference entry point; its default behavior is the unchanged
baseline fusion.

## Required layout

```text
model/
  df_arena_1b/
  htdemucs/
  panns/
data/
  test/
  sample_submission.csv
```

## Generate the three fusion candidates once

This performs expensive PANNs, HTDemucs, and DF-Arena inference once, then
writes all file-level fusion alternatives:

```bash
python script.py --all-fusions
```

It creates:

```text
output/submission_baseline.csv
output/submission_component_max.csv
output/submission_soft_or.csv
```

## Generate one evaluator-compatible submission

```bash
python script.py --fusion baseline
python script.py --fusion component_max
python script.py --fusion soft_or
```

Each command writes `output/submission.csv`. Preserve or rename that CSV before
running the next command. For DACON upload, package `model/`, `script.py`,
and `requirements.txt` only.

## Build three upload-ready archives

```bash
python tools/build_submissions.py
```

This creates `dist/submit_baseline.zip`, `dist/submit_component_max.zip`, and
`dist/submit_soft_or.zip`. Each archive pins its own default fusion while
keeping the evaluator's top-level package layout unchanged. The shared Python
modules are placed under `model/deepvoice/` inside the archive.
