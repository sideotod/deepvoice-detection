# DeepVoice Inference Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the notebook baseline into a testable, DACON-compatible Python inference package with three selectable file-level fusion methods.

**Architecture:** `script.py` is the submission entry point and delegates input validation, model inference, fusion, and CSV writing to focused `deepvoice` modules. Heavy model imports remain lazy so pure fusion, metric, and CLI tests run without model files or GPU dependencies.

**Tech Stack:** Python 3.11, pytest, NumPy, scikit-learn, PyTorch, librosa, torchaudio, Demucs, PANNs inference.

**Spec:** `docs/superpowers/specs/2026-09-14-deepvoice-inference-refactor-design.md`

## Global Constraints

- Keep the supplied notebook unchanged as the reference baseline.
- Keep all Hugging Face use offline and load model assets only from `model/`.
- `script.py` must default to `baseline` and write `output/submission.csv`.
- Support exactly `baseline`, `component_max`, and `soft_or`; no MLP or learned fusion.
- Preserve the original PANNs → HTDemucs → DF-Arena lifecycle and release PANNs before the later models load.
- Do not run end-to-end inference without local `model/` and `data/` assets.

---

### Task 1: Fusion module

**Files:**
- Create: `tests/test_fusion.py`
- Create: `deepvoice/__init__.py`
- Create: `deepvoice/fusion.py`

**Interfaces:**
- Produces: `FUSION_NAMES: tuple[str, ...]`, `combine_file_fake_score(name: str, voice_fake: float, music_fake: float, voice_present: float, music_present: float) -> float`.

- [ ] **Step 1: Write failing fusion tests**

```python
assert combine_file_fake_score("baseline", 0.8, 0.3, 0.5, 0.9) == 0.4
assert combine_file_fake_score("component_max", 0.8, 0.3, 0.5, 0.9) == 0.8
assert combine_file_fake_score("soft_or", 0.8, 0.3, 0.5, 0.9) == 0.562
```

- [ ] **Step 2: Run the test to verify it fails because the module is absent**

Run: `pytest tests/test_fusion.py -v`

- [ ] **Step 3: Implement the three pure functions and input validation**

```python
if name == "baseline":
    return max(voice_present * voice_fake, music_present * music_fake)
if name == "component_max":
    return max(voice_fake, music_fake)
if name == "soft_or":
    return 1 - (1 - voice_present * voice_fake) * (1 - music_present * music_fake)
raise ValueError(...)
```

- [ ] **Step 4: Run the fusion tests and verify they pass**

Run: `pytest tests/test_fusion.py -v`

### Task 2: Audio helpers and official metrics

**Files:**
- Create: `tests/test_audio.py`
- Create: `tests/test_metrics.py`
- Create: `deepvoice/audio.py`
- Create: `deepvoice/metrics.py`

**Interfaces:**
- Produces: `get_segment_starts(audio_length: int, segment_samples: int) -> list[int]`, `extract_segment(audio: np.ndarray, start: int, segment_samples: int) -> np.ndarray`, and `competition_score(...) -> dict[str, float]`.

- [ ] **Step 1: Write failing boundary and metric tests**

```python
assert get_segment_starts(10, 4) == [0, 4, 6]
assert extract_segment(np.array([1, 2]), 0, 5).tolist() == [1, 2, 1, 2, 1]
```

- [ ] **Step 2: Run tests to verify they fail because helpers are absent**

Run: `pytest tests/test_audio.py tests/test_metrics.py -v`

- [ ] **Step 3: Implement helpers and EER/AUC weighted official metric calculation**

- [ ] **Step 4: Run the tests and verify they pass**

Run: `pytest tests/test_audio.py tests/test_metrics.py -v`

### Task 3: Model adapters and pipeline

**Files:**
- Create: `deepvoice/presence.py`
- Create: `deepvoice/separation.py`
- Create: `deepvoice/detector.py`
- Create: `deepvoice/pipeline.py`

**Interfaces:**
- Produces: `run_inference(test_dir: Path, sample_submission: Path, output: Path, device_name: str, fusion_name: str) -> None`.

- [ ] **Step 1: Port the notebook's PANNs, HTDemucs, and DF-Arena logic into model-specific modules without changing its numerical behavior**

- [ ] **Step 2: Add pipeline integration that calculates the selected fusion score and writes five required CSV fields**

- [ ] **Step 3: Run pure tests to ensure lazy imports keep them independent of model assets**

Run: `pytest tests -v`

### Task 4: Submission CLI

**Files:**
- Create: `tests/test_cli.py`
- Create: `script.py`

**Interfaces:**
- Produces: `parse_arguments(argv: list[str] | None = None) -> argparse.Namespace`, with `--fusion` choices from `FUSION_NAMES`.

- [ ] **Step 1: Write a failing test that parses `--fusion soft_or --device cpu`**

```python
args = parse_arguments(["--fusion", "soft_or", "--device", "cpu"])
assert args.fusion == "soft_or"
assert args.device == "cpu"
```

- [ ] **Step 2: Run the test to verify it fails because the script is absent**

Run: `pytest tests/test_cli.py -v`

- [ ] **Step 3: Implement the thin CLI entry point and forward arguments to `run_inference`**

- [ ] **Step 4: Run all tests and compile the package**

Run: `pytest tests -v && python -m compileall -q deepvoice script.py`

### Task 5: Experiment instructions

**Files:**
- Create: `README.md`

**Interfaces:**
- Documents: three command lines that each produce a submission CSV and the required DACON package layout.

- [ ] **Step 1: Add setup, model/data layout, and three fusion-run commands**
- [ ] **Step 2: Confirm every documented CLI option is accepted without loading models**

Run: `python script.py --help`
