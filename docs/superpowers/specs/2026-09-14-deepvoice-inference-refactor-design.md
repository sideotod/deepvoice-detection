# DeepVoice inference refactor and fusion experiment design

## Goal

Keep the supplied baseline notebook unchanged while extracting its inference
logic into testable Python modules.  The resulting code must produce DACON's
`output/submission.csv` and support three deterministic file-level fusion
strategies for separate submissions:

- `baseline`: `max(voice_present * voice_fake, music_present * music_fake)`
- `component_max`: `max(voice_fake, music_fake)`
- `soft_or`: `1 - (1 - voice_present * voice_fake) * (1 - music_present * music_fake)`

No learned fusion model or MLP is in scope.

## Architecture

`script.py` is the only executable submission entry point and imports the
implementation from `deepvoice/`:

```text
script.py
  -> deepvoice.pipeline.run_inference
       -> audio / presence / separation / detector
       -> fusion
       -> CSV writer
```

Modules:

- `deepvoice/audio.py`: audio loading and fixed-length segment helpers.
- `deepvoice/presence.py`: PANNs setup and voice/music presence inference.
- `deepvoice/separation.py`: HTDemucs model loading and component separation.
- `deepvoice/detector.py`: DF-Arena model loading and per-component scoring.
- `deepvoice/fusion.py`: named, pure fusion functions and validation.
- `deepvoice/pipeline.py`: input validation, model lifecycle, prediction loop,
  and submission writing.
- `deepvoice/metrics.py`: official score calculation helpers for future local
  validation (not required at inference time).

The original notebook remains unmodified as the baseline reference.

## Submission and experiments

`script.py --fusion <name>` writes exactly `output/submission.csv`, satisfying
the evaluator contract. The three submissions are created by running the same
package three times, once for each fusion name, and preserving each resulting
CSV or zip as an experiment artifact before the next run.

The default is `baseline` so a no-argument evaluator invocation retains the
provided baseline behavior. CLI arguments are honored locally; this fixes the
notebook code's `parse_arguments([])` behavior that discards supplied flags.

## Error handling and compatibility

- Inputs, sample-submission columns, and ID alignment are validated before
  model inference.
- The pipeline uses only model files stored beneath `model/` and keeps
  Hugging Face offline flags enabled.
- CUDA remains the default device because the competition evaluator supplies an
  L4 GPU, with an explicit CPU option for local checks.
- The PANNs model is released before DF-Arena/HTDemucs are loaded to limit VRAM
  pressure.

## Testing

Tests are written before the implementation and cover:

1. Each fusion's formula and probability bounds.
2. Invalid fusion names.
3. Segment boundary behavior.
4. Official metric helpers, including component-subset EER and weighted score.
5. CLI argument parsing without loading heavyweight models.

Full end-to-end inference is not run until the local `model/` and `data/`
assets are available.
