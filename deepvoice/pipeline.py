"""End-to-end inference and DACON submission CSV generation."""

import csv
from pathlib import Path

from deepvoice.detector import load_df_arena_model, predict_fake
from deepvoice.fusion import FUSION_NAMES, combine_file_fake_score
from deepvoice.presence import predict_presence_for_all_files
from deepvoice.separation import load_htdemucs_model, separate_voice_and_music

PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)
SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


def select_device(device_name: str):
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; rerun with --device cpu")
    return torch.device(device_name)


def find_audio_files(test_dir: Path) -> list[Path]:
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    audio_files = sorted(
        (
            path
            for path in test_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ),
        key=lambda path: path.stem,
    )
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in {test_dir}")
    ids = [path.stem for path in audio_files]
    if len(ids) != len(set(ids)):
        raise ValueError("Audio IDs must be unique across supported file extensions")
    return audio_files


def read_sample_submission(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)
    if columns is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")
    required = {"ID", *PREDICTION_COLUMNS}
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"Sample submission is missing columns: {missing}")
    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id
    return columns, rows


def order_audio_files(audio_files: list[Path], rows: list[dict[str, str]]) -> list[Path]:
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in rows]
    missing = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing or extra:
        raise ValueError(
            "Test audio and sample-submission IDs do not match. "
            f"Missing: {missing[:5]}, Extra: {extra[:5]}"
        )
    return [audio_by_id[audio_id] for audio_id in submission_ids]


def _predict_component_scores(
    audio_files: list[Path], model_dir: Path, device
) -> list[dict[str, float]]:
    """Run the expensive models once and retain the four component scores."""
    from tqdm import tqdm

    presence_scores = predict_presence_for_all_files(audio_files, model_dir / "panns", device)
    df_arena_model, fake_label_index = load_df_arena_model(
        model_dir, model_dir / "df_arena_1b", device
    )
    htdemucs_model = load_htdemucs_model(model_dir / "htdemucs")
    component_scores = []
    for audio_path in tqdm(audio_files, desc="Components"):
        voice_audio, music_audio = separate_voice_and_music(audio_path, htdemucs_model, device)
        voice_fake = predict_fake(df_arena_model, fake_label_index, voice_audio, device)
        music_fake = predict_fake(df_arena_model, fake_label_index, music_audio, device)
        voice_present, music_present = presence_scores[audio_path.stem]
        component_scores.append(
            {
                "voice_fake": voice_fake,
                "music_fake": music_fake,
                "voice_present": voice_present,
                "music_present": music_present,
            }
        )
    return component_scores


def _write_submission(
    output_path: Path,
    columns: list[str],
    source_rows: list[dict[str, str]],
    component_scores: list[dict[str, float]],
    fusion_name: str,
) -> None:
    rows = []
    for source_row, scores in zip(source_rows, component_scores, strict=True):
        row = source_row.copy()
        row["FILE_FAKE_PROB"] = round(
            combine_file_fake_score(fusion_name, **scores), 10
        )
        row["VOICE_FAKE_PROB"] = round(scores["voice_fake"], 10)
        row["MUSIC_FAKE_PROB"] = round(scores["music_fake"], 10)
        row["VOICE_PRESENT_PROB"] = round(scores["voice_present"], 10)
        row["MUSIC_PRESENT_PROB"] = round(scores["music_present"], 10)
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_inference(
    *,
    test_dir: Path,
    sample_submission: Path,
    output_path: Path,
    model_dir: Path,
    device_name: str,
    fusion_name: str,
) -> None:
    """Generate one evaluator-compatible `submission.csv` with the chosen fusion."""
    if fusion_name not in FUSION_NAMES:
        raise ValueError(f"Unknown fusion method: {fusion_name}")
    device = select_device(device_name)
    columns, rows = read_sample_submission(sample_submission)
    audio_files = order_audio_files(find_audio_files(test_dir), rows)
    scores = _predict_component_scores(audio_files, model_dir, device)
    _write_submission(output_path, columns, rows, scores, fusion_name)
    print(f"Saved {len(rows)} {fusion_name} predictions to {output_path}")


def run_all_fusions(
    *,
    test_dir: Path,
    sample_submission: Path,
    output_dir: Path,
    model_dir: Path,
    device_name: str,
) -> list[Path]:
    """Generate all three fusion CSVs after one shared model-inference pass."""
    device = select_device(device_name)
    columns, rows = read_sample_submission(sample_submission)
    audio_files = order_audio_files(find_audio_files(test_dir), rows)
    scores = _predict_component_scores(audio_files, model_dir, device)
    paths = []
    for fusion_name in FUSION_NAMES:
        output_path = output_dir / f"submission_{fusion_name}.csv"
        _write_submission(output_path, columns, rows, scores, fusion_name)
        paths.append(output_path)
    print(f"Saved {len(rows)} predictions for: {', '.join(FUSION_NAMES)}")
    return paths
