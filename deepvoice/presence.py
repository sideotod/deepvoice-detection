"""PANNs-based voice and music presence inference."""

import json
import shutil
from pathlib import Path

import numpy as np

from deepvoice.audio import (
    AUDIO_SAMPLE_RATE,
    PANNS_SAMPLE_RATE,
    extract_segment,
    get_segment_starts,
    load_audio,
)


def prepare_panns_labels(panns_dir: Path) -> None:
    """Install PANNs' label file at the package's expected local path."""
    source = panns_dir / "class_labels_indices.csv"
    if not source.is_file():
        raise FileNotFoundError(f"PANNs labels not found: {source}")
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(panns_dir: Path, device) -> tuple[object, list[int], list[int]]:
    """Load the local PANNs checkpoint and configured AudioSet label groups."""
    prepare_panns_labels(panns_dir)
    from panns_inference import AudioTagging, labels

    checkpoint = panns_dir / "Cnn14_mAP=0.431.pth"
    model = AudioTagging(checkpoint_path=str(checkpoint), device=device.type)
    label_groups = json.loads(
        (panns_dir / "component_labels.json").read_text(encoding="utf-8")
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    try:
        voice_indices = [label_to_index[label] for label in label_groups["voice"]]
        music_indices = [label_to_index[label] for label in label_groups["music"]]
    except KeyError as error:
        raise ValueError(f"Invalid PANNs component label configuration: {error}") from error
    return model, voice_indices, music_indices


def make_panns_segments(audio: np.ndarray) -> np.ndarray:
    """Convert detector segments from 16 kHz to PANNs' 32 kHz input rate."""
    import librosa

    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        resampled = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(resampled.astype(np.float32))
    return np.stack(segments)


def predict_presence(
    model: object, voice_indices: list[int], music_indices: list[int], audio: np.ndarray
) -> tuple[float, float]:
    """Return maximum voice and music existence probabilities across the file."""
    predictions, _ = model.inference(make_panns_segments(audio))
    return (
        float(predictions[:, voice_indices].max()),
        float(predictions[:, music_indices].max()),
    )


def predict_presence_for_all_files(
    audio_files: list[Path], panns_dir: Path, device
) -> dict[str, tuple[float, float]]:
    """Infer component presence scores while PANNs is the only loaded GPU model."""
    import torch
    from tqdm import tqdm

    model, voice_indices, music_indices = load_panns_model(panns_dir, device)
    scores = {}
    for audio_path in tqdm(audio_files, desc="Presence"):
        scores[audio_path.stem] = predict_presence(
            model, voice_indices, music_indices, load_audio(audio_path)
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores
