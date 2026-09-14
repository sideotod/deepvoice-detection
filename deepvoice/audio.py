"""Audio loading and fixed-length segment utilities."""

from pathlib import Path

import numpy as np

AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600


def load_audio(audio_path: Path) -> np.ndarray:
    """Load an audio file as finite 16 kHz mono float32 samples."""
    import librosa

    audio, _ = librosa.load(
        audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


def get_segment_starts(audio_length: int, segment_samples: int = SEGMENT_SAMPLES) -> list[int]:
    """Return contiguous window starts plus a final window covering the tail."""
    if audio_length <= 0:
        raise ValueError("audio_length must be positive")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if audio_length <= segment_samples:
        return [0]

    last_start = audio_length - segment_samples
    starts = list(range(0, last_start + 1, segment_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(
    audio: np.ndarray, start: int, segment_samples: int = SEGMENT_SAMPLES
) -> np.ndarray:
    """Return one detector-sized segment, repeating short audio when required."""
    if audio.size == 0:
        raise ValueError("audio must not be empty")
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if audio.size < segment_samples:
        repeat_count = segment_samples // audio.size + 1
        return np.tile(audio, repeat_count)[:segment_samples].astype(np.float32)
    return audio[start : start + segment_samples].astype(np.float32, copy=False)
