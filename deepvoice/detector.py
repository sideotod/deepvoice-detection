"""DF-Arena component fake-risk inference."""

import os
import sys
from pathlib import Path

import numpy as np

from deepvoice.audio import extract_segment, get_segment_starts

SILENCE_RMS = 1e-5


def load_df_arena_model(model_dir: Path, df_arena_dir: Path, device):
    """Load the local DF-Arena model and return its spoof-class index."""
    import torch

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(df_arena_dir)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(df_arena_dir), local_files_only=True, low_cpu_mem_usage=True
        )
    finally:
        os.chdir(previous_directory)

    model = model.to(device).eval()
    return model, int(model.config.label2id["spoof"])


def calculate_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_fake(model, fake_label_index: int, audio: np.ndarray, device) -> float:
    """Return the maximum spoof probability over fixed-length audio segments."""
    import torch

    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    scores = []
    for start in get_segment_starts(audio.size):
        segment_tensor = torch.from_numpy(extract_segment(audio, start)).to(device)
        with torch.inference_mode():
            logits = model(input_values=segment_tensor)["logits"]
            probabilities = torch.softmax(logits.float(), dim=-1)
        scores.append(float(probabilities[0, fake_label_index]))
    return max(scores)
