"""Local reproduction of the competition's ADS, CPS, and total score."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _as_equal_length_arrays(
    y_true: Sequence[float], y_score: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    score = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or score.ndim != 1 or truth.size != score.size:
        raise ValueError("Labels and scores must be one-dimensional arrays of equal length")
    if truth.size == 0:
        raise ValueError("Labels and scores must not be empty")
    return truth, score


def equal_error_rate(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """Calculate EER with the exact threshold-selection rule in the contest."""
    truth, score = _as_equal_length_arrays(y_true, y_score)
    if np.unique(truth).size != 2:
        raise ValueError("EER requires both real and fake labels")
    fpr, tpr, _ = roc_curve(truth, score, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def competition_score(
    *,
    file_true: Sequence[float],
    file_score: Sequence[float],
    voice_true: Sequence[float],
    voice_score: Sequence[float],
    voice_present_true: Sequence[float],
    voice_present_score: Sequence[float],
    music_true: Sequence[float],
    music_score: Sequence[float],
    music_present_true: Sequence[float],
    music_present_score: Sequence[float],
) -> dict[str, float]:
    """Return the official file/component EERs, ADS, CPS, and total score."""
    file_true, file_score = _as_equal_length_arrays(file_true, file_score)
    voice_true, voice_score = _as_equal_length_arrays(voice_true, voice_score)
    music_true, music_score = _as_equal_length_arrays(music_true, music_score)
    voice_present_true, voice_present_score = _as_equal_length_arrays(
        voice_present_true, voice_present_score
    )
    music_present_true, music_present_score = _as_equal_length_arrays(
        music_present_true, music_present_score
    )

    expected_length = file_true.size
    arrays = (voice_true, music_true, voice_present_true, music_present_true)
    if any(values.size != expected_length for values in arrays):
        raise ValueError("All competition inputs must have the same length")

    voice_mask = voice_present_true == 1
    music_mask = music_present_true == 1
    file_eer = equal_error_rate(file_true, file_score)
    voice_eer = equal_error_rate(voice_true[voice_mask], voice_score[voice_mask])
    music_eer = equal_error_rate(music_true[music_mask], music_score[music_mask])
    voice_presence_auc = float(roc_auc_score(voice_present_true, voice_present_score))
    music_presence_auc = float(roc_auc_score(music_present_true, music_present_score))
    ads = 0.5 * (1.0 - file_eer) + 0.2 * (1.0 - voice_eer) + 0.3 * (1.0 - music_eer)
    cps = 0.5 * voice_presence_auc + 0.5 * music_presence_auc
    return {
        "file_eer": file_eer,
        "voice_eer": voice_eer,
        "music_eer": music_eer,
        "voice_presence_auc": voice_presence_auc,
        "music_presence_auc": music_presence_auc,
        "ads": ads,
        "cps": cps,
        "score": 0.9 * ads + 0.1 * cps,
    }
