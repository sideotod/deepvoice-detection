"""File-level fusion strategies for component fake-risk scores."""

FUSION_NAMES = ("baseline", "component_max", "soft_or")


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def combine_file_fake_score(
    fusion_name: str,
    voice_fake: float,
    music_fake: float,
    voice_present: float,
    music_present: float,
) -> float:
    """Return a file-level fake probability using a named fusion method."""
    voice_fake = _validate_probability("voice_fake", voice_fake)
    music_fake = _validate_probability("music_fake", music_fake)
    voice_present = _validate_probability("voice_present", voice_present)
    music_present = _validate_probability("music_present", music_present)

    voice_risk = voice_present * voice_fake
    music_risk = music_present * music_fake

    if fusion_name == "baseline":
        return max(voice_risk, music_risk)
    if fusion_name == "component_max":
        return max(voice_fake, music_fake)
    if fusion_name == "soft_or":
        return 1.0 - (1.0 - voice_risk) * (1.0 - music_risk)

    raise ValueError(
        f"Unknown fusion method: {fusion_name}. "
        f"Choose one of: {', '.join(FUSION_NAMES)}"
    )
