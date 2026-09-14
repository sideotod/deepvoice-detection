"""HTDemucs loading and voice/accompaniment separation."""

from pathlib import Path

import numpy as np

from deepvoice.audio import AUDIO_SAMPLE_RATE


def load_htdemucs_model(htdemucs_dir: Path):
    """Load the local HTDemucs checkpoint with PyTorch 2.6+ compatibility."""
    import torch
    from demucs.pretrained import get_model

    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=htdemucs_dir)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path: Path, model, device) -> tuple[np.ndarray, np.ndarray]:
    """Return 16 kHz mono vocals and all non-vocal stems summed together."""
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.separate import load_track

    waveform = load_track(audio_path, model.audio_channels, model.samplerate).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)
    music_sources = [
        sources[index]
        for index, source_name in enumerate(model.sources)
        if source_name != "vocals"
    ]
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )
