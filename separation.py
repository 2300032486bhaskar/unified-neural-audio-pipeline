# src/separation.py

import os
import soundfile as sf
import torch
from demucs import pretrained
from demucs.apply import apply_model


def separate_sources(input_wav, output_dir):
    """
    Separate mixture into stems using Demucs (htdemucs model).
    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load Demucs model (htdemucs is reliable)
    model = pretrained.get_model('htdemucs')
    model.to("cpu")

    # Load audio with soundfile (gives (samples,) or (samples, channels))
    wav, sr = sf.read(input_wav)

    # ---- Ensure channel-first and 2 channels (Demucs htdemucs expects stereo) ----
    # Case: mono (samples,)
    if wav.ndim == 1:
        wav = wav[None, :]          # (1, samples)
        wav = wav.repeat(2, axis=0) # (2, samples)

    # Case: (samples, channels) -> transpose -> (channels, samples)
    elif wav.ndim == 2 and wav.shape[0] < wav.shape[1]:
        wav = wav.T

    # If somehow still mono channel dimension
    if wav.shape[0] == 1:
        wav = wav.repeat(2, axis=0)

    # Convert to tensor and add batch dim -> (1, channels, samples)
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

    # Run separation
    with torch.no_grad():
        print("[INFO] Running Demucs separation...")
        estimates = apply_model(model, wav_tensor, device="cpu")

    # Save stems
    stem_names = model.sources  # e.g. ['drums','bass','other','vocals']
    output_files = []

    for i, name in enumerate(stem_names):
        out_path = os.path.join(output_dir, f"{name}.wav")
        # estimates: (batch, n_sources, channels, samples) -> take batch 0, source i
        audio = estimates[0, i].cpu().numpy()
        # audio shape (channels, samples) -> transpose for soundfile (samples, channels)
        audio_out = audio.T
        sf.write(out_path, audio_out, sr)
        output_files.append(out_path)

    return output_files
