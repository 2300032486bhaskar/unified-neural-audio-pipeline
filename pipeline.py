# src/pipeline.py

import os
import shutil
import json
from asr_module import ASRModule
from separation import separate_sources

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simple_target_isolation(mixture_path, target_path, out_target_path):
    """
    Separate mixture and choose the 'vocals' stem (if present) as target.
    If separation fails, fallback to provided target_path (copy).
    """
    sep_dir = os.path.join(OUTPUT_DIR, "separated")
    os.makedirs(sep_dir, exist_ok=True)

    separated_files = separate_sources(mixture_path, sep_dir)

    # prefer a stem named 'vocals' or 'sources' that contains 'voc'
    chosen = None
    for f in separated_files:
        name = os.path.basename(f).lower()
        if "voc" in name or "voice" in name or "singer" in name:
            chosen = f
            break

    if chosen is None and len(separated_files) > 0:
        # fallback to first stem (often vocals are last; adapt as needed)
        chosen = separated_files[0]

    if chosen:
        shutil.copyfile(chosen, out_target_path)
    else:
        # fallback to provided target sample
        shutil.copyfile(target_path, out_target_path)

    return out_target_path


def run_offline_job(mixture_path, target_path, device="cpu"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_target = os.path.join(OUTPUT_DIR, "target_speaker.wav")

    simple_target_isolation(mixture_path, target_path, out_target)

    asr = ASRModule(device=device)
    segments = asr.transcribe_file(out_target)

    diar_path = os.path.join(OUTPUT_DIR, "diarization.json")
    with open(diar_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    return out_target, diar_path
