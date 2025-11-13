# src/speaker_embedding.py

import os
from speechbrain.inference import SpeakerRecognition

# Disable symlink issues on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SB_FETCH_LOCAL_STRATEGY"] = "copy"

class SpeakerEmbedder:
    def __init__(self):
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": "cpu"},
            savedir_overwrite=True
        )

    def embed(self, wav_path: str):
        emb = self.model.encode_file(wav_path)
        return emb.squeeze().tolist()
