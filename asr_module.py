# src/asr_module.py

from faster_whisper import WhisperModel

class ASRModule:
    def __init__(self, model_size: str = "small", device: str = "cpu"):
        # model_size: "tiny", "base", "small", "medium", "large"
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.device = device

    def transcribe_file(self, audio_path: str):
        """
        Returns list of segments: each is dict {start, end, text}
        """
        segments_list = []

        # ✔ Correct: unpack the return values
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5,
            word_timestamps=False
        )

        # ✔ Now iterate over the REAL segment generator
        for seg in segments:
            segments_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip()
            })

        return segments_list
