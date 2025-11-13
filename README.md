🎧 Unified Neural Audio Pipeline
Speaker Extraction • Speech Recognition • Audio Source Separation

This project implements an end-to-end neural audio processing pipeline capable of:

✔ Separating background music & vocals from any audio (Demucs)
✔ Extracting a target speaker’s voice from a mixture using speaker embedding
✔ Generating transcription using Whisper ASR (faster-whisper)
✔ Providing a simple FastAPI backend to upload & process audio files

This pipeline is lightweight, easy to customize, and built from scratch for demonstrations, academic reviews, and technical interviews.

🚀 Features
🔹 1. Audio Source Separation (Demucs)

Separates input audio into:

Vocals

Drums

Bass

Other

🔹 2. Speaker Embedding + Target Speaker Extraction

Uses SpeechBrain ECAPA-TDNN to:

Generate speaker embeddings

Match similarity

Extract only the target speaker’s voice from the mixture

🔹 3. Speech Recognition (Whisper ASR)

Uses Faster-Whisper small model for transcription.

🔹 4. FastAPI Backend

Simple /submit-offline/ endpoint to upload:

mixture_audio.wav

target_speaker_sample.wav

Returns:

Extracted target speaker audio

JSON diarization/transcription metadata

🗂️ Project Structure
unified-neural-pipeline/
│
├── outputs/                # pipeline results (auto-created)
│
├── src/
│   ├── app.py              # FastAPI server
│   ├── pipeline.py         # full processing pipeline
│   ├── separation.py       # audio separation (Demucs)
│   ├── asr_module.py       # Whisper ASR wrapper
│   ├── speaker_embedding.py# Speaker embedding (ECAPA)
│   ├── utils.py            # helper utilities
│   ├── __init__.py
│
├── requirements.txt        # dependencies
├── README.md               # documentation
├── run.py                  # one-line runner

🔧 Installation
1️⃣ Create a virtual environment
python -m venv .venv

2️⃣ Activate it

Windows:

.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Running the Server

Start the FastAPI backend:

python src/app.py


You will see:

Uvicorn running on http://0.0.0.0:8000


Open the browser:
👉 http://localhost:8000/docs

Upload the files and run the pipeline.

📤 API Usage
POST /submit-offline/

Upload two audio files:

Parameter	Type	Description
mixture	file (.wav)	Audio containing mixture (music + speech)
target	file (.wav)	3–5 second sample of target speaker
✔ Sample Response
{
  "output_target": "outputs/target_speaker.wav",
  "diarization_output": "outputs/diarization.json"
}

🧠 Models Used
Task	Model
Source Separation	Demucs v4
Speaker Embedding	ECAPA-TDNN (SpeechBrain)
Speech Recognition	Whisper-Small (Faster-Whisper)
📌 Notes

Works fully offline after the first model download.

Designed as an interview + academic review-friendly project.

Clean modular codebase for easy expansion.

🏁 Conclusion

This project demonstrates a full neural audio pipeline integrating:

Source separation

Speaker extraction

Speech recognition

Backend service