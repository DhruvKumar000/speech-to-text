# Speech-to-Text with Accuracy & Efficiency

A speech-to-text project built on OpenAI Whisper. A person speaks (100+ words), and the system
writes down the words and reports how **accurate** and how **efficient** the recognition was.

## Live demo
- GitHub Pages: `https://dhruvkumar000.github.io/speech-to-text/`
- Hugging Face: `https://huggingface.co/spaces/Dhruv00101/speech-to-text-app`

## Files
| File | What it does |
|------|--------------|
| `index.html` | Web app. Whisper (base) runs in the browser with Transformers.js. Shows transcript, accuracy and efficiency. No server needed. |
| `stt.py` | Command-line transcription with faster-whisper (file or microphone). Saves `output.txt` and `output.srt`. |
| `app.py` | Local Gradio web app using faster-whisper. |
| `evaluate.py` | Calculates Word Error Rate (WER) and accuracy against a correct reference text. |
| `requirements.txt` | Python libraries. |

## How accuracy and efficiency are measured
- **Accuracy (web app):** estimated from the model's confidence in every word it recognised.
  Unsure words are highlighted yellow, probably wrong words red.
- **Accuracy (evaluate.py):** Word Error Rate against the exact text that was spoken.
  Accuracy = 1 − WER.
- **Efficiency:** processing time compared with audio length.
  Real-Time Factor (RTF) = processing time ÷ audio length. Below 1 means faster than real time.

## Run the Python version (Mac)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python stt.py --mic 60                 # record 60 s from microphone
python stt.py --audio speech.mp3       # transcribe a file
python stt.py --audio speech.mp3 --language hi
python app.py                          # local web app
python evaluate.py --reference reference.txt --hypothesis output.txt
```
Requires ffmpeg (`brew install ffmpeg`). Python 3.12 recommended.

## Tech stack
OpenAI Whisper, faster-whisper (CTranslate2, int8 quantization), Transformers.js (WebGPU / WASM), Gradio, jiwer.

## License
MIT
