"""
Local web app (Gradio) for speech-to-text.
Run:  python app.py   -> opens a local link and a temporary public link.
"""
import time
import gradio as gr
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(audio, language):
    if audio is None:
        return "Please record or upload audio.", ""
    lang = None if language == "Auto" else language
    start = time.time()
    segments, info = model.transcribe(audio, language=lang, beam_size=5, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    elapsed = time.time() - start
    words = len(text.split())
    rtf = elapsed / info.duration if info.duration else 0
    status = "Requirement met" if words >= 100 else "Less than 100 words"
    stats = (f"Language: {info.language}\nWords: {words} ({status})\n"
             f"Audio length: {info.duration:.1f}s\nProcessing time: {elapsed:.1f}s\n"
             f"Real-Time Factor: {rtf:.3f}")
    return text, stats


demo = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or upload audio"),
            gr.Dropdown(["Auto", "en", "hi"], value="Auto", label="Language")],
    outputs=[gr.Textbox(label="Transcript", lines=10), gr.Textbox(label="Stats", lines=5)],
    title="Speech-to-Text (100+ words)",
    description="Record or upload speech. Built with faster-whisper.",
)

demo.launch(share=True)
