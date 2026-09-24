"""
Speech-to-Text (100+ words) using faster-whisper
Focus: accuracy + efficiency

Usage:
  python stt.py --audio speech.wav
  python stt.py --mic 60                 # record 60 seconds from microphone
  python stt.py --audio speech.mp3 --model medium --language hi
"""

import argparse
import time
import os
import subprocess
import json

from faster_whisper import WhisperModel


def get_device():
    """Use GPU if available, else CPU."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"  # int8 = fast + small on CPU


def record_from_mic(seconds, path="mic_recording.wav", sr=16000):
    import sounddevice as sd
    import soundfile as sf
    print(f"🎙️  Recording for {seconds} seconds... speak now")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, sr)
    print(f"✅ Saved recording to {path}")
    return path


def get_audio_duration(path):
    """Audio length in seconds (needs ffmpeg/ffprobe)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, check=True)
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def to_srt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


def transcribe(audio_path, model_size="small", language=None,
               min_words=100, prompt=None, out_prefix="output"):
    device, compute_type = get_device()
    print(f"⚙️  Loading model '{model_size}' on {device} ({compute_type})...")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"   Model loaded in {time.time() - t0:.1f}s")

    print("📝 Transcribing...")
    start = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,            # None = auto-detect
        beam_size=5,                  # better accuracy than greedy
        vad_filter=True,              # removes silence, avoids hallucinations
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=True,  # keeps long speech consistent
        initial_prompt=prompt,        # domain words/names help accuracy
    )

    segments = list(segments)  # runs the actual transcription
    elapsed = time.time() - start

    text = " ".join(s.text.strip() for s in segments).strip()
    word_count = len(text.split())

    # Save plain text
    with open(f"{out_prefix}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # Save subtitles with timestamps
    with open(f"{out_prefix}.srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            f.write(f"{i}\n{to_srt_time(s.start)} --> {to_srt_time(s.end)}\n"
                    f"{s.text.strip()}\n\n")

    # Efficiency report
    duration = info.duration or get_audio_duration(audio_path)
    rtf = elapsed / duration if duration else None

    print("\n" + "=" * 60)
    print("TRANSCRIPT:\n")
    print(text)
    print("=" * 60)
    print(f"Detected language : {info.language} ({info.language_probability:.0%})")
    print(f"Word count        : {word_count}")
    if duration:
        print(f"Audio length      : {duration:.1f}s")
    print(f"Processing time   : {elapsed:.1f}s")
    if rtf:
        print(f"Real-Time Factor  : {rtf:.3f}  (below 1 = faster than real time)")
    print(f"Saved             : {out_prefix}.txt, {out_prefix}.srt")

    if word_count < min_words:
        print(f"\n⚠️  Only {word_count} words recognised (minimum is {min_words}). "
              f"Use longer audio or check audio quality.")
    else:
        print(f"\n✅ Requirement met: {word_count} ≥ {min_words} words")

    return text, word_count, rtf


def main():
    p = argparse.ArgumentParser(description="Speech-to-Text (100+ words)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="Path to audio file (wav, mp3, m4a, ...)")
    src.add_argument("--mic", type=int, help="Record from microphone for N seconds")
    p.add_argument("--model", default="small",
                   help="tiny | base | small | medium | large-v3 (default: small)")
    p.add_argument("--language", default=None, help="e.g. en, hi (default: auto)")
    p.add_argument("--min-words", type=int, default=100)
    p.add_argument("--prompt", default=None,
                   help="Domain words to improve accuracy, e.g. 'Kurukshetra, NIT, IoT'")
    p.add_argument("--out", default="output", help="Output file prefix")
    args = p.parse_args()

    audio = args.audio
    if args.mic:
        # ~150 words per minute of speech -> 60s is enough for 100+ words
        audio = record_from_mic(args.mic)

    if not os.path.exists(audio):
        raise FileNotFoundError(audio)

    transcribe(audio, args.model, args.language, args.min_words,
               args.prompt, args.out)


if __name__ == "__main__":
    main()
