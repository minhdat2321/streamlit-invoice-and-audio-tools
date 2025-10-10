# audio → text utility used by Streamlit UI
from pathlib import Path
from typing import Optional, BinaryIO

# Using the OpenAI python SDK v1.x (2024+)
# We accept an existing OpenAI client injected from the app

def transcribe_file_like(client, file_like: BinaryIO, filename: str, model: str = "gpt-4o-transcribe", language: str = "vi", response_format: Optional[str] = None) -> str:
    """Transcribe an audio file-like object. Returns raw text / SRT / VTT string.
    response_format: None → txt (default), or "srt" / "vtt".
    """
    kwargs = {
        "model": model,
        "file": (filename, file_like),  # file-like tuple works with SDK
        "language": language,
    }
    if response_format in {"srt", "vtt"}:
        kwargs["response_format"] = response_format

    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"ASR API failed: {e}")

    text = getattr(result, "text", None)
    return text if text else str(result)