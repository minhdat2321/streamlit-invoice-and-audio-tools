from pathlib import Path
from typing import Optional, BinaryIO

def transcribe_meeting(
    client,
    file_like: BinaryIO,
    filename: str,
    model_asr: str = "gpt-4o-transcribe",
    model_format: str = "gpt-4o-mini",
    language: Optional[str] = None,   # Auto-detect if None
    meeting_context: Optional[str] = None,
    response_format: Optional[str] = None,
    model: Optional[str] = None,  # legacy alias for model_asr (accept callers using 'model')
) -> str:
    """
    Transcribe an audio file and format it into a clean, full conversation transcript.
    Auto-detects language and formats the result based on the meeting context.

    Parameters
    ----------
    client : openai.Client
        Initialized OpenAI client.
    file_like : BinaryIO
        Audio file-like object (e.g., uploaded via Streamlit).
    filename : str
        Name of the uploaded file.
    model_asr : str
        Model for automatic speech recognition (default: gpt-4o-transcribe).
    model_format : str
        GPT model for formatting the transcript (default: gpt-4o-mini).
    language : Optional[str]
        Specify language code if known (e.g., "en", "vi", "ko"). 
        If None → auto-detect.
    meeting_context : Optional[str]
        Optional text description of the meeting (e.g., "supplier negotiation", "R&D review").
    response_format : Optional[str]
        "txt" (default), or "srt"/"vtt" for subtitles.

    Returns
    -------
    str
        A readable, conversation-style transcript.
    """

    # --- Step 1: Transcribe raw audio ---
    # Support legacy callers that pass `model=` (some older code used that kwarg)
    if model:
        model_asr = model

    kwargs = {
        "model": model_asr,
        "file": (filename, file_like),
    }
    if language:
        kwargs["language"] = language
    if response_format in {"srt", "vtt"}:
        kwargs["response_format"] = response_format

    try:
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"ASR transcription failed: {e}")

    raw_text = getattr(result, "text", None) or str(result)

    # --- Step 2: Build prompt for formatting ---
    meeting_description = (
        f"This is a transcript from a meeting about {meeting_context}.\n"
        if meeting_context else
        "This is a general meeting transcript.\n"
    )

    prompt = (
        f"You are a professional transcription formatter specializing in meetings.\n"
        f"{meeting_description}"
        "Convert the following raw ASR output into a full, readable conversation transcript.\n"
        "→ Preserve *all spoken content* (no summarization).\n"
        "→ Have clear context and speaker turns.\n"
        "→ Restore punctuation and paragraph structure.\n"
        "→ Insert logical speaker tags (Speaker 1, Speaker 2, etc.) if not already present.\n"
        "→ Reflect natural dialogue flow (e.g., questions, responses, follow-ups).\n"
        "→ Maintain formal, clear, and professional tone suitable for documentation.\n"
        "→ If multiple languages appear, preserve them as spoken (do not translate).\n\n"
        "Raw ASR text:\n"
        f"{raw_text}\n\n"
        "Now produce the final formatted transcript below:\n"
    )

    # --- Step 3: Use GPT to structure transcript ---
    try:
        response = client.responses.create(
            model=model_format,
            input=prompt,
            temperature=0.3,
        )
        formatted_text = response.output_text.strip()
    except Exception as e:
        raise RuntimeError(f"Formatting step failed: {e}")

    return formatted_text
