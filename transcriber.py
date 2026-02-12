"""faster-whisper transcription integration."""

from faster_whisper import WhisperModel

# Cache loaded models to avoid reloading on every call
_model_cache: dict[str, WhisperModel] = {}


def _get_model(model_size: str) -> WhisperModel:
    """Return a cached WhisperModel instance, loading it if necessary."""
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(
            model_size, device="auto", compute_type="auto"
        )
    return _model_cache[model_size]


def transcribe(
    file_path: str,
    model_size: str = "large-v3-turbo",
    language: str | None = None,
) -> list[dict]:
    """Transcribe an audio/video file using faster-whisper.

    Args:
        file_path: Path to the audio or video file.
        model_size: Whisper model size to use.
        language: Language code (e.g. "en"). None for auto-detection.

    Returns:
        List of segment dicts with keys "start", "end", "text".
    """
    model = _get_model(model_size)
    segments, info = model.transcribe(
        file_path,
        language=language,
        vad_filter=True,
        word_timestamps=True,
        initial_prompt="Transcribe with proper grammar, punctuation, and sentence structure.",
    )
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]
