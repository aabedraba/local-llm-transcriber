"""Whisper model management utilities."""

from pathlib import Path

from faster_whisper.utils import download_model

AVAILABLE_MODELS: dict[str, str] = {
    "tiny": "~75 MB - Fastest, least accurate",
    "base": "~142 MB - Fast, basic accuracy",
    "small": "~466 MB - Good balance",
    "medium": "~1.5 GB - High accuracy",
    "large-v3-turbo": "~1.6 GB - Best speed/quality (recommended)",
    "large-v3": "~3.1 GB - Highest accuracy, slowest",
}


def get_model_path() -> str:
    """Return the local HuggingFace Hub cache directory path for models.

    Returns:
        Absolute path string to the cache directory.
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return str(cache_dir)


def get_downloaded_models() -> list[str]:
    """Check which whisper models are already cached locally.

    Scans the HuggingFace Hub cache directory for directories matching
    the faster-whisper model naming convention.

    Returns:
        List of model size names that are available locally.
    """
    cache_dir = Path(get_model_path())
    if not cache_dir.exists():
        return []

    downloaded = []
    for model_name in AVAILABLE_MODELS:
        # faster-whisper models are stored under the Systran org on HuggingFace
        # The cache directory format is: models--Systran--faster-whisper-{model_name}
        repo_dir = cache_dir / f"models--Systran--faster-whisper-{model_name}"
        if repo_dir.exists():
            downloaded.append(model_name)
    return downloaded
