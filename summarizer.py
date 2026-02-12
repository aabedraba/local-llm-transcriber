"""Ollama HTTP client for LLM-powered transcript summarization."""

import httpx

OLLAMA_URL = "http://localhost:11434"


def is_ollama_running(base_url: str = OLLAMA_URL) -> bool:
    """Check if Ollama is running and reachable.

    Args:
        base_url: Ollama server base URL.

    Returns:
        True if Ollama responds with HTTP 200, False otherwise.
    """
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def list_models(base_url: str = OLLAMA_URL) -> list[str]:
    """List available models from the Ollama server.

    Args:
        base_url: Ollama server base URL.

    Returns:
        List of model name strings. Empty list if Ollama is unreachable.
    """
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return [m["name"] for m in r.json().get("models", [])]
    except (httpx.ConnectError, httpx.TimeoutException):
        return []


def summarize(
    transcript: str,
    model: str = "llama3.2",
    prompt: str = "Summarize this transcript concisely:",
    base_url: str = OLLAMA_URL,
) -> str:
    """Summarize a transcript using an Ollama model.

    Args:
        transcript: The transcript text to summarize.
        model: Ollama model name to use.
        prompt: System prompt prepended to the transcript.
        base_url: Ollama server base URL.

    Returns:
        Summary text from the model.

    Raises:
        httpx.ConnectError: If Ollama is not reachable.
        httpx.TimeoutException: If the request times out.
    """
    r = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": f"{prompt}\n\n{transcript}",
            "stream": False,
        },
        timeout=120,
    )
    return r.json()["response"]
