"""Export transcription segments to SRT, VTT, TXT, and JSON formats."""

import json


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def to_srt(segments: list[dict]) -> str:
    """Convert segments to SRT subtitle format.

    Args:
        segments: List of dicts with "start", "end", "text" keys.

    Returns:
        SRT formatted string.
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(seg["start"])
        end = _format_timestamp_srt(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[dict]) -> str:
    """Convert segments to WebVTT subtitle format.

    Args:
        segments: List of dicts with "start", "end", "text" keys.

    Returns:
        WebVTT formatted string.
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(seg["start"])
        end = _format_timestamp_vtt(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def to_txt(segments: list[dict]) -> str:
    """Convert segments to plain text, one segment per line.

    Args:
        segments: List of dicts with "start", "end", "text" keys.

    Returns:
        Plain text string.
    """
    return "\n".join(seg["text"] for seg in segments)


def to_json(segments: list[dict]) -> str:
    """Convert segments to a JSON array with timestamps.

    Args:
        segments: List of dicts with "start", "end", "text" keys.

    Returns:
        JSON string.
    """
    return json.dumps(segments, indent=2, ensure_ascii=False)
