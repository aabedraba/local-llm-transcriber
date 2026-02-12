"""Local LLM Transcriber - Local-only audio/video transcription app."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel

import exporter
import models
import summarizer
import transcriber


# ── Helper functions ──


def run_transcription(file, model_size, language):
    """Transcribe a recorded audio file and return segments + display text."""
    if file is None:
        gr.Warning("Please record audio first.")
        return [], "", "No audio recorded"

    lang = None if language == "Auto-detect" else language

    try:
        segments = transcriber.transcribe(file, model_size=model_size, language=lang)
        text = exporter.to_txt(segments)
        status = f"Transcribed {len(segments)} segments"
        if language == "Auto-detect":
            status += " (language auto-detected)"
        return segments, text, status
    except Exception as e:
        gr.Warning(f"Transcription failed: {e}")
        return [], "", f"Error: {e}"


def export_transcript(segments, fmt):
    """Export segments to the chosen format and return a temp file path."""
    if not segments:
        gr.Warning("No transcript to export. Run transcription first.")
        return gr.update(visible=False)

    format_map = {
        "SRT": (exporter.to_srt, ".srt"),
        "VTT": (exporter.to_vtt, ".vtt"),
        "TXT": (exporter.to_txt, ".txt"),
        "JSON": (exporter.to_json, ".json"),
    }

    fn, ext = format_map[fmt]
    content = fn(segments)

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=ext, mode="w", encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


def run_summarize(segments, ollama_model, custom_prompt, ollama_url):
    """Summarize the transcript using Ollama."""
    if not segments:
        gr.Warning("No transcript to summarize. Run transcription first.")
        return ""

    text = exporter.to_txt(segments)
    prompt = (
        custom_prompt.strip()
        if custom_prompt.strip()
        else "Summarize this transcript concisely:"
    )
    base_url = ollama_url.strip() if ollama_url.strip() else summarizer.OLLAMA_URL

    try:
        return summarizer.summarize(text, model=ollama_model, prompt=prompt, base_url=base_url)
    except Exception as e:
        gr.Warning(f"Summarization failed: {e}")
        return f"Error: {e}"


def batch_process(files, model_size, language, progress=gr.Progress()):
    """Process multiple files and return results."""
    if not files:
        gr.Warning("Please upload files first.")
        return [], "No files uploaded"

    lang = None if language == "Auto-detect" else language
    all_results = []
    status_lines = []

    for i, file_path in enumerate(files):
        name = Path(file_path).name
        progress(i / len(files), desc=f"Processing {name}...")
        try:
            segments = transcriber.transcribe(
                file_path, model_size=model_size, language=lang
            )
            all_results.append({"file": name, "segments": segments})
            status_lines.append(f"[OK] {name}: {len(segments)} segments")
        except Exception as e:
            all_results.append({"file": name, "segments": []})
            status_lines.append(f"[FAIL] {name}: {e}")

    progress(1.0, desc="Done")
    return all_results, "\n".join(status_lines)


def batch_export_zip(batch_results, fmt):
    """Export all batch results as a ZIP file."""
    if not batch_results:
        gr.Warning("No batch results to export. Run batch processing first.")
        return gr.update(visible=False)

    format_map = {
        "SRT": (exporter.to_srt, ".srt"),
        "VTT": (exporter.to_vtt, ".vtt"),
        "TXT": (exporter.to_txt, ".txt"),
        "JSON": (exporter.to_json, ".json"),
    }

    fn, ext = format_map[fmt]

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_zip.close()

    with zipfile.ZipFile(tmp_zip.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in batch_results:
            if result["segments"]:
                content = fn(result["segments"])
                name = Path(result["file"]).stem + ext
                zf.writestr(name, content)

    return gr.update(value=tmp_zip.name, visible=True)


def refresh_downloaded_models():
    """Return formatted string of downloaded models."""
    downloaded = models.get_downloaded_models()
    if not downloaded:
        return "No models downloaded yet."
    return "\n".join(f"  {m} - {models.AVAILABLE_MODELS[m]}" for m in downloaded)


def get_available_models_display():
    """Return formatted string of all available models."""
    return "\n".join(f"  {k}: {v}" for k, v in models.AVAILABLE_MODELS.items())


def download_model_action(model_size):
    """Download a whisper model by loading it."""
    if not model_size:
        return "Please select a model first.", refresh_downloaded_models()
    try:
        WhisperModel(model_size, device="auto", compute_type="auto")
        return f"Model '{model_size}' downloaded successfully.", refresh_downloaded_models()
    except Exception as e:
        return f"Download failed: {e}", refresh_downloaded_models()


def delete_model_action(model_size):
    """Delete a cached whisper model."""
    if not model_size:
        return "Please select a model first.", refresh_downloaded_models()
    cache_dir = Path(models.get_model_path())
    repo_dir = cache_dir / f"models--Systran--faster-whisper-{model_size}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
        return f"Model '{model_size}' deleted.", refresh_downloaded_models()
    return f"Model '{model_size}' not found in cache.", refresh_downloaded_models()


def check_ollama_status(base_url):
    """Check Ollama status and return available models."""
    url = base_url.strip() if base_url.strip() else summarizer.OLLAMA_URL
    running = summarizer.is_ollama_running(base_url=url)
    if running:
        available = summarizer.list_models(base_url=url)
        status = f"Ollama is running ({len(available)} models available)"
        return (
            status,
            gr.update(choices=available, value=available[0] if available else None),
            gr.update(choices=available, value=available[0] if available else None),
        )
    return (
        "Ollama is not running",
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
    )


# ── Startup checks ──

_ollama_running = summarizer.is_ollama_running()
_ollama_models = summarizer.list_models() if _ollama_running else []

# ── Build UI ──

with gr.Blocks(title="Local LLM Transcriber") as demo:
    gr.Markdown("# Local LLM Transcriber")
    gr.Markdown("Local-only audio/video transcription. No data leaves your machine.")

    # Shared state
    segments_state = gr.State([])
    batch_results_state = gr.State([])

    with gr.Tabs():
        # ── Tab 1: Transcribe ──────────────────────────────────────────
        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Record Audio",
                        sources=["microphone"],
                        type="filepath",
                        format="wav",
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(models.AVAILABLE_MODELS.keys()),
                        value="large-v3-turbo",
                        label="Model",
                    )
                    language_dropdown = gr.Dropdown(
                        choices=[
                            "Auto-detect", "en", "ar", "fr", "de", "es", "zh", "ja",
                        ],
                        value="Auto-detect",
                        label="Language",
                    )

                with gr.Column(scale=2):
                    status_display = gr.Textbox(
                        label="Status", interactive=False, lines=1
                    )
                    transcript_output = gr.Textbox(
                        label="Transcript", lines=20, interactive=False
                    )

            # ── Summarize section ──
            gr.Markdown("### AI Summary")
            with gr.Row():
                with gr.Column():
                    ollama_status_msg = gr.Markdown(
                        "Ollama is running" if _ollama_running
                        else (
                            "> **Ollama not detected.** "
                            "Install [Ollama](https://ollama.ai) and start it "
                            "to enable AI-powered transcript summaries."
                        )
                    )
                    ollama_model_dropdown = gr.Dropdown(
                        choices=_ollama_models,
                        value=_ollama_models[0] if _ollama_models else None,
                        label="Ollama Model",
                        visible=_ollama_running,
                    )
                    custom_prompt = gr.Textbox(
                        label="Custom Prompt",
                        value="Summarize this transcript concisely:",
                        lines=2,
                        visible=_ollama_running,
                    )
                    summarize_btn = gr.Button(
                        "Summarize", visible=_ollama_running
                    )
                with gr.Column():
                    summary_output = gr.Textbox(
                        label="Summary",
                        lines=10,
                        interactive=False,
                        visible=_ollama_running,
                    )

            # ── Transcribe tab event handlers ──
            audio_input.stop_recording(
                fn=run_transcription,
                inputs=[audio_input, model_dropdown, language_dropdown],
                outputs=[segments_state, transcript_output, status_display],
            )

        # ── Tab 2: Batch ──────────────────────────────────────────────
        with gr.Tab("Batch"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="Upload Multiple Files",
                        file_types=["audio", "video"],
                        file_count="multiple",
                    )
                    batch_model = gr.Dropdown(
                        choices=list(models.AVAILABLE_MODELS.keys()),
                        value="large-v3-turbo",
                        label="Model",
                    )
                    batch_language = gr.Dropdown(
                        choices=[
                            "Auto-detect", "en", "ar", "fr", "de", "es", "zh", "ja",
                        ],
                        value="Auto-detect",
                        label="Language",
                    )
                    batch_btn = gr.Button("Process All", variant="primary")

                with gr.Column(scale=2):
                    batch_status = gr.Textbox(
                        label="Results", lines=15, interactive=False
                    )
                    with gr.Row():
                        batch_export_fmt = gr.Dropdown(
                            choices=["SRT", "VTT", "TXT", "JSON"],
                            value="SRT",
                            label="Export Format",
                        )
                        batch_export_btn = gr.Button("Export All as ZIP")
                    batch_export_file = gr.File(
                        label="Download ZIP", interactive=False, visible=False
                    )

            # ── Batch tab event handlers ──
            batch_btn.click(
                fn=batch_process,
                inputs=[batch_files, batch_model, batch_language],
                outputs=[batch_results_state, batch_status],
            )

            batch_export_btn.click(
                fn=batch_export_zip,
                inputs=[batch_results_state, batch_export_fmt],
                outputs=[batch_export_file],
            )

        # ── Tab 3: Settings ───────────────────────────────────────────
        with gr.Tab("Settings"):
            gr.Markdown("### Whisper Models")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Downloaded Models**")
                    downloaded_display = gr.Textbox(
                        value=refresh_downloaded_models(),
                        label="Downloaded",
                        interactive=False,
                        lines=6,
                    )
                    refresh_models_btn = gr.Button("Refresh")

                with gr.Column():
                    gr.Markdown("**Available Models**")
                    available_display = gr.Textbox(
                        value=get_available_models_display(),
                        label="All Models",
                        interactive=False,
                        lines=6,
                    )

            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=list(models.AVAILABLE_MODELS.keys()),
                    label="Select Model",
                )
                download_btn = gr.Button("Download Model")
                delete_btn = gr.Button("Delete Model", variant="stop")

            settings_status = gr.Textbox(
                label="Status", interactive=False, lines=1
            )

            cache_path_display = gr.Textbox(
                value=models.get_model_path(),
                label="Cache Directory",
                interactive=False,
            )

            gr.Markdown("### Ollama Settings")

            with gr.Row():
                ollama_url_input = gr.Textbox(
                    value=summarizer.OLLAMA_URL,
                    label="Ollama Base URL",
                )
                settings_ollama_model = gr.Dropdown(
                    choices=_ollama_models,
                    value=_ollama_models[0] if _ollama_models else None,
                    label="Default Ollama Model",
                )
                ollama_refresh_btn = gr.Button("Refresh Ollama")

            ollama_settings_status = gr.Textbox(
                label="Ollama Status", interactive=False, lines=1
            )

            # ── Settings tab event handlers ──
            refresh_models_btn.click(
                fn=refresh_downloaded_models, outputs=[downloaded_display]
            )

            download_btn.click(
                fn=download_model_action,
                inputs=[model_selector],
                outputs=[settings_status, downloaded_display],
            )

            delete_btn.click(
                fn=delete_model_action,
                inputs=[model_selector],
                outputs=[settings_status, downloaded_display],
            )

            ollama_refresh_btn.click(
                fn=check_ollama_status,
                inputs=[ollama_url_input],
                outputs=[
                    ollama_settings_status,
                    settings_ollama_model,
                    ollama_model_dropdown,
                ],
            )

    # ── Wire up the summarize button (needs ollama_url from Settings) ──
    summarize_btn.click(
        fn=run_summarize,
        inputs=[segments_state, ollama_model_dropdown, custom_prompt, ollama_url_input],
        outputs=[summary_output],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
