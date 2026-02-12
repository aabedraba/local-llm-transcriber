# Local LLM Transcriber

Local-only transcription with summarization for your private voice notes. No data leaves your machine.

Use it as a [Raycast extension](#raycast-integration).

![Demo](assets/screenshot.png)

## Features

- Audio/video transcription using Whisper
- AI summaries with Ollama (optional)
- Batch processing multiple files

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.ai) (optional, for summaries)

## Quick Start

```bash
# Install dependencies
uv sync

# Run the app
uv run local-transcriber
```

The app will open in your browser automatically.

### Development Mode

Run with hot reload to automatically pick up code changes:

```bash
uv run gradio src/local_llm_transcriber/app.py
```

## Ollama Setup (Optional)

For AI-powered transcript summaries:

```bash
# Install Ollama from https://ollama.ai

# Pull a recommended model
ollama pull qwen2.5:7b
```

Recommended models for summarization:
- **qwen3:8b** - Best balance of quality and speed

## Raycast Integration

Launch the transcriber from anywhere with [Raycast](https://raycast.com):

1. Open Raycast Settings (`Cmd+,`)
2. Go to **Extensions** → **Script Commands**
3. Click **Add Directories** and select the `scripts/` folder from this project
4. Type **"Local Transcriber"** in Raycast to launch

The script will start the app if it's not running, or open the existing instance in your browser. You can assign a global hotkey to it in Raycast's extension settings.

## Usage

1. **Transcribe**: Record or upload audio, select model and language
2. **Summarize**: Use Ollama to generate AI summaries
3. **Batch**: Process multiple files at once
4. **Export**: Download transcripts in your preferred format
5. **Settings**: Manage Whisper models and Ollama configuration
