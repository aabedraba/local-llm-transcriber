#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Local Transcriber
# @raycast.mode compact

# Optional parameters:
# @raycast.icon images/icon.png
# @raycast.packageName Local LLM Transcriber

# Documentation:
# @raycast.description Launch the local transcriber Gradio app
# @raycast.author aabedraba

PROJECT_DIR="/Users/abdallah/github/aabedraba/local-llm-transcriber"
UV="/Users/abdallah/.local/bin/uv"
PORT=7860

# If already running, just open the browser
if lsof -i :"$PORT" -sTCP:LISTEN &>/dev/null; then
  open "http://localhost:$PORT"
  echo "Opened existing transcriber"
  exit 0
fi

# Launch the app in the background
cd "$PROJECT_DIR" || exit 1
nohup "$UV" run main.py &>/dev/null &

# Wait for the server to be ready (up to 30s)
for i in $(seq 1 60); do
  if lsof -i :"$PORT" -sTCP:LISTEN &>/dev/null; then
    echo "Transcriber started"
    exit 0
  fi
  sleep 0.5
done

echo "Timed out waiting for transcriber to start"
exit 1
