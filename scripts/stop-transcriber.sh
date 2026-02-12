#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Stop Transcriber
# @raycast.mode compact

# Optional parameters:
# @raycast.icon images/icon.png
# @raycast.packageName Local LLM Transcriber

# Documentation:
# @raycast.description Stop the local transcriber Gradio app
# @raycast.author aabedraba

PORT=7860

PID=$(lsof -ti :"$PORT" -sTCP:LISTEN)

if [ -z "$PID" ]; then
  echo "Transcriber is not running"
  exit 0
fi

kill "$PID"
echo "Transcriber stopped"
