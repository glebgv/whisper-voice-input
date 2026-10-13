#!/usr/bin/env bash
# run.sh — run VoiceInput from WSL (launches Windows exe, or .py if exe missing)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$DIR/dist/VoiceInput.exe" ]; then
    echo "[run] starting dist/VoiceInput.exe"
    powershell.exe -NoProfile -Command "Start-Process '$(wslpath -w "$DIR/dist/VoiceInput.exe")'"
else
    echo "[run] exe not found — running from source"
    powershell.exe -NoProfile -Command "Start-Process 'C:\Program Files\Python312\python.exe' -ArgumentList '$(wslpath -w "$DIR/voice_input.py")'"
fi
