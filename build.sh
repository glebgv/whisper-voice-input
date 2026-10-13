#!/usr/bin/env bash
# build.sh — build VoiceInput.exe from WSL (invokes Windows Python/PyInstaller)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
WIN_PY='C:\Program Files\Python312\python.exe'

echo "[build] deps..."
powershell.exe -NoProfile -Command "& '$WIN_PY' -m pip install --quiet --user numpy sounddevice requests pyperclip pyautogui pynput PySide6 pyinstaller" 2>&1 | grep -v "WARNING: Ignoring invalid distribution" || true

echo "[build] cleaning old artifacts..."
rm -rf "$DIR/build" "$DIR/dist" "$DIR/VoiceInput.spec"

echo "[build] running PyInstaller..."
powershell.exe -NoProfile -Command "Set-Location 'C:\Users\Happy\voice_input'; & '$WIN_PY' -m PyInstaller --noconsole --onefile --name VoiceInput --collect-all sounddevice voice_input.py" 2>&1 | grep -viE "warning|info:.*(upx|checking|appending)" | tail -20

if ls "$DIR"/dist/VoiceInput.exe >/dev/null 2>&1; then
    echo "[build] OK: $DIR/dist/VoiceInput.exe ($(du -h "$DIR/dist/VoiceInput.exe" | cut -f1))"
else
    echo "[build] FAILED"
    exit 1
fi
