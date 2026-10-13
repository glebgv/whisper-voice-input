#!/usr/bin/env bash
# Server setup & run (WSL/Linux). Creates venv and installs deps on first run.
set -euo pipefail
cd "$(dirname "$0")"

VENV="$HOME/.venvs/whisper-server"
if [ ! -d "$VENV" ]; then
    echo "[setup] creating venv..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    echo "[setup] installing deps..."
    "$VENV/bin/pip" install faster-whisper "av<16" fastapi "uvicorn[standard]" \
        python-multipart nvidia-cublas-cu12
fi

# activate + CUDA libs from pip nvidia packages
source "$VENV/bin/activate"
NVIDIA_LIBS=""
for d in "$VENV"/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="$NVIDIA_LIBS:$d"
done
export LD_LIBRARY_PATH="${NVIDIA_LIBS#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[run] starting whisper server on :8000"
exec python server.py
