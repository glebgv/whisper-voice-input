"""Whisper transcription server.

OpenAI-compatible /v1/audio/transcriptions endpoint, backed by faster-whisper.
Runs on CUDA when available, falls back to CPU otherwise.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import WhisperModel

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("whisper-server")

MODEL_NAME = os.environ.get("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
HOST = os.environ.get("WHISPER_HOST", "0.0.0.0")
PORT = int(os.environ.get("WHISPER_PORT", "8000"))

log.info("Loading model=%s device=%s compute_type=%s", MODEL_NAME, DEVICE, COMPUTE_TYPE)
_model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
log.info("Model loaded")

app = FastAPI(title="Whisper Server", version="1.0.0")


def _decode_response(segments, full_text: str, fmt: str) -> JSONResponse | PlainTextResponse:
    if fmt == "text":
        return PlainTextResponse(full_text)
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/vtt")
    return JSONResponse({"text": full_text})


def _to_srt(segments) -> str:
    out = []
    for i, s in enumerate(segments, 1):
        out.append(f"{i}\n{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}\n{s['text'].strip()}\n")
    return "\n".join(out)


def _to_vtt(segments) -> str:
    out = ["WEBVTT", ""]
    for s in segments:
        out.append(f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}\n{s['text'].strip()}\n")
    return "\n".join(out)


def _fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "compute": COMPUTE_TYPE}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Literal["json", "text", "srt", "vtt"] = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    t0 = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "empty file")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        segments_iter, info = _model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        segs = []
        text_parts = []
        for s in segments_iter:
            segs.append({"start": s.start, "end": s.end, "text": s.text})
            text_parts.append(s.text)
        full_text = "".join(text_parts).strip()
    except Exception as exc:
        log.exception("transcribe failed")
        raise HTTPException(500, f"transcribe failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    dt = time.perf_counter() - t0
    log.info(
        "transcribed file=%s lang=%s dur=%.1fs took=%.2fs segs=%d",
        file.filename, info.language, info.duration, dt, len(segs),
    )
    return _decode_response(segs, full_text, response_format)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
