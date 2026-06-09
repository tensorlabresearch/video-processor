import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("video-processor")

MODEL_SIZE = os.environ.get("MODEL_SIZE", "large-v3")
PORT = int(os.environ.get("PORT", "8082"))

app = FastAPI(title="video-processor")

# Global model state — set to True after startup completes loading
_model: WhisperModel | None = None
_model_loaded: bool = False


@app.on_event("startup")
async def load_model() -> None:
    global _model, _model_loaded
    logger.info("Loading Whisper model %s on CUDA (float16) ...", MODEL_SIZE)
    # Run blocking model init in a thread so the event loop stays responsive
    loop = asyncio.get_event_loop()
    _model = await loop.run_in_executor(
        None,
        lambda: WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16"),
    )
    _model_loaded = True
    logger.info("Whisper model loaded successfully")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "model_loaded": _model_loaded})


@app.post("/process")
async def process_video(body: dict) -> JSONResponse:
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse(status_code=400, content={"error": "url is required"})

    if not _model_loaded or _model is None:
        return JSONResponse(status_code=503, content={"error": "model not yet loaded"})

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        try:
            title = await _download_audio(url, audio_path)
        except Exception as exc:
            logger.error("yt-dlp download failed for %s: %s", url, exc)
            return JSONResponse(status_code=500, content={"error": str(exc)})

        try:
            transcript, language, duration = await _transcribe(audio_path)
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return JSONResponse({
        "transcript": transcript,
        "title": title,
        "duration_seconds": duration,
        "language": language,
    })


async def _download_audio(url: str, output_path: Path) -> str:
    """Download audio-only stream and return video title."""
    logger.info("Downloading audio from %s", url)

    # First pass: get title
    info_cmd = [
        "yt-dlp",
        "--get-title",
        "--no-playlist",
        url,
    ]
    loop = asyncio.get_event_loop()
    title = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: _run_cmd(info_cmd)),
        timeout=60,
    )
    title = title.strip() or "Untitled"

    # Second pass: download audio
    dl_cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-playlist",
        "-o", str(output_path),
        url,
    ]
    await asyncio.wait_for(
        loop.run_in_executor(None, lambda: _run_cmd(dl_cmd)),
        timeout=600,  # 10 minutes
    )
    logger.info("Download complete: %s", output_path)
    return title


def _run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


async def _transcribe(audio_path: Path) -> tuple[str, str, float]:
    """Transcribe audio file and return (transcript, language, duration_seconds)."""
    logger.info("Transcribing %s ...", audio_path)
    loop = asyncio.get_event_loop()

    def _do_transcribe():
        segments, info = _model.transcribe(str(audio_path), beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        return text, info.language, info.duration

    transcript, language, duration = await loop.run_in_executor(None, _do_transcribe)
    logger.info(
        "Transcription complete: lang=%s duration=%.1fs chars=%d",
        language, duration, len(transcript),
    )
    return transcript, language, duration


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
