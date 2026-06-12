import asyncio
import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import requests
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
# Max concurrent video processing jobs. GPU runs one Whisper job efficiently;
# queue up to this many waiters before returning 429.
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
# How long a caller waits to acquire the semaphore before getting 429.
QUEUE_TIMEOUT = float(os.environ.get("QUEUE_TIMEOUT_SECS", "30"))

app = FastAPI(title="video-processor")

# Global model state — set to True after startup completes loading
_model: WhisperModel | None = None
_model_loaded: bool = False
# Serialises GPU-intensive work so the model isn't hit concurrently.
_process_semaphore: asyncio.Semaphore | None = None


@app.on_event("startup")
async def load_model() -> None:
    global _model, _model_loaded, _process_semaphore
    _process_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    logger.info("Loading Whisper model %s on CUDA (float16) ...", MODEL_SIZE)
    # Run blocking model init in a thread so the event loop stays responsive
    loop = asyncio.get_event_loop()
    _model = await loop.run_in_executor(
        None,
        lambda: WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16"),
    )
    _model_loaded = True
    logger.info("Whisper model loaded successfully (max_concurrent=%d, queue_timeout=%.0fs)",
                MAX_CONCURRENT, QUEUE_TIMEOUT)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    sem = _process_semaphore
    queued = (MAX_CONCURRENT - sem._value) if sem else 0
    return JSONResponse({"status": "ok", "model_loaded": _model_loaded, "active_jobs": queued})


@app.post("/process")
async def process_video(body: dict) -> JSONResponse:
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse(status_code=400, content={"error": "url is required"})

    if not _model_loaded or _model is None:
        return JSONResponse(status_code=503, content={"error": "model not yet loaded"})

    try:
        acquired = await asyncio.wait_for(_process_semaphore.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Rate limit: semaphore not acquired within %.0fs for %s", QUEUE_TIMEOUT, url)
        return JSONResponse(
            status_code=429,
            content={"error": f"server busy — try again in a few minutes"},
        )

    try:
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
    finally:
        _process_semaphore.release()


@app.post("/fetch")
async def fetch_url(body: dict) -> JSONResponse:
    url = body.get("url", "").strip()
    if not url:
        return JSONResponse(status_code=400, content={"error": "url is required"})

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _fetch_and_extract(url)),
            timeout=60,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "fetch timed out after 60s"})
    except Exception as exc:
        logger.error("fetch failed for %s: %s", url, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return JSONResponse(result)


def _fetch_and_extract(url: str) -> dict:
    logger.info("Fetching %s", url)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    logger.info("Content-Type: %s (%d bytes)", content_type, len(resp.content))

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _extract_pdf(resp.content, url)
    else:
        return _extract_html(resp.text, url)


def _extract_pdf(data: bytes, url: str) -> dict:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)
    text = "\n\n".join(pages_text)
    title = (reader.metadata.title or "").strip() if reader.metadata else ""
    if not title:
        title = url.split("/")[-1]
    logger.info("PDF extracted: %d pages, %d chars", len(reader.pages), len(text))
    return {"text": text, "title": title, "content_type": "application/pdf", "char_count": len(text)}


def _extract_html(html: str, url: str) -> dict:
    from readability import Document
    doc = Document(html)
    title = doc.title() or ""
    # readability returns HTML summary; strip tags for plain text
    import re
    raw = doc.summary()
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    logger.info("HTML extracted: title=%r %d chars", title, len(text))
    return {"text": text, "title": title, "content_type": "text/html", "char_count": len(text)}


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
