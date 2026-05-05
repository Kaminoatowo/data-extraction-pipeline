import asyncio
import io
import sys
import uuid
import zipfile
import threading
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

# Stato run
RUNS: dict[str, subprocess.Popen] = {}
RUN_CONFIGS: dict[str, dict] = {}
RUN_LOGS: dict[str, list[str]] = {}

STEP_FLAGS = {
    "ocr":       ["--run_split", "--run_ocr"],
    "rag":       ["--run_rag"],
    "equations": ["--run_equations"],
    "qa":        ["--run_qa"],
    "synth":     ["--run_synth"],
    "gen_cpt":   ["--gen_cpt"],
    "gen_ft":    ["--gen_ft"],
}

CAT_PATHS = {
    "ocr":       BASE_DIR / "data" / "ocr_output",
    "rag":       BASE_DIR / "data" / "outputs" / "output_rag",
    "qa":        BASE_DIR / "data" / "outputs" / "qa_pairs",
    "equations": BASE_DIR / "data" / "outputs" / "formatted_equations",
    "synth":     BASE_DIR / "data" / "outputs",
    "merged":    BASE_DIR / "data" / "jsonl_files",
}


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


# ── PROCESS RUNNER (THREAD) ─────────────────────────────────────────

def run_process(cmd, run_id):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        text=True,
        bufsize=1
    )

    RUNS[run_id] = proc
    RUN_LOGS[run_id] = []

    for line in proc.stdout:
        line = line.rstrip()
        RUN_LOGS[run_id].append(line)

    proc.wait()
    RUN_LOGS[run_id].append(f"[PIPELINE DONE {proc.returncode}]")


# ── Upload ─────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile):
    dest_dir = BASE_DIR / "data" / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    pages = 0
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = len(reader.pages)
    except Exception:
        pass

    rel = dest.relative_to(BASE_DIR).as_posix()
    return {"path": rel, "name": file.filename, "pages": pages}


# ── Run ────────────────────────────────────────────────────────────

class RunConfig(BaseModel):
    pdf: str
    enabled: dict
    pages_per_chunk: int = 10
    debug_mode: bool = False
    archive_on_finish: bool = False


@app.post("/api/run")
async def start_run(cfg: RunConfig):
    run_id = "run_" + uuid.uuid4().hex[:6]

    flags = []
    for step_id, step_flags in STEP_FLAGS.items():
        if cfg.enabled.get(step_id):
            flags.extend(step_flags)

    cmd = [
        sys.executable, "main.py",
        "--input_pdf", cfg.pdf,
        "--pages_per_chunk", str(cfg.pages_per_chunk),
        *flags,
        *(["--debug_mode"] if cfg.debug_mode else []),
        *(["--archive_files"] if cfg.archive_on_finish else []),
    ]

    threading.Thread(target=run_process, args=(cmd, run_id), daemon=True).start()

    RUN_CONFIGS[run_id] = cfg.model_dump()
    return {"run_id": run_id}


@app.post("/api/clean")
async def clean_outputs():
    proc = subprocess.run(
        [sys.executable, "main.py", "--clean_all"],
        input="yes\nyes\n",
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=proc.stderr or "clean_all failed")
    return {"status": "ok"}


@app.delete("/api/run/{run_id}")
async def cancel_run(run_id: str):
    proc = RUNS.get(run_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Run not found")
    proc.kill()
    return {"status": "killed"}


# ── WebSocket STREAM ───────────────────────────────────────────────

@app.websocket("/ws/{run_id}")
async def ws_stream(websocket: WebSocket, run_id: str):
    await websocket.accept()

    last_index = 0

    while True:
        logs = RUN_LOGS.get(run_id, [])

        while last_index < len(logs):
            await websocket.send_text(logs[last_index])
            last_index += 1

        proc = RUNS.get(run_id)

        if proc and proc.poll() is not None:
            break

        await asyncio.sleep(0.2)

    await websocket.close()


# ── Outputs ────────────────────────────────────────────────────────

@app.get("/api/outputs")
async def list_outputs():
    result = {}
    for cat_id, cat_path in CAT_PATHS.items():
        files = []

        if cat_id == "synth":
            f = cat_path / "synthetic_data.jsonl"
            if f.exists():
                files.append({
                    "name": f.name,
                    "size": _fmt_size(f.stat().st_size),
                    "ext": f.suffix.lstrip("."),
                })

        elif cat_path.exists():
            for f in sorted(cat_path.iterdir()):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "size": _fmt_size(f.stat().st_size),
                        "ext": f.suffix.lstrip("."),
                    })

        result[cat_id] = files

    return result


# ── Download ───────────────────────────────────────────────────────

@app.get("/api/download/{cat}/{name}")
async def download_file(cat: str, name: str):
    if cat not in CAT_PATHS:
        raise HTTPException(status_code=404, detail="Unknown category")

    base = CAT_PATHS[cat]

    if cat == "synth":
        path = base / "synthetic_data.jsonl"
    else:
        path = (base / name).resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, filename=name)


@app.get("/api/download-zip/{cat}")
async def download_zip(cat: str):
    if cat not in CAT_PATHS:
        raise HTTPException(status_code=404, detail="Unknown category")

    base = CAT_PATHS[cat]
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if cat == "synth":
            f = base / "synthetic_data.jsonl"
            if f.exists():
                zf.write(f, f.name)
        elif base.exists():
            for f in sorted(base.iterdir()):
                if f.is_file():
                    zf.write(f, f.name)

    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={cat}_outputs.zip"},
    )


# ── Dataset stats ─────────────────────────────────────────────────

def _count_lines(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return None


@app.get("/api/stats")
async def get_stats():
    out = BASE_DIR / "data" / "outputs"
    return {
        "equations_ft":  _count_lines(out / "formatted_equations" / "finetuning_dataset.jsonl"),
        "equations_cpt": _count_lines(out / "formatted_equations" / "pretraining_dataset.jsonl"),
        "qa_pairs":      _count_lines(out / "qa_pairs" / "all_qa_pairs.jsonl"),
        "synthetic":     _count_lines(out / "synthetic_data.jsonl"),
    }


# ── UI ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="ui", html=True), name="ui")