import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import civitai_service
import generator
import hf_service
from huggingface_hub import HfApi, hf_hub_download
from state import (
    DATA_DIR,
    LORA_FILES_DIR,
    SDXL_LORA_DIR,
    DOWNLOAD_TASKS,
    _DOWNLOAD_LOCK,
    MODEL_DOWNLOAD_TASKS,
    _MODEL_DOWNLOAD_LOCK,
    _loras_lock,
    _read_loras,
    _write_loras,
    _inspect_safetensors,
)

router = APIRouter(tags=["downloads"])


class CivitaiImportRequest(BaseModel):
    url_or_id: str
    name: str | None = None
    api_key: str | None = None


class HFImportRequest(BaseModel):
    url_or_repo: str
    name: str | None = None
    token: str | None = None


class DirectUrlImportRequest(BaseModel):
    url: str
    name: str | None = None
    triggers: list[str] | None = None
    base_model: str | None = None


class UnifiedDownloadRequest(BaseModel):
    url_or_id: str
    name: str | None = None
    token: str | None = None
    api_key: str | None = None
    triggers: list[str] | None = None
    base_model: str | None = None


def make_download_progress_cb(task_id: str):
    def on_progress(pdata):
        with _DOWNLOAD_LOCK:
            t = DOWNLOAD_TASKS.get(task_id)
            if not t or t["status"] in ("cancelled", "error"):
                return
            if isinstance(pdata, dict):
                t["status_text"] = pdata.get("status", t["status_text"])
                t["progress"] = pdata.get("progress", t["progress"])
                t["downloaded_bytes"] = pdata.get("downloaded_bytes", t["downloaded_bytes"])
                t["total_bytes"] = pdata.get("total_bytes", t["total_bytes"])
                t["speed_mb_s"] = pdata.get("speed_mb_s", t["speed_mb_s"])
            else:
                t["status_text"] = str(pdata)
    return on_progress


def _fail_task(task_id: str, error: Exception | str, auth: bool = False, auth_msg: str = ""):
    with _DOWNLOAD_LOCK:
        t = DOWNLOAD_TASKS.get(task_id)
        if t:
            if auth:
                t["status"] = "error"
                t["error"] = f"AUTH_REQUIRED: {error}"
                t["status_text"] = auth_msg or "Authentication required"
            else:
                is_cancelled = "cancelled by user" in str(error).lower()
                t["status"] = "cancelled" if is_cancelled else "error"
                t["error"] = str(error)
                t["status_text"] = "Download cancelled" if is_cancelled else f"Download failed: {error}"
            t["finished_at"] = time.time()
            t.pop("cancel_event", None)


def _finish_task(task_id: str, entry: dict, status_text: str):
    with _DOWNLOAD_LOCK:
        t = DOWNLOAD_TASKS.get(task_id)
        if t:
            t["status"] = "done"
            t["progress"] = 1.0
            t["finished_at"] = time.time()
            t["status_text"] = status_text
            t["result"] = entry
            if "base_model" in entry:
                t["base_model"] = entry["base_model"]
            t.pop("cancel_event", None)


def _run_civitai_download(task_id: str, req: CivitaiImportRequest, meta: dict, vid: int, cancel_event: threading.Event):
    target_dir = SDXL_LORA_DIR if meta["base_model"] == "sdxl" else LORA_FILES_DIR
    on_progress = make_download_progress_cb(task_id)

    try:
        dest_path, enriched_meta = civitai_service.download_civitai_lora(
            version_id=vid,
            dest_dir=target_dir,
            custom_name=req.name,
            api_key=req.api_key,
            progress_cb=on_progress,
            cancel_event=cancel_event,
        )

        with _loras_lock:
            loras = _read_loras()
            display_name = req.name or enriched_meta.get("civitai_model_name") or dest_path.stem
            entry = {
                "name": display_name,
                "path": str(dest_path.resolve()),
                "scale": 1.0,
                "triggers": enriched_meta.get("triggers", []),
                "base_model": enriched_meta.get("base_model", "sdxl"),
                "sha256": enriched_meta.get("sha256"),
                "source": "civitai",
                "civitai_version_id": enriched_meta.get("civitai_version_id"),
                "civitai_model_id": enriched_meta.get("civitai_model_id"),
                "civitai_model_name": enriched_meta.get("civitai_model_name"),
                "civitai_version_name": enriched_meta.get("civitai_version_name"),
            }
            loras = [l for l in loras if l["name"] != entry["name"] and l["path"] != entry["path"]]
            loras.append(entry)
            _write_loras(loras)

        _finish_task(task_id, entry, f"Installed {entry['name']} successfully!")

    except civitai_service.CivitaiAuthError as ae:
        _fail_task(task_id, ae, auth=True, auth_msg="Authentication required")
    except Exception as e:
        _fail_task(task_id, e)


def _run_hf_download(task_id: str, req: HFImportRequest, meta: dict, cancel_event: threading.Event):
    target_dir = SDXL_LORA_DIR if meta["base_model"] == "sdxl" else LORA_FILES_DIR
    on_progress = make_download_progress_cb(task_id)

    try:
        dest_path, enriched_meta = hf_service.download_hf_lora(
            repo_id=meta["repo_id"],
            filename=meta["filename"],
            dest_dir=target_dir,
            revision=meta.get("revision", "main"),
            custom_name=req.name,
            token=req.token,
            progress_cb=on_progress,
            cancel_event=cancel_event,
        )

        detected_base, detected_triggers = _inspect_safetensors(dest_path)
        combined_triggers = list(enriched_meta.get("triggers", []))
        for dt in detected_triggers:
            if dt not in combined_triggers:
                combined_triggers.append(dt)

        display_name = req.name or enriched_meta.get("model_name") or dest_path.stem
        entry = {
            "name": display_name,
            "path": str(dest_path.resolve()),
            "scale": 1.0,
            "triggers": combined_triggers,
            "base_model": enriched_meta.get("base_model", detected_base or "sdxl"),
            "sha256": enriched_meta.get("sha256"),
            "source": "huggingface",
            "hf_repo_id": enriched_meta.get("repo_id"),
            "hf_filename": enriched_meta.get("filename"),
        }

        if enriched_meta.get("sha256"):
            try:
                civitai_data = civitai_service.fetch_by_hash(enriched_meta["sha256"])
                if civitai_data:
                    cm = civitai_service.extract_civitai_metadata(civitai_data)
                    if cm.get("civitai_version_id"):
                        entry["civitai_version_id"] = cm["civitai_version_id"]
                    if cm.get("civitai_model_id"):
                        entry["civitai_model_id"] = cm["civitai_model_id"]
                    if cm.get("civitai_model_name"):
                        entry["civitai_model_name"] = cm["civitai_model_name"]
                    if cm.get("civitai_version_name"):
                        entry["civitai_version_name"] = cm["civitai_version_name"]
            except Exception:
                pass

        with _loras_lock:
            loras = _read_loras()
            loras = [l for l in loras if l["name"] != entry["name"] and l["path"] != entry["path"]]
            loras.append(entry)
            _write_loras(loras)

        _finish_task(task_id, entry, f"Installed {entry['name']} successfully!")

    except hf_service.HFAuthError as ae:
        _fail_task(task_id, ae, auth=True, auth_msg="Hugging Face authentication required")
    except Exception as e:
        _fail_task(task_id, e)


def _start_hf_download_task(url_or_repo: str, custom_name: str | None = None, token: str | None = None) -> dict:
    parsed = hf_service.parse_hf_input(url_or_repo)
    if not parsed:
        raise HTTPException(400, f"Invalid Hugging Face URL or Repo ID: {url_or_repo}")

    try:
        meta = hf_service.fetch_hf_metadata(
            repo_id=parsed["repo_id"],
            filename=parsed.get("filename"),
            revision=parsed.get("revision", "main"),
            token=token,
        )
    except hf_service.HFAuthError as ae:
        raise HTTPException(401, str(ae))
    except ValueError as ve:
        raise HTTPException(404, str(ve))
    except Exception as e:
        raise HTTPException(400, str(e))

    with _DOWNLOAD_LOCK:
        for tid, t in DOWNLOAD_TASKS.items():
            if t.get("source") == "huggingface" and t.get("repo_id") == meta["repo_id"] and t.get("filename") == meta["filename"] and t.get("status") == "downloading":
                return {
                    "task_id": tid,
                    "status": "downloading",
                    "model_name": t.get("model_name"),
                    "base_model": t.get("base_model"),
                    "source": "huggingface",
                    "already_running": True,
                }

    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id,
        "source": "huggingface",
        "repo_id": meta["repo_id"],
        "filename": meta["filename"],
        "model_name": custom_name or meta.get("model_name") or meta["repo_id"],
        "base_model": meta.get("base_model", "sdxl"),
        "status": "downloading",
        "progress": 0.05,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed_mb_s": 0.0,
        "status_text": f"Connecting to Hugging Face for {meta.get('model_name', 'LoRA')}...",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "cancel_event": cancel_event,
    }

    with _DOWNLOAD_LOCK:
        DOWNLOAD_TASKS[task_id] = task

    req = HFImportRequest(url_or_repo=url_or_repo, name=custom_name, token=token)
    worker_thread = threading.Thread(
        target=_run_hf_download,
        args=(task_id, req, meta, cancel_event),
        daemon=True,
    )
    worker_thread.start()

    return {
        "task_id": task_id,
        "status": "downloading",
        "model_name": task["model_name"],
        "base_model": task["base_model"],
        "source": "huggingface",
    }


def _run_direct_url_download(task_id: str, req: DirectUrlImportRequest, cancel_event: threading.Event):
    url = req.url.strip()
    headers = {"User-Agent": civitai_service.DEFAULT_USER_AGENT}
    on_progress = make_download_progress_cb(task_id)

    fd, tmp_file = tempfile.mkstemp(dir=str(DATA_DIR), suffix=".download")
    tmp_path = Path(tmp_file)

    try:
        on_progress({"status": f"Connecting to {urlparse(url).netloc}...", "progress": 0.05, "downloaded_bytes": 0, "total_bytes": 0, "speed_mb_s": 0.0})
        parsed_host = (urlparse(url).hostname or "").lower()
        proxies = {"http": None, "https": None} if parsed_host in ("localhost", "127.0.0.1", "::1") else None
        with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30, proxies=proxies) as resp:
            resp.raise_for_status()

            detected_fname = None
            cd = resp.headers.get("Content-Disposition", "")
            if "filename=" in cd:
                m = re.search(r'filename=["\']?([^"\';]+)["\']?', cd)
                if m:
                    detected_fname = m.group(1).strip()
            if not detected_fname:
                url_path_name = Path(urlparse(resp.url).path).name
                if url_path_name and len(url_path_name) > 3:
                    detected_fname = url_path_name

            if not detected_fname or not detected_fname.endswith(".safetensors"):
                if req.name:
                    safe_n = re.sub(r"[^A-Za-z0-9._-]+", "_", req.name).strip("._")
                    detected_fname = f"{safe_n}.safetensors"
                else:
                    detected_fname = f"direct_lora_{task_id}.safetensors"

            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            start_time = time.time()
            last_speed_time = start_time
            last_speed_bytes = 0
            current_speed_mb = 0.0

            with os.fdopen(fd, "wb") as out_f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("Download cancelled by user")
                    if not chunk:
                        continue
                    out_f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    dt = now - last_speed_time
                    if dt >= 0.5:
                        current_speed_mb = ((downloaded - last_speed_bytes) / (1024 * 1024)) / dt
                        last_speed_time = now
                        last_speed_bytes = downloaded

                    frac = min(0.95, downloaded / float(total_bytes)) if total_bytes > 0 else 0.5
                    status_text = (
                        f"Downloading: {downloaded // (1024*1024)}MB / {total_bytes // (1024*1024)}MB "
                        f"({frac*100:.0f}%) - {current_speed_mb:.1f} MB/s"
                    ) if total_bytes > 0 else f"Downloading: {downloaded // (1024*1024)}MB - {current_speed_mb:.1f} MB/s"
                    on_progress({"status": status_text, "progress": frac, "downloaded_bytes": downloaded, "total_bytes": total_bytes, "speed_mb_s": current_speed_mb})

                out_f.flush()
                os.fsync(out_f.fileno())

        if not civitai_service.is_valid_safetensors(tmp_path):
            raise ValueError("Downloaded file is not a valid .safetensors archive.")

        on_progress({"status": "Inspecting model architecture...", "progress": 0.96, "downloaded_bytes": downloaded, "total_bytes": total_bytes, "speed_mb_s": 0.0})
        detected_base, detected_triggers = _inspect_safetensors(tmp_path)
        final_base = req.base_model or detected_base or "sdxl"
        final_triggers = req.triggers if req.triggers is not None else detected_triggers

        target_dir = SDXL_LORA_DIR if final_base == "sdxl" else LORA_FILES_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        final_dest = target_dir / detected_fname
        if final_dest.is_file():
            final_dest = target_dir / f"{final_dest.stem}_{task_id}{final_dest.suffix}"

        os.replace(tmp_path, final_dest)

        on_progress({"status": "Computing SHA-256 digest...", "progress": 0.98, "downloaded_bytes": downloaded, "total_bytes": total_bytes, "speed_mb_s": 0.0})
        sha = civitai_service.compute_file_sha256(final_dest)

        display_name = req.name or final_dest.stem
        entry = {
            "name": display_name,
            "path": str(final_dest.resolve()),
            "scale": 1.0,
            "triggers": final_triggers,
            "base_model": final_base,
            "sha256": sha,
            "source": "direct_url",
            "url": url,
        }

        try:
            civitai_data = civitai_service.fetch_by_hash(sha)
            if civitai_data:
                cm = civitai_service.extract_civitai_metadata(civitai_data)
                if cm.get("civitai_version_id"):
                    entry["civitai_version_id"] = cm["civitai_version_id"]
                if cm.get("civitai_model_id"):
                    entry["civitai_model_id"] = cm["civitai_model_id"]
                if cm.get("civitai_model_name"):
                    entry["civitai_model_name"] = cm["civitai_model_name"]
                if cm.get("civitai_version_name"):
                    entry["civitai_version_name"] = cm["civitai_version_name"]
                if not entry.get("triggers") and cm.get("triggers"):
                    entry["triggers"] = cm["triggers"]
        except Exception:
            pass

        with _loras_lock:
            loras = _read_loras()
            loras = [l for l in loras if l["name"] != entry["name"] and l["path"] != entry["path"]]
            loras.append(entry)
            _write_loras(loras)

        _finish_task(task_id, entry, f"Installed {entry['name']} successfully!")

    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        _fail_task(task_id, e)


def _start_civitai_download_task(req: CivitaiImportRequest):
    """Start asynchronous download and registration of a LoRA from Civitai, with HF auto-routing."""
    cleaned = req.url_or_id.strip()

    # Auto-routing: if user pasted an HF URL or repo ID, handle transparently
    if hf_service.parse_hf_input(cleaned):
        return _start_hf_download_task(cleaned, custom_name=req.name, token=req.api_key)

    vid = civitai_service.parse_civitai_input(cleaned, api_key=req.api_key)
    if not vid:
        raise HTTPException(400, f"Invalid Civitai or Hugging Face URL / ID: {req.url_or_id}")

    version_info = civitai_service.fetch_by_version_id(vid, api_key=req.api_key)
    if not version_info:
        raise HTTPException(404, f"Civitai model version {vid} not found")

    try:
        meta = civitai_service.extract_civitai_metadata(version_info)
    except ValueError as ve:
        raise HTTPException(400, str(ve))

    with _DOWNLOAD_LOCK:
        for tid, t in DOWNLOAD_TASKS.items():
            if t.get("version_id") == vid and t.get("status") == "downloading":
                return {
                    "task_id": tid,
                    "status": "downloading",
                    "model_name": t.get("model_name"),
                    "base_model": t.get("base_model"),
                    "source": "civitai",
                    "already_running": True,
                }

    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id,
        "source": "civitai",
        "version_id": vid,
        "model_name": meta.get("civitai_model_name") or f"LoRA #{vid}",
        "base_model": meta.get("base_model", "sdxl"),
        "status": "downloading",
        "progress": 0.05,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed_mb_s": 0.0,
        "status_text": f"Connecting to Civitai for {meta.get('civitai_model_name', 'LoRA')}...",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "cancel_event": cancel_event,
    }

    with _DOWNLOAD_LOCK:
        DOWNLOAD_TASKS[task_id] = task

    worker_thread = threading.Thread(
        target=_run_civitai_download,
        args=(task_id, req, meta, vid, cancel_event),
        daemon=True,
    )
    worker_thread.start()

    return {
        "task_id": task_id,
        "status": "downloading",
        "model_name": task["model_name"],
        "base_model": task["base_model"],
        "source": "civitai",
    }


def _start_direct_url_download_task(req: DirectUrlImportRequest):
    """Start asynchronous download and registration of a LoRA from any direct HTTP/HTTPS URL."""
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    if hf_service.parse_hf_input(url):
        return _start_hf_download_task(url, custom_name=req.name)

    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id,
        "source": "direct_url",
        "url": url,
        "model_name": req.name or Path(urlparse(url).path).stem or f"Direct LoRA #{task_id}",
        "base_model": req.base_model or "sdxl",
        "status": "downloading",
        "progress": 0.05,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed_mb_s": 0.0,
        "status_text": f"Connecting to {urlparse(url).netloc}...",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "cancel_event": cancel_event,
    }

    with _DOWNLOAD_LOCK:
        DOWNLOAD_TASKS[task_id] = task

    worker_thread = threading.Thread(
        target=_run_direct_url_download,
        args=(task_id, req, cancel_event),
        daemon=True,
    )
    worker_thread.start()

    return {
        "task_id": task_id,
        "status": "downloading",
        "model_name": task["model_name"],
        "base_model": task["base_model"],
        "source": "direct_url",
    }


@router.post("/api/loras/download")
def download_lora_unified(req: UnifiedDownloadRequest):
    """Universal endpoint to download and register a LoRA from Civitai, Hugging Face, or direct HTTP/HTTPS link."""
    raw = req.url_or_id.strip()
    if not raw:
        raise HTTPException(400, "URL or Model ID is required.")

    # 1. Hugging Face
    if hf_service.parse_hf_input(raw):
        return _start_hf_download_task(
            raw,
            custom_name=req.name,
            token=req.token,
        )

    # 2. Civitai
    if "civitai.com" in raw.lower() or "civitai.red" in raw.lower() or (raw.isdigit() and len(raw) <= 10):
        civitai_req = CivitaiImportRequest(
            url_or_id=raw,
            api_key=req.api_key or req.token,
        )
        return _start_civitai_download_task(civitai_req)

    # 3. Direct HTTP/HTTPS
    if raw.startswith(("http://", "https://")):
        direct_req = DirectUrlImportRequest(
            url=raw,
            name=req.name,
            triggers=req.triggers,
            base_model=req.base_model,
        )
        return _start_direct_url_download_task(direct_req)

    # 4. Fallback check with civitai parser
    try:
        if civitai_service.parse_civitai_input(raw):
            civitai_req = CivitaiImportRequest(
                url_or_id=raw,
                api_key=req.api_key or req.token,
            )
            return _start_civitai_download_task(civitai_req)
    except Exception:
        pass

    raise HTTPException(
        400,
        f"Could not identify download source for '{raw}'. Please provide a valid Civitai link/ID, Hugging Face repo/link, or direct HTTP/HTTPS link."
    )


@router.get("/api/loras/downloads")
def get_lora_downloads():
    """Get active and recent LoRA download tasks with live progress and transfer speed."""
    with _DOWNLOAD_LOCK:
        now = time.time()
        to_delete = [
            tid for tid, t in DOWNLOAD_TASKS.items()
            if t["status"] in ("done", "error", "cancelled")
            and t.get("finished_at")
            and (now - t["finished_at"] > 600)
        ]
        for tid in to_delete:
            DOWNLOAD_TASKS.pop(tid, None)

        res = []
        for tid, t in DOWNLOAD_TASKS.items():
            item = {k: v for k, v in t.items() if k != "cancel_event"}
            res.append(item)
        res.sort(key=lambda x: x.get("started_at", 0), reverse=True)
        return res


@router.delete("/api/loras/downloads/{task_id}")
def cancel_lora_download(task_id: str):
    """Cancel an ongoing LoRA download."""
    with _DOWNLOAD_LOCK:
        t = DOWNLOAD_TASKS.get(task_id)
        if not t:
            raise HTTPException(404, "Download task not found")
        if t["status"] == "downloading":
            ce = t.get("cancel_event")
            if ce:
                ce.set()
            t["status"] = "cancelled"
            t["status_text"] = "Download cancelled by user"
            t["finished_at"] = time.time()
        return {"status": "ok", "task_id": task_id}


class ModelDownloadRequest(BaseModel):
    model_id: str


class _ModelDownloadTqdm:
    """tqdm factory that reports per-chunk bytes into MODEL_DOWNLOAD_TASKS."""

    def __init__(self, task_id: str, cancel_event: threading.Event):
        self._task_id = task_id
        self._cancel_event = cancel_event

    def __call__(self, *args, **kwargs):
        from tqdm import tqdm

        return _ReportingTqdm(tqdm, self._task_id, self._cancel_event, *args, **kwargs)


class _ReportingTqdm:
    def __init__(self, base, task_id, cancel_event, *args, **kwargs):
        self._base = base(*args, **kwargs)
        self._task_id = task_id
        self._cancel_event = cancel_event

    def __getattr__(self, item):
        return getattr(self._base, item)

    def __enter__(self):
        self._base.__enter__()
        return self

    def __exit__(self, *exc):
        # Cancel the in-flight file when the user aborts mid-chunk.
        if self._cancel_event.is_set():
            raise RuntimeError("Download cancelled by user")
        return self._base.__exit__(*exc)

    def update(self, n=1):
        if self._cancel_event.is_set():
            raise RuntimeError("Download cancelled by user")
        with _MODEL_DOWNLOAD_LOCK:
            t = MODEL_DOWNLOAD_TASKS.get(self._task_id)
            if t and t["status"] == "downloading":
                t["downloaded_bytes"] = t.get("downloaded_bytes", 0) + int(n)
                total = t.get("total_bytes") or 0
                t["progress"] = min(0.99, t["downloaded_bytes"] / total) if total else 0.5
                now = time.time()
                dt = now - t.get("_speed_t", now)
                if dt >= 0.4:
                    t["speed_mb_s"] = ((t["downloaded_bytes"] - t.get("_speed_b", 0)) / (1024 * 1024)) / dt
                    t["_speed_t"] = now
                    t["_speed_b"] = t["downloaded_bytes"]
                if total:
                    mb_dl = t["downloaded_bytes"] // (1024 * 1024)
                    mb_tot = total // (1024 * 1024)
                    t["status_text"] = (
                        f"Downloading {mb_dl}/{mb_tot} MB ({t['progress']*100:.0f}%)"
                        + (f" - {t['speed_mb_s']:.1f} MB/s" if t.get("speed_mb_s") else "")
                    )
        return self._base.update(n)


def _fail_model_task(task_id: str, error: Exception | str):
    with _MODEL_DOWNLOAD_LOCK:
        t = MODEL_DOWNLOAD_TASKS.get(task_id)
        if t:
            is_cancelled = "cancelled by user" in str(error).lower()
            t["status"] = "cancelled" if is_cancelled else "error"
            t["error"] = str(error)
            t["status_text"] = "Download cancelled" if is_cancelled else f"Download failed: {error}"
            t["finished_at"] = time.time()
            t.pop("cancel_event", None)


def _finish_model_task(task_id: str, status_text: str):
    with _MODEL_DOWNLOAD_LOCK:
        t = MODEL_DOWNLOAD_TASKS.get(task_id)
        if t:
            t["status"] = "done"
            t["progress"] = 1.0
            t["finished_at"] = time.time()
            t["status_text"] = status_text
            t["result"] = {"model_id": t.get("model_id")}
            t.pop("cancel_event", None)


def _run_model_download(task_id: str, model_id: str, cancel_event: threading.Event):
    minfo = generator.get_model_info(model_id)
    repo = generator.model_download_repo(model_id, minfo)
    label = minfo.get("label", model_id)

    if not repo or repo.startswith("local:"):
        _fail_model_task(task_id, f"{label} is a local model — nothing to download from Hugging Face.")
        return

    target_dir = None
    if minfo.get("engine") == "sdxl":
        target_dir = minfo.get("model_dir")
        if target_dir:
            Path(target_dir).mkdir(parents=True, exist_ok=True)

    token = hf_service.get_hf_token()
    files = []
    total = 0
    try:
        # Precompute the file list + total bytes so the bar tracks global progress.
        info = HfApi(token=token).model_info(repo, files_metadata=True, token=token)
        for sib in getattr(info, "siblings", []) or []:
            files.append(sib.rfilename)
            total += getattr(sib, "size", 0) or 0
        with _MODEL_DOWNLOAD_LOCK:
            t = MODEL_DOWNLOAD_TASKS.get(task_id)
            if t:
                t["total_bytes"] = total
                t["status_text"] = f"Downloading {label} weights from {repo}…"
    except Exception as e:
        _fail_model_task(task_id, f"Could not inspect {repo}: {e}")
        return

    try:
        from huggingface_hub import hf_hub_download

        for i, fname in enumerate(files, start=1):
            if cancel_event.is_set():
                raise RuntimeError("Download cancelled by user")
            with _MODEL_DOWNLOAD_LOCK:
                t = MODEL_DOWNLOAD_TASKS.get(task_id)
                if t:
                    t["status_text"] = f"Downloading {label} weights: {fname} ({i}/{len(files)})"
            # Individual hf_hub_download calls DO thread tqdm_class through per-byte chunk
            # reporting (snapshot_download does not). Local dir for SDXL diffusers layouts,
            # default HF hub cache otherwise (so mflux + is_model_cached find the weights).
            hf_hub_download(
                repo_id=repo,
                filename=fname,
                token=token,
                local_dir=str(target_dir) if target_dir else None,
                tqdm_class=_ModelDownloadTqdm(task_id, cancel_event),
            )
        _finish_model_task(task_id, f"{label} installed successfully!")
    except Exception as e:
        _fail_model_task(task_id, e)


@router.post("/api/models/download")
def download_model(req: ModelDownloadRequest):
    """Start an asynchronous download of a model's weights with a live progress bar."""
    model_id = req.model_id.strip()
    minfo = generator.get_model_info(model_id)
    if model_id != minfo.get("id") and minfo.get("id") != req.model_id:
        raise HTTPException(404, f"Unknown model: {model_id}")

    if generator.is_model_cached(model_id):
        return {"status": "already_installed", "model_id": model_id, "model_name": minfo.get("label", model_id)}

    with _MODEL_DOWNLOAD_LOCK:
        for tid, t in MODEL_DOWNLOAD_TASKS.items():
            if t.get("model_id") == model_id and t.get("status") == "downloading":
                return {
                    "task_id": tid,
                    "status": "downloading",
                    "model_name": t.get("model_name"),
                    "already_running": True,
                }

    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id,
        "source": "model",
        "model_id": model_id,
        "model_name": minfo.get("label", model_id),
        "engine": minfo.get("engine", "mflux"),
        "status": "downloading",
        "progress": 0.0,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed_mb_s": 0.0,
        "status_text": f"Preparing download of {minfo.get('label', model_id)}…",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "cancel_event": cancel_event,
    }
    with _MODEL_DOWNLOAD_LOCK:
        MODEL_DOWNLOAD_TASKS[task_id] = task

    worker_thread = threading.Thread(
        target=_run_model_download,
        args=(task_id, model_id, cancel_event),
        daemon=True,
    )
    worker_thread.start()

    return {
        "task_id": task_id,
        "status": "downloading",
        "model_name": task["model_name"],
        "source": "model",
    }


@router.get("/api/models/downloads")
def get_model_downloads():
    """Get active and recent model (weights) download tasks with live progress."""
    with _MODEL_DOWNLOAD_LOCK:
        now = time.time()
        to_delete = [
            tid for tid, t in MODEL_DOWNLOAD_TASKS.items()
            if t["status"] in ("done", "error", "cancelled")
            and t.get("finished_at")
            and (now - t["finished_at"] > 600)
        ]
        for tid in to_delete:
            MODEL_DOWNLOAD_TASKS.pop(tid, None)

        res = []
        for tid, t in MODEL_DOWNLOAD_TASKS.items():
            item = {k: v for k, v in t.items() if k not in ("cancel_event", "_speed_t", "_speed_b")}
            res.append(item)
        res.sort(key=lambda x: x.get("started_at", 0), reverse=True)
        return res


@router.delete("/api/models/downloads/{task_id}")
def cancel_model_download(task_id: str):
    """Cancel an ongoing model (weights) download."""
    with _MODEL_DOWNLOAD_LOCK:
        t = MODEL_DOWNLOAD_TASKS.get(task_id)
        if not t:
            raise HTTPException(404, "Download task not found")
        if t["status"] == "downloading":
            ce = t.get("cancel_event")
            if ce:
                ce.set()
            t["status"] = "cancelled"
            t["status_text"] = "Download cancelled by user"
            t["finished_at"] = time.time()
        return {"status": "ok", "task_id": task_id}
