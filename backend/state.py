import json
import os
import queue
import re
import struct
import tempfile
import threading
import time
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel, Field

import generator
import civitai_service

# Directories & Files
DATA_DIR = generator.DATA_DIR
GENERATED_DIR = generator.GENERATED_DIR
LORA_FILES_DIR = DATA_DIR / "lora_files"
LORA_FILES_DIR.mkdir(parents=True, exist_ok=True)
LORAS_FILE = DATA_DIR / "loras.json"
SDXL_LORA_DIR = DATA_DIR / "SDXL"
SDXL_LORA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Validation & Limits
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_LORA_UPLOAD_BYTES = 8 * (1 << 30)  # 8 GB
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_KNOWN_TRIGGERS = {
    "evi@decrepPunk": ["evi@decrepPunk"],
}


def _validate_image_id(image_id: str) -> str:
    if not SAFE_ID_RE.match(image_id):
        raise HTTPException(400, "invalid id")
    return image_id


def _atomic_write_text(path: Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _sanitize_component(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip())
    return cleaned.strip("._")[:80]


def _validate_lora_path(path: str) -> str | None:
    """Return error message, or None if acceptable.
    Absolute paths must point to existing .safetensors file; else treated as HF repo id."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        return None
    try:
        resolved = p.resolve()
        if not resolved.exists():
            return f"LoRA file not found: {path}"
        if not resolved.is_file():
            return f"LoRA path is not a file: {path}"
        if not resolved.name.lower().endswith(".safetensors"):
            return "only .safetensors files are supported"
        return None
    except Exception as e:
        return f"Invalid LoRA path: {e}"


# --- Gallery Index ---
_gallery_lock = threading.Lock()
GALLERY_INDEX: dict[str, dict] = {}


def _init_gallery_index():
    with _gallery_lock:
        GALLERY_INDEX.clear()
        for jf in GENERATED_DIR.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "id" in data:
                    GALLERY_INDEX[data["id"]] = data
            except Exception:
                continue



# --- Hard Limits ---
MAX_LORAS = 16
MAX_QUEUED_IMAGES = 512


# --- Pydantic Request Models ---
class LoRA(BaseModel):
    path: str
    scale: float = Field(default=1.0, ge=0.0, le=10.0)


class GenerateRequest(BaseModel):
    prompt: str
    model: str = generator.DEFAULT_MODEL_ID
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=4, ge=1, le=50)
    guidance: float | None = Field(default=None, ge=0.0, le=10.0)
    seed: int | None = None
    quantization: int = 4
    loras: list[LoRA] = Field(default_factory=list, max_length=MAX_LORAS)
    batch: int = Field(default=1, ge=1, le=16)
    negative_prompt: str = ""
    sampler: str | None = None
    cache_interval: int = Field(default=1, ge=1, le=10)
    reference_images: list[str] = []
    reference_strength: float | None = None
    output_format: str = "png"
    stealth: bool = False
    fast_vae: bool = True


class TokenRequest(BaseModel):
    token: str


# --- Jobs & Generation Queue ---
_JOBS_LOCK = threading.Lock()
_JOBS_TTL_SECONDS = 3600
_MAX_FINISHED_JOBS = 50

JOBS: dict[str, dict] = {}
_job_queue: queue.Queue = queue.Queue()


def _prune_jobs():
    """Caller must hold _JOBS_LOCK."""
    now = time.time()
    dead = [
        jid
        for jid, j in JOBS.items()
        if j["status"] in ("done", "error", "cancelled")
        and now - j.get("finished_at", now) > _JOBS_TTL_SECONDS
    ]
    for jid in dead:
        del JOBS[jid]
    finished = [jid for jid, j in JOBS.items() if j["status"] in ("done", "error", "cancelled")]
    for jid in finished[:-_MAX_FINISHED_JOBS]:
        JOBS.pop(jid, None)


# --- Queue Recovery & Crash Resilience ---
QUEUE_RECOVERY_FILE = DATA_DIR / "queue_recovery.json"
PENDING_QUEUE_FILE = DATA_DIR / "pending_queue.json"
_RECOVERY_LOCK = threading.Lock()
_MAX_RECOVERABLE_RECORDS = 100


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"[queue_recovery] warning loading {path.name}: {e}")
    return []


def _save_json_list(path: Path, data: list[dict]):
    try:
        _atomic_write_text(path, json.dumps(data, indent=2))
    except Exception as e:
        print(f"[queue_recovery] error saving {path.name}: {e}")


def record_pending_job(job_id: str, request_data: dict):
    with _RECOVERY_LOCK:
        items = _load_json_list(PENDING_QUEUE_FILE)
        items = [x for x in items if x.get("id") != job_id]
        items.append({
            "id": job_id,
            "request": request_data,
            "enqueued_at": time.time(),
        })
        _save_json_list(PENDING_QUEUE_FILE, items)


def remove_pending_job(job_id: str):
    with _RECOVERY_LOCK:
        items = _load_json_list(PENDING_QUEUE_FILE)
        new_items = [x for x in items if x.get("id") != job_id]
        if len(new_items) != len(items):
            _save_json_list(PENDING_QUEUE_FILE, new_items)


def record_recoverable_job(job_id: str, request_data: dict, reason: str = "cancelled"):
    with _RECOVERY_LOCK:
        # Remove from pending queue file
        pend = _load_json_list(PENDING_QUEUE_FILE)
        pend = [x for x in pend if x.get("id") != job_id]
        _save_json_list(PENDING_QUEUE_FILE, pend)

        # Add to recovery file
        records = _load_json_list(QUEUE_RECOVERY_FILE)
        records = [r for r in records if r.get("id") != job_id]
        records.insert(0, {
            "id": job_id,
            "reason": reason,
            "timestamp": time.time(),
            "request": request_data,
        })
        _save_json_list(QUEUE_RECOVERY_FILE, records[:_MAX_RECOVERABLE_RECORDS])


def get_recoverable_jobs() -> list[dict]:
    with _RECOVERY_LOCK:
        return _load_json_list(QUEUE_RECOVERY_FILE)


def delete_recoverable_job(job_id: str | None = None):
    with _RECOVERY_LOCK:
        if job_id is None:
            _save_json_list(QUEUE_RECOVERY_FILE, [])
        else:
            records = _load_json_list(QUEUE_RECOVERY_FILE)
            records = [r for r in records if r.get("id") != job_id]
            _save_json_list(QUEUE_RECOVERY_FILE, records)


def clear_recoverable_jobs():
    delete_recoverable_job(None)


def clear_recoverable_state():
    """Clear the recovery archive AND the pending-queue file so nothing is
    restored on next startup (clear & forget)."""
    with _RECOVERY_LOCK:
        _save_json_list(QUEUE_RECOVERY_FILE, [])
        _save_json_list(PENDING_QUEUE_FILE, [])


def _init_queue_recovery_on_startup():
    """Detect crash / abrupt shutdown from previous session and migrate pending jobs to recovery."""
    with _RECOVERY_LOCK:
        pending = _load_json_list(PENDING_QUEUE_FILE)
        if pending:
            records = _load_json_list(QUEUE_RECOVERY_FILE)
            for item in pending:
                records = [r for r in records if r.get("id") != item["id"]]
                records.insert(0, {
                    "id": item["id"],
                    "reason": "interrupted",
                    "timestamp": item.get("enqueued_at", time.time()),
                    "interrupted_at": time.time(),
                    "request": item.get("request", {}),
                })
            _save_json_list(QUEUE_RECOVERY_FILE, records[:_MAX_RECOVERABLE_RECORDS])
            _save_json_list(PENDING_QUEUE_FILE, [])
            print(f"[queue_recovery] Restored {len(pending)} interrupted jobs to recovery archive", flush=True)


_init_queue_recovery_on_startup()


def _worker():
    while True:
        try:
            job_id = _job_queue.get()
        except BaseException:
            break
        try:
            cancel_event = threading.Event()
            with _JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None or job.get("status") == "cancelled":
                    remove_pending_job(job_id)
                    continue
                req = job["request"]
                ref_imgs = list(req.reference_images)
                variant = "edit" if (req.model in ("flux2-klein-4b", "flux2-klein-9b") and ref_imgs) else "standard"
                is_loaded = generator.is_pipeline_loaded(
                    model_id=req.model,
                    quantization=req.quantization,
                    loras=[l.model_dump() for l in req.loras],
                    variant=variant,
                )
                job["status"] = "generating"
                job["phase"] = "preparing" if is_loaded else "loading_model"
                job["phase_detail"] = "Preparing prompt conditioning & latents..." if is_loaded else "Loading model weights into Apple Silicon unified memory..."
                job["cancel_event"] = cancel_event

            results = []
            status = "generating"
            try:
                req = job["request"]
                batch = max(1, req.batch)
                base_seed = req.seed if req.seed is not None else int(time.time())
                for i in range(batch):
                    img_start_time = time.time()
                    if i > 0:
                        with _JOBS_LOCK:
                            job["phase"] = "preparing"
                            job["phase_detail"] = f"Preparing image {i + 1}/{batch}..."

                    def on_step(t):
                        done = t + 1
                        elapsed = time.time() - img_start_time
                        eta = elapsed / done * (req.steps - done) if done else None
                        with _JOBS_LOCK:
                            job["phase"] = "generating"
                            job["phase_detail"] = f"Denoising step {done}/{req.steps}..."
                            job["progress"] = {
                                "step": done,
                                "steps": req.steps,
                                "elapsed": round(elapsed, 1),
                                "eta_seconds": round(eta, 1) if eta is not None else None,
                                "image_index": i,
                                "batch": batch,
                            }

                    def on_phase(phase, detail=""):
                        with _JOBS_LOCK:
                            job["phase"] = phase
                            if detail:
                                job["phase_detail"] = detail

                    with _JOBS_LOCK:
                        if job["status"] == "cancelled":
                            raise generator.GenerationCancelled()

                    ref_imgs = list(req.reference_images)

                    meta = generator.generate(
                        prompt=req.prompt,
                        model=req.model,
                        width=req.width,
                        height=req.height,
                        steps=req.steps,
                        guidance=req.guidance,
                        seed=base_seed + 1024 * i,
                        quantization=req.quantization,
                        loras=[l.model_dump() for l in req.loras],
                        progress_cb=on_step,
                        phase_cb=on_phase,
                        cancel_event=cancel_event,
                        negative_prompt=req.negative_prompt,
                        sampler=req.sampler,
                        cache_interval=req.cache_interval,
                        reference_images=ref_imgs,
                        image_strength=req.reference_strength,
                        output_format=req.output_format,
                        stealth=req.stealth,
                        fast_vae=req.fast_vae,
                    )
                    results.append(meta)
                    with _gallery_lock:
                        GALLERY_INDEX[meta["id"]] = meta
                    with _JOBS_LOCK:
                        job["phase"] = "saving"
                        job["phase_detail"] = "Finalizing and saving image..."
                        job["last_saved"] = {"id": meta["id"], "index": i + 1, "batch": batch}
                        job["last_result"] = meta
                        job["partial_results"] = list(results)

                if len(results) == batch and results:
                    status = "done"
                else:
                    status = "error"
            except generator.GenerationCancelled:
                status = "cancelled"
            except Exception as e:
                import traceback
                traceback.print_exc()
                status = "error"
                with _JOBS_LOCK:
                    job["error"] = str(e)
                generator._drop_mflux_pipeline()
            except BaseException as be:
                import traceback
                traceback.print_exc()
                status = "error"
                with _JOBS_LOCK:
                    job["error"] = f"Interrupted: {type(be).__name__}"
                generator._drop_mflux_pipeline()
            finally:
                remove_pending_job(job_id)
                with _JOBS_LOCK:
                    job.pop("cancel_event", None)
                    if job.get("status") == "cancelled" or status == "cancelled":
                        job["status"] = "cancelled"
                        if results:
                            job["partial_results"] = results
                        req_data = job["request"].model_dump() if hasattr(job.get("request"), "model_dump") else dict(job.get("request") or {})
                        record_recoverable_job(job_id, req_data, reason="cancelled")
                    elif status == "done" and results:
                        job["status"] = "done"
                        job["phase"] = "done"
                        job["phase_detail"] = "Complete"
                        job["result"] = results[-1]
                        job["results"] = results
                        total_gen_time = round(sum(r.get("generation_time", 0) for r in results), 1)
                        if "progress" in job and isinstance(job["progress"], dict):
                            job["progress"]["elapsed"] = total_gen_time
                            job["progress"]["eta_seconds"] = 0
                    else:
                        job["status"] = "error"
                        if "error" not in job:
                            job["error"] = "Generation stopped or failed before producing an image"
                        if results:
                            job["partial_results"] = results
                    job["finished_at"] = time.time()
                    _prune_jobs()
        except Exception as outer_be:
            print(f"[worker] critical job loop error: {outer_be}", flush=True)
            with _JOBS_LOCK:
                if job_id in JOBS and JOBS[job_id].get("status") == "generating":
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = f"Fatal worker error: {outer_be}"
                    JOBS[job_id]["finished_at"] = time.time()
        finally:
            _job_queue.task_done()


# Start background worker daemon
threading.Thread(target=_worker, daemon=True).start()


# --- LoRA Registry State & Helpers ---
_loras_lock = threading.Lock()


def _read_loras() -> list[dict]:
    with _loras_lock:
        if LORAS_FILE.exists():
            try:
                return json.loads(LORAS_FILE.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []


def _write_loras(loras: list[dict]):
    with _loras_lock:
        LORAS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(LORAS_FILE, json.dumps(loras, indent=2))


def _inspect_safetensors(path: Path) -> tuple[str | None, list[str]]:
    """Fast inspection of safetensors header to determine base_model and triggers."""
    if not civitai_service.is_valid_safetensors(path):
        return None, []
    try:
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        meta = header.get("__metadata__", {})
        keys = [k for k in header.keys() if k != "__metadata__"]

        version = str(meta.get("ss_base_model_version", "")).lower()
        arch = str(meta.get("modelspec.architecture", "")).lower()
        title = str(meta.get("modelspec.title", "")).lower()
        name_lower = path.name.lower()

        if "sdxl" in version or "sdxl" in arch or "sdxl" in name_lower or any(k.startswith(("lora_unet_", "lora_te1_", "lora_te2_")) for k in keys[:20]):
            base_model = "sdxl"
        elif "krea" in name_lower or "krea" in title or "krea" in version:
            base_model = "krea2"
        elif (
            "z-image" in name_lower
            or "z_image" in name_lower
            or "zit" in name_lower
            or "zimage" in name_lower
            or "zimage" in version
            or "z-image" in version
            or "z_image" in version
            or "zimage" in arch
            or "z-image" in arch
        ):
            base_model = "z-image"
        elif any("double_blocks" in k for k in keys[:20]) or "flux" in version or "flux" in arch or "flux" in name_lower:
            base_model = "flux2"
        else:
            base_model = "sdxl" if "sdxl" in name_lower else "flux2"

        triggers = []
        if "ss_tag_frequency" in meta:
            try:
                tf = json.loads(meta["ss_tag_frequency"])
                for cat, tags in tf.items():
                    for tag, count in tags.items():
                        if count >= 10 and len(tag) > 2 and tag.lower() not in (
                            "simple background", "white background", "no humans", "1girl", "solo", "looking at viewer"
                        ):
                            if tag not in triggers:
                                triggers.append(tag)
            except Exception:
                pass
        if "modelspec.trigger_phrase" in meta:
            phrase = meta["modelspec.trigger_phrase"]
            if phrase and phrase not in triggers:
                triggers.append(phrase)

        return base_model, triggers[:3]
    except Exception:
        return None, []


def sync_lora_entry_with_civitai(entry: dict, force: bool = False) -> bool:
    """Enrich single LoRA entry with SHA256 and Civitai metadata. Returns True if changed."""
    changed = False
    p = Path(entry.get("path", ""))
    if not p.is_file():
        return False

    if not entry.get("sha256") or force:
        try:
            entry["sha256"] = civitai_service.compute_file_sha256(p)
            changed = True
        except Exception:
            return changed

    if (not entry.get("civitai_version_id") or force) and entry.get("sha256"):
        try:
            civitai_data = civitai_service.fetch_by_hash(entry["sha256"])
            if civitai_data:
                cm = civitai_service.extract_civitai_metadata(civitai_data)
                if cm.get("civitai_version_id"):
                    entry["civitai_version_id"] = cm["civitai_version_id"]
                    changed = True
                if cm.get("civitai_model_id"):
                    entry["civitai_model_id"] = cm["civitai_model_id"]
                    changed = True
                if cm.get("civitai_model_name"):
                    entry["civitai_model_name"] = cm["civitai_model_name"]
                    changed = True
                if cm.get("civitai_version_name"):
                    entry["civitai_version_name"] = cm["civitai_version_name"]
                    changed = True
                if cm.get("base_model"):
                    entry["base_model"] = cm["base_model"]
                    changed = True
                current_triggers = list(entry.get("triggers", []))
                for t in cm.get("triggers", []):
                    if t and t not in current_triggers:
                        current_triggers.append(t)
                        changed = True
                entry["triggers"] = current_triggers
        except Exception as e:
            print(f"[civitai] Failed querying Civitai for {entry.get('name')}: {e}", flush=True)

    return changed


def _discover_local_loras():
    """Register safetensors dropped into lora_files/ or SDXL/. Runs at startup and after uploads."""
    loras = _read_loras()
    changed = False

    valid_loras = []
    for l in loras:
        p = Path(l.get("path", ""))
        if not p.is_file() or not civitai_service.is_valid_safetensors(p):
            print(f"[loras] Purging missing or invalid LoRA entry: {l.get('name')} ({p})", flush=True)
            changed = True
            continue
        valid_loras.append(l)
    loras = valid_loras

    for l in loras:
        l.setdefault("triggers", [])
        l.setdefault("base_model", "sdxl" if "SDXL" in l.get("path", "") or "sdxl" in l.get("name", "").lower() else "flux2")
        if not l["triggers"]:
            l["triggers"] = next(
                (t for k, t in _KNOWN_TRIGGERS.items() if k in l["name"]), [])
            if l["triggers"]:
                changed = True
        if not l.get("civitai_version_id") or not l.get("sha256"):
            if sync_lora_entry_with_civitai(l):
                changed = True

    known_paths = {l["path"] for l in loras}
    known_names = {l["name"] for l in loras}
    for folder in (LORA_FILES_DIR, SDXL_LORA_DIR):
        for f in sorted(folder.glob("*.safetensors")):
            if "f42SDXL" in f.name or "Juggernaut" in f.name:
                continue
            try:
                if f.stat().st_size > 2 * (1 << 30):
                    continue
            except OSError:
                pass
            resolved = str(f.resolve())
            if resolved in known_paths:
                continue
            name = f.stem.split("__", 1)[-1]
            if name in known_names:
                continue
            base, triggers = _inspect_safetensors(f)
            if not base:
                continue
            entry = {
                "name": name,
                "path": resolved,
                "scale": 1.0,
                "triggers": triggers or next((t for k, t in _KNOWN_TRIGGERS.items() if k in name), []),
                "base_model": base,
            }
            old = next((l for l in loras if l["name"] == entry["name"]), None)
            if old and old.get("base_model"):
                entry["base_model"] = old["base_model"]
            sync_lora_entry_with_civitai(entry)
            loras = [l for l in loras if l["name"] != entry["name"]]
            loras.append(entry)
            known_paths.add(resolved)
            changed = True
    if changed:
        _write_loras(loras)



# --- Async Downloads State ---
DOWNLOAD_TASKS: dict[str, dict] = {}
_DOWNLOAD_LOCK = threading.RLock()

# --- Async Model (weights) Download State ---
MODEL_DOWNLOAD_TASKS: dict[str, dict] = {}
_MODEL_DOWNLOAD_LOCK = threading.RLock()
