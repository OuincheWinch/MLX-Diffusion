import time
import uuid
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

import generator
from state import (
    GenerateRequest,
    _validate_lora_path,
    _JOBS_LOCK,
    JOBS,
    _job_queue,
    _prune_jobs,
    MAX_LORAS,
    MAX_QUEUED_IMAGES,
    record_pending_job,
    record_recoverable_job,
    get_recoverable_jobs,
    delete_recoverable_job,
    clear_recoverable_state,
)

router = APIRouter(tags=["jobs"])


@router.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is required")
    if len(req.loras) > MAX_LORAS:
        raise HTTPException(
            400,
            f"Maximum of {MAX_LORAS} LoRAs allowed per generation (received {len(req.loras)})"
        )
    info = generator.get_model_info(req.model)
    if not info:
        raise HTTPException(400, f"unknown model: {req.model}")
    req.model = info["id"]
    for lora in req.loras:
        err = _validate_lora_path(lora.path)
        if err:
            raise HTTPException(400, err)
    if not info.get("supports_guidance") and req.guidance is not None and req.guidance > 1.0 and info.get("engine") != "sdxl":
        raise HTTPException(
            400,
            f"{info['label']} does not support guidance > 1.0",
        )
    if req.loras and not info.get("supports_loras"):
        raise HTTPException(
            400,
            f"LoRAs are only supported on {generator.get_model_info(generator.DEFAULT_MODEL_ID)['label']} "
            f"(needs {info['lora_format']}-format LoRAs)",
        )
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        active_images = sum(
            getattr(j.get("request"), "batch", 1)
            for j in JOBS.values()
            if j.get("status") in ("generating", "queued")
        )
        if active_images + req.batch > MAX_QUEUED_IMAGES:
            raise HTTPException(
                429,
                f"Queue limit reached ({active_images}/{MAX_QUEUED_IMAGES} images in queue). Cannot queue {req.batch} more."
            )
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "request": req,
            "created_at": time.time(),
        }
        record_pending_job(job_id, req.model_dump())
        _prune_jobs()
    _job_queue.put(job_id)
    return {"job_id": job_id}


@router.get("/api/jobs")
def list_jobs(limit: int = 25):
    limit = min(max(1, limit), 50)
    with _JOBS_LOCK:
        active = [
            j for j in JOBS.values() if j.get("status") in ("generating", "queued")
        ]
        active.sort(key=lambda j: (0 if j.get("status") == "generating" else 1, j.get("created_at", 0)))

        finished = [
            j for j in JOBS.values() if j.get("status") not in ("generating", "queued")
        ]
        finished.sort(key=lambda j: j.get("finished_at", 0), reverse=True)

        combined = active + finished
        jobs = combined[:limit]

        out = []
        for j in jobs:
            prog = dict(j["progress"]) if j.get("progress") else {}
            if j.get("last_saved"):
                ls = j["last_saved"]
                prog.update({"saved_id": ls["id"], "saved_index": ls["index"], "batch": ls["batch"]})
            req = j.get("request")
            req_batch = getattr(req, "batch", 1) if req else (prog.get("batch") or 1)
            if "batch" not in prog and req_batch > 1:
                prog["batch"] = req_batch
            out.append({
                "id": j["id"],
                "status": j["status"],
                "phase": j.get("phase"),
                "phase_detail": j.get("phase_detail"),
                "progress": prog or None,
                "batch": req_batch,
                "model": getattr(req, "model", "") if req else "",
                "prompt": getattr(req, "prompt", "")[:80] if req else "",
                "finished_at": j.get("finished_at"),
                "created_at": j.get("created_at"),
            })
        return out


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        snapshot = dict(job)
        if "progress" in snapshot and isinstance(snapshot["progress"], dict):
            snapshot["progress"] = dict(snapshot["progress"])
        if "results" in snapshot:
            snapshot["results"] = list(snapshot["results"])
        if "partial_results" in snapshot:
            snapshot["partial_results"] = list(snapshot["partial_results"])
        if "last_result" in snapshot and isinstance(snapshot["last_result"], dict):
            snapshot["last_result"] = dict(snapshot["last_result"])
    out = {k: v for k, v in snapshot.items() if k not in ("request", "cancel_event")}
    if "request" in snapshot and hasattr(snapshot["request"], "model_dump"):
        out["request"] = snapshot["request"].model_dump()
    if snapshot.get("last_saved"):
        ls = dict(snapshot["last_saved"])
        out["last_saved"] = ls
        p = dict(out.get("progress") or {})
        p.update({"saved_id": ls["id"], "saved_index": ls["index"], "batch": ls["batch"]})
        out["progress"] = p
    return out


@router.post("/api/jobs/cancel-all")
def cancel_all_jobs():
    """Empty the queue, archive cancelled jobs to recovery, and kill the current generation."""
    killed = []
    running_event = None
    with _JOBS_LOCK:
        for job in JOBS.values():
            if job.get("status") in ("done", "error", "cancelled"):
                continue
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
            killed.append(job["id"])
            req_data = job["request"].model_dump() if hasattr(job.get("request"), "model_dump") else dict(job.get("request") or {})
            record_recoverable_job(job["id"], req_data, reason="cancelled")
            ev = job.get("cancel_event")
            if ev is not None:
                running_event = ev
    # Drain pending items from _job_queue
    try:
        while not _job_queue.empty():
            try:
                _job_queue.get_nowait()
                _job_queue.task_done()
            except Exception:
                break
    except Exception:
        pass

    if running_event is not None:
        running_event.set()
    generator.cancel_current()
    return {"cancelled": killed}


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] in ("done", "error", "cancelled"):
            raise HTTPException(400, f"job already {job['status']}")
        was_running = job["status"] == "generating"
        job["status"] = "cancelled"
        job["finished_at"] = time.time()
        req_data = job["request"].model_dump() if hasattr(job.get("request"), "model_dump") else dict(job.get("request") or {})
        record_recoverable_job(job_id, req_data, reason="cancelled")
        event = job.get("cancel_event")
    if was_running:
        if event is not None:
            event.set()
        generator.cancel_current()
    return {"status": "cancelled", "job_id": job_id}


@router.get("/api/queue/recovery")
def list_recoverable_queue():
    """Return all recoverable (cancelled or interrupted) jobs."""
    return {"items": get_recoverable_jobs()}


class RestoreQueueRequest(BaseModel):
    job_ids: list[str] = Field(default_factory=list)


@router.post("/api/queue/recovery/restore")
def restore_queue_jobs(req_body: RestoreQueueRequest):
    """Restore selected or all recoverable jobs by re-enqueueing them."""
    items = get_recoverable_jobs()
    if req_body.job_ids:
        to_restore = [x for x in items if x["id"] in req_body.job_ids]
    else:
        to_restore = list(items)
    restored_ids = []
    for item in to_restore:
        req_dict = item.get("request")
        if not req_dict:
            continue
        try:
            req_obj = GenerateRequest(**req_dict)
            new_job_id = uuid.uuid4().hex
            with _JOBS_LOCK:
                JOBS[new_job_id] = {
                    "id": new_job_id,
                    "status": "queued",
                    "request": req_obj,
                    "created_at": time.time(),
                }
                record_pending_job(new_job_id, req_obj.model_dump())
                _prune_jobs()
            _job_queue.put(new_job_id)
            delete_recoverable_job(item["id"])
            restored_ids.append(new_job_id)
        except Exception as e:
            print(f"[queue_recovery] restore error for {item.get('id')}: {e}")
    return {"restored": restored_ids, "count": len(restored_ids)}


@router.delete("/api/queue/recovery")
def clear_recovery(job_id: str | None = None, forget: bool = False):
    """Delete all recoverable records or a specific one.

    With `forget=true` (and no job_id) the pending-queue file is also wiped,
    so no interrupted job is restored on the next startup.
    """
    if job_id is None and forget:
        clear_recoverable_state()
    else:
        delete_recoverable_job(job_id)
    return {"status": "ok"}
