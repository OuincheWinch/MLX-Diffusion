"""Settings, engine status and Hugging Face cache management endpoints.

Backs the ⚙️ Parameters tab: user preferences (persisted in data/settings.json),
live Metal/pipeline status, and inspection/cleanup of the local HF hub cache.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import generator
import app_settings

router = APIRouter(tags=["settings"])

HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


class SettingsUpdate(BaseModel):
    artist_name: str | None = None
    default_output_format: str | None = None
    default_stealth: bool | None = None
    default_fast_vae: bool | None = None
    default_sampler: str | None = None
    default_cache_interval: int | None = None
    model_defaults: dict | None = None


@router.get("/api/settings")
def get_settings():
    return app_settings.get_settings()


@router.post("/api/settings")
def update_settings(req: SettingsUpdate):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        return app_settings.update_settings(updates)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/engine/status")
def engine_status():
    return generator.get_engine_status()


class EngineConfigRequest(BaseModel):
    memory_wired_limit_gb: float | None = None
    memory_krea_wired_limit_gb: float | None = None
    idle_kill_s_mflux: int | None = None
    idle_kill_s_sdxl: int | None = None


@router.post("/api/engine/config")
def update_engine_config(req: EngineConfigRequest):
    """Persist runtime engine tuning (wired limits, idle policies). Values are
    applied lazily — on the next generation start / idle rearm — and never
    interrupt a running generation or restart the daemon."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "nothing to update")
    try:
        app_settings.update_settings(updates)
    except ValueError as e:
        raise HTTPException(400, str(e))
    generator.rearm_engine_watchdogs()
    return generator.get_engine_status()


def _hf_cache_payload() -> dict:
    repos = []
    total = 0
    error = None
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for r in info.repos:
            size = int(getattr(r, "size_on_disk", 0) or 0)
            total += size
            repos.append(
                {
                    "repo_id": r.repo_id,
                    "repo_type": getattr(r, "repo_type", "model"),
                    "size_bytes": size,
                    "n_revisions": len(getattr(r, "revisions", []) or []),
                    "n_files": int(getattr(r, "nb_files", 0) or 0),
                    "last_accessed": getattr(r, "last_accessed", None),
                }
            )
        repos.sort(key=lambda x: x["size_bytes"], reverse=True)
    except Exception as e:
        error = str(e)

    return {
        "root": str(HF_HUB_CACHE),
        "exists": HF_HUB_CACHE.exists(),
        "total_bytes": total,
        "repos": repos,
        "error": error,
    }


@router.get("/api/hf/cache")
def hf_cache():
    return _hf_cache_payload()


class CacheClearRequest(BaseModel):
    repo_id: str | None = None


@router.post("/api/hf/cache/clear")
def clear_hf_cache(req: CacheClearRequest):
    before = _hf_cache_payload()
    if not req.repo_id:
        # Refuse a blanket wipe implicitly: the frontend must name a repo.
        raise HTTPException(400, "repo_id is required")

    target = next((r for r in before["repos"] if r["repo_id"] == req.repo_id), None)
    if target is None:
        raise HTTPException(404, f"{req.repo_id} not found in HF cache")

    # Refuse while a weights download for this repo is in-flight.
    try:
        import state

        with state._MODEL_DOWNLOAD_LOCK:
            for t in state.MODEL_DOWNLOAD_TASKS.values():
                if t.get("status") == "downloading" and t.get("repo_id") == req.repo_id:
                    raise HTTPException(409, f"{req.repo_id} is currently downloading")
    except HTTPException:
        raise
    except Exception:
        pass

    freed = target["size_bytes"]
    ok = generator._delete_hf_repo_cache(req.repo_id)
    if not ok:
        raise HTTPException(500, f"failed to remove {req.repo_id} from HF cache")
    return {"status": "cleared", "repo_id": req.repo_id, "freed_bytes": freed}
