from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import generator
import app_settings
from state import (
    LORA_FILES_DIR,
    SDXL_LORA_DIR,
    _loras_lock,
    _read_loras,
    _write_loras,
    _validate_lora_path,
    _inspect_safetensors,
    sync_lora_entry_with_civitai,
)

router = APIRouter(tags=["loras"])


@router.get("/api/models")
def list_models():
    items = []
    for m in generator.MODELS.values():
        entry = app_settings.apply_model_defaults(m["id"], m)
        entry["installed"] = generator.is_model_cached(m["id"])
        entry["download_repo"] = generator.model_download_repo(m["id"], m)
        entry["disk_usage_bytes"] = generator.model_disk_usage(m["id"], m)
        items.append(entry)
    return items


@router.delete("/api/models/{model_id}")
def remove_model(model_id: str):
    """Remove a model's weights from disk (SDXL local dir / HF hub cache / krea2 bundle)."""
    if model_id not in generator.MODELS:
        raise HTTPException(404, "unknown model")
    ok, reason = generator.uninstall_model(model_id)
    if not ok:
        raise HTTPException(409, reason or "model is in use")
    return {"status": "removed", "model_id": model_id}


class LocalInstallRequest(BaseModel):
    model_id: str
    path: str


@router.get("/api/models/local-sources")
def list_local_sources(model_id: str | None = None):
    """Already-downloaded weight locations (HF hub cache snapshots) the user can
    install from instead of re-downloading."""
    return {"sources": generator.list_local_model_sources(model_id)}


@router.post("/api/models/install-local")
def install_local_model(req: LocalInstallRequest):
    """Register an existing on-disk folder (or HF cache snapshot) as a model's
    weights. Nothing is copied: the model loads directly from this path."""
    if req.model_id not in generator.MODELS:
        raise HTTPException(404, "unknown model")
    ok, reason = generator.validate_local_model_dir(req.model_id, req.path)
    if not ok:
        raise HTTPException(400, reason or "invalid model directory")
    resolved = generator.resolve_local_model_path(req.model_id, req.path)
    if resolved is None:
        raise HTTPException(400, "path does not exist")
    app_settings.update_settings({"model_paths": {req.model_id: str(resolved)}})
    return {"status": "installed", "model_id": req.model_id, "local_path": str(resolved)}


@router.get("/api/loras")
def list_loras():
    return _read_loras()


@router.post("/api/loras/sync-civitai")
def sync_all_loras_civitai():
    """Scan and synchronize all registered LoRAs with Civitai API."""
    with _loras_lock:
        loras = _read_loras()
        updated = 0
        for l in loras:
            if sync_lora_entry_with_civitai(l, force=False):
                updated += 1
        if updated > 0:
            _write_loras(loras)
        return {
            "status": "ok",
            "total": len(loras),
            "updated": updated,
            "loras": loras,
        }


class LoRASaveRequest(BaseModel):
    name: str
    path: str
    scale: float = Field(default=1.0, ge=0.0, le=10.0)
    base_model: str | None = None
    triggers: list[str] = []
    sha256: str | None = None


@router.post("/api/loras")
def save_lora(req: LoRASaveRequest):
    err = _validate_lora_path(req.path)
    if err:
        raise HTTPException(400, err)
    loras = _read_loras()
    existing = next((l for l in loras if l.get("name") == req.name or l.get("path") == req.path), {})
    entry = {**existing, **req.model_dump(exclude_unset=True)}
    p = Path(req.path).expanduser()
    if p.is_file() and (not entry.get("base_model") or not entry.get("triggers")):
        detected_base, detected_triggers = _inspect_safetensors(p)
        if not entry.get("base_model"):
            entry["base_model"] = detected_base
        if not entry.get("triggers"):
            entry["triggers"] = detected_triggers
    entry.setdefault("base_model", "sdxl" if "sdxl" in req.path.lower() else "flux2")
    entry.setdefault("triggers", [])
    sync_lora_entry_with_civitai(entry)
    loras = [l for l in loras if l["name"] != req.name and l["path"] != req.path]
    loras.append(entry)
    _write_loras(loras)
    return entry


@router.delete("/api/loras/{name}")
def delete_lora(name: str):
    with _loras_lock:
        loras = _read_loras()
        target = next((l for l in loras if l["name"] == name), None)
        if not target:
            raise HTTPException(404, "not found")
        remaining = [l for l in loras if l["name"] != name]
        _write_loras(remaining)

        # Physical file removal if located inside local application folders
        raw_path = target.get("path", "")
        if raw_path:
            p = Path(raw_path).expanduser().resolve()
            if p.exists() and (
                p.is_relative_to(LORA_FILES_DIR.resolve())
                or p.is_relative_to(SDXL_LORA_DIR.resolve())
            ):
                try:
                    p.unlink()
                except OSError:
                    pass
        return {"deleted": name}
