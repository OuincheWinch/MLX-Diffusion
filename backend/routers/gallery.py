import asyncio
import json
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import generator
import app_settings
from prompt_enhancer import EnhancementCancelled, enhance_prompt, list_engine_system_prompts
from state import (
    GENERATED_DIR,
    _gallery_lock,
    GALLERY_INDEX,
    _validate_image_id,
    _atomic_write_text,
    _read_loras,
)

router = APIRouter(tags=["gallery"])


def _matches_lora_filter(item: dict, lora_filter: str) -> bool:
    if not lora_filter or lora_filter.strip().lower() in ("", "all", "*"):
        return True
    loras = item.get("loras")
    if isinstance(loras, (dict, str)):
        loras = [loras]
    elif not isinstance(loras, list):
        loras = []

    clean = lora_filter.strip().lower()
    if clean in ("__none__", "none", "no-lora", "no_lora", "without-lora", "without_lora"):
        return len(loras) == 0
    if clean in ("__any__", "any", "has-lora", "has_lora", "with-lora", "with_lora"):
        return len(loras) > 0

    for l in loras:
        if isinstance(l, dict):
            name = (l.get("name") or "").strip().lower()
            path = (l.get("path") or "").strip().lower()
            stem = Path(path).stem.lower() if path else ""
            filename = Path(path).name.lower() if path else ""
            m_ver = str(l.get("modelVersionId") or l.get("civitai_version_id") or "").lower()
            m_id = str(l.get("modelId") or l.get("civitai_model_id") or "").lower()
            if clean in (name, stem, filename, m_ver, m_id):
                return True
            if clean in name or (stem and clean in stem):
                return True
        elif isinstance(l, str):
            val = l.strip().lower()
            stem = Path(val).stem.lower()
            if clean in (val, stem) or clean in val:
                return True
    return False


@router.get("/api/gallery")
def gallery(
    query: str = "",
    tags: str = "",
    sort: str = "newest",
    page: int = 1,
    limit: int = 24,
    model: str = "",
    lora: str = "",
):
    page = max(1, page)
    limit = min(max(1, limit), 100)
    with _gallery_lock:
        items = list(GALLERY_INDEX.values())
    if query:
        q = query.lower().strip()
        items = [
            i
            for i in items
            if q in i.get("prompt", "").lower() or q in str(i.get("seed", ""))
        ]
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    if tag_list:
        items = [i for i in items if all(t in i.get("tags", []) for t in tag_list)]
    if model:
        mrepo = next(
            (m["repo"] for m in generator.MODELS.values() if m["id"] == model), None
        )
        if mrepo:
            items = [i for i in items if i.get("model", generator.DEFAULT_MODEL) == mrepo]
    if lora:
        items = [i for i in items if _matches_lora_filter(i, lora)]
    items.sort(key=lambda i: i.get("created_at", 0), reverse=(sort == "newest"))
    total = len(items)
    start = (page - 1) * limit
    return {
        "total": total,
        "page": page,
        "items": items[start : start + limit],
    }


@router.get("/api/gallery/loras")
def gallery_loras(model: str = ""):
    with _gallery_lock:
        items = list(GALLERY_INDEX.values())
    if model:
        mrepo = next(
            (m["repo"] for m in generator.MODELS.values() if m["id"] == model), None
        )
        if mrepo:
            items = [i for i in items if i.get("model", generator.DEFAULT_MODEL) == mrepo]

    registry = _read_loras()
    reg_by_name = {l.get("name", "").lower(): l for l in registry if "name" in l}
    reg_by_stem = {Path(l.get("path", "")).stem.lower(): l for l in registry if "path" in l}

    counts = Counter()
    lora_details = {}
    with_lora = 0
    without_lora = 0

    for it in items:
        loras = it.get("loras")
        if isinstance(loras, (dict, str)):
            loras = [loras]
        elif not isinstance(loras, list):
            loras = []

        if not loras:
            without_lora += 1
        else:
            with_lora += 1
            seen_in_image = set()
            for l in loras:
                if isinstance(l, dict):
                    raw_name = (l.get("name") or (Path(l["path"]).stem if l.get("path") else None) or "Unknown").strip()
                else:
                    raw_name = Path(str(l)).stem.strip()

                match = reg_by_name.get(raw_name.lower()) or reg_by_stem.get(raw_name.lower())
                canonical = match.get("name") if match else raw_name
                base_model = match.get("base_model") if match else None

                if canonical.lower() not in seen_in_image:
                    seen_in_image.add(canonical.lower())
                    counts[canonical] += 1
                    if canonical not in lora_details:
                        lora_details[canonical] = {
                            "name": canonical,
                            "base_model": base_model,
                        }

    loras_list = [
        {
            "name": name,
            "count": counts[name],
            "base_model": lora_details.get(name, {}).get("base_model"),
        }
        for name, _ in counts.most_common()
    ]

    return {
        "total": len(items),
        "with_lora": with_lora,
        "without_lora": without_lora,
        "loras": loras_list,
    }


@router.get("/api/images/{image_id}")
def image_meta(image_id: str):
    _validate_image_id(image_id)
    with _gallery_lock:
        if image_id in GALLERY_INDEX:
            return GALLERY_INDEX[image_id]
    f = GENERATED_DIR / f"{image_id}.json"
    if not f.exists():
        raise HTTPException(404, "not found")
    data = json.loads(f.read_text(encoding="utf-8"))
    with _gallery_lock:
        GALLERY_INDEX[image_id] = data
    return data


@router.get("/api/images/{image_id}/file")
def image_file(image_id: str, thumb: bool = False):
    _validate_image_id(image_id)
    if thumb:
        try:
            tp = generator.thumbnail_path(image_id)
            if tp.exists():
                return FileResponse(
                    tp,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f'inline; filename="{image_id}_thumb.png"',
                        "Access-Control-Expose-Headers": "Content-Disposition",
                    },
                )
        except Exception:
            pass
    f = GENERATED_DIR / f"{image_id}.png"
    if not f.exists():
        f = GENERATED_DIR / f"{image_id}.jpeg"
    if not f.exists():
        f = GENERATED_DIR / f"{image_id}.jpg"
    if not f.exists():
        raise HTTPException(404, "not found")
    media_type = "image/jpeg" if f.suffix.lower() in (".jpeg", ".jpg") else "image/png"
    return FileResponse(
        f,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{f.name}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.post("/api/images/{image_id}/reveal")
def reveal_image(image_id: str):
    _validate_image_id(image_id)
    for sfx in (".png", ".jpeg", ".jpg"):
        f = GENERATED_DIR / f"{image_id}{sfx}"
        if f.exists():
            subprocess.run(["open", "-R", str(f)], check=False)
            return {"ok": True, "path": str(f)}
    raise HTTPException(404, "not found")



class TagsRequest(BaseModel):
    tags: list[str]


@router.post("/api/images/{image_id}/tags")
def set_tags(image_id: str, req: TagsRequest):
    _validate_image_id(image_id)
    f = GENERATED_DIR / f"{image_id}.json"
    if not f.exists():
        raise HTTPException(404, "not found")
    meta = json.loads(f.read_text(encoding="utf-8"))
    meta["tags"] = sorted({t.strip().lower() for t in req.tags if t.strip()})
    _atomic_write_text(f, json.dumps(meta, indent=2))
    with _gallery_lock:
        GALLERY_INDEX[image_id] = meta
    return meta


class UpscaleRequest(BaseModel):
    scale: int = Field(default=2, ge=2, le=4)


class PromptEnhanceRequest(BaseModel):
    prompt: str
    model: str = "flux2-klein-4b"
    loras: list[dict] = []
    format: str = "text"  # "text" | "json" (structured JSON prompt schema)


@router.post("/api/prompt/enhance")
async def enhance_prompt_route(req: PromptEnhanceRequest, request: Request):
    # The enhancer runs a small local LLM. If the caller disconnects (the user
    # hits Cancel), abort cooperatively instead of finishing the token budget.
    cancel_event = threading.Event()
    result: dict = {}

    def _worker():
        try:
            result["res"] = enhance_prompt(
                req.prompt,
                engine=req.model,
                loras=req.loras,
                output_format=req.format if req.format in ("text", "json") else "text",
                cancel_event=cancel_event,
            )
        except EnhancementCancelled:
            result["cancelled"] = True

    task = asyncio.get_running_loop().run_in_executor(None, _worker)
    while not task.done():
        if await request.is_disconnected():
            print(
                "[prompt_enhancer] client disconnected — cancelling enhancement",
                file=sys.stderr,
            )
            cancel_event.set()
            break
        await asyncio.sleep(0.2)
    try:
        await task
    except EnhancementCancelled:
        return {"cancelled": True, "original": req.prompt}
    except Exception as e:
        raise HTTPException(500, f"prompt enhancement failed: {e}")

    if result.get("cancelled"):
        return {"cancelled": True, "original": req.prompt}

    res = result.get("res") or {}
    if res.get("error"):
        raise HTTPException(500, f"prompt enhancement failed: {res['error']}")
    return res


class SystemPromptUpdate(BaseModel):
    engine_key: str
    instructions: str
    mode: str = "text"


@router.get("/api/prompt/enhancer/system-prompts")
def list_system_prompts():
    """Per-engine system prompts used by the prompt enhancer (view + preview)."""
    try:
        return {"engines": list_engine_system_prompts()}
    except Exception as e:
        raise HTTPException(500, f"failed to load enhancer profiles: {e}")


@router.post("/api/prompt/enhancer/system-prompts")
def update_system_prompt(req: SystemPromptUpdate):
    """Save a custom engine-guidance override ("" restores the built-in).

    ``mode`` selects which output-mode guidance to edit: "text" (prose prompts)
    or "json" (structured schema). The two overrides are stored independently."""
    if req.engine_key not in app_settings.PROMPT_ENHANCER_KEYS:
        raise HTTPException(400, f"unknown engine '{req.engine_key}'")
    mode = (req.mode or "text").strip().lower()
    if mode not in ("text", "json"):
        raise HTTPException(400, f"unknown mode '{req.mode}'")
    settings_key = "prompt_enhancer_json" if mode == "json" else "prompt_enhancer"
    try:
        app_settings.update_settings({settings_key: {req.engine_key: req.instructions}})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"engines": list_engine_system_prompts()}


@router.post("/api/images/{image_id}/upscale")
def upscale(image_id: str, req: UpscaleRequest):
    _validate_image_id(image_id)
    try:
        meta = generator.upscale_image(image_id, scale=req.scale)
        with _gallery_lock:
            GALLERY_INDEX[meta["id"]] = meta
        return meta
    except FileNotFoundError:
        raise HTTPException(404, "image not found")
    except Exception as e:
        raise HTTPException(500, f"upscale failed: {e}")


@router.delete("/api/images/{image_id}")
def delete_image(image_id: str):
    _validate_image_id(image_id)
    removed = False
    for suffix in ("png", "jpeg", "jpg", "json"):
        f = GENERATED_DIR / f"{image_id}.{suffix}"
        if f.exists():
            f.unlink()
            removed = True
    t = GENERATED_DIR / f"{image_id}_thumb.png"
    if t.exists():
        t.unlink()
    with _gallery_lock:
        GALLERY_INDEX.pop(image_id, None)
    if not removed:
        raise HTTPException(404, "not found")
    return {"deleted": image_id}
