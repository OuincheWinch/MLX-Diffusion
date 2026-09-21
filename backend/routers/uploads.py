import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from state import (
    LORA_FILES_DIR,
    SDXL_LORA_DIR,
    UPLOADS_DIR,
    MAX_LORA_UPLOAD_BYTES,
    _sanitize_component,
    _inspect_safetensors,
    _discover_local_loras,
    _read_loras,
    _write_loras,
    sync_lora_entry_with_civitai,
)

router = APIRouter(tags=["uploads"])

SUPPORTED_REF_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")


@router.get("/api/uploads/{filename}")
def get_uploaded_image(filename: str):
    safe = Path(filename).name
    path = UPLOADS_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "upload not found")
    media_type = "image/png"
    if safe.lower().endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif safe.lower().endswith(".webp"):
        media_type = "image/webp"
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{safe}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.post("/api/uploads")
def upload_reference_image(file: UploadFile):
    """Store a reference/init image for img2img generation."""
    if file.filename is None or not file.filename.lower().endswith(SUPPORTED_REF_EXTENSIONS):
        raise HTTPException(400, f"only {', '.join(SUPPORTED_REF_EXTENSIONS)} images are supported")
    safe_name = Path(file.filename).name
    if safe_name in (".", ".."):
        raise HTTPException(400, "invalid filename")

    is_heic = safe_name.lower().endswith((".heic", ".heif"))
    prefix = uuid.uuid4().hex[:12]

    if is_heic:
        # Stream raw HEIC/HEIF chunks to disk
        raw_dest = UPLOADS_DIR / f"raw_{prefix}_{safe_name}"
        total = 0
        with raw_dest.open("wb") as f:
            while chunk := file.file.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    f.close()
                    try:
                        raw_dest.unlink()
                    except OSError:
                        pass
                    raise HTTPException(413, "image file exceeds 50 MB limit")
                f.write(chunk)

        # Convert to standardized, upright RGB PNG
        dest = UPLOADS_DIR / f"{prefix}_{Path(safe_name).stem}.png"
        converted = False
        try:
            with Image.open(raw_dest) as img:
                img.load()
                img_trans = ImageOps.exif_transpose(img)
                if img_trans is None:
                    img_trans = img
                if img_trans.mode not in ("RGB", "RGBA"):
                    img_trans = img_trans.convert("RGB")
                img_trans.save(dest, format="PNG")
                converted = True
        except Exception:
            # Fallback to macOS hardware sips
            try:
                sips_res = subprocess.run(
                    ["/usr/bin/sips", "-s", "format", "png", str(raw_dest), "--out", str(dest)],
                    capture_output=True,
                    timeout=15,
                )
                if sips_res.returncode == 0 and dest.exists():
                    converted = True
            except Exception:
                pass

        try:
            raw_dest.unlink()
        except OSError:
            pass

        if not converted or not dest.exists():
            raise HTTPException(400, "failed to decode or convert HEIC/HEIF image")
    else:
        dest = UPLOADS_DIR / f"{prefix}_{safe_name}"
        total = 0
        with dest.open("wb") as f:
            while chunk := file.file.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > 50 * 1024 * 1024:
                    f.close()
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                    raise HTTPException(413, "image file exceeds 50 MB limit")
                f.write(chunk)
        # Verify image integrity with Pillow
        try:
            with Image.open(dest) as img:
                img.verify()
        except Exception:
            try:
                dest.unlink()
            except OSError:
                pass
            raise HTTPException(400, "uploaded file is corrupted or not a valid image")

    return {
        "path": str(dest.resolve()),
        "url": f"/api/uploads/{dest.name}",
        "name": dest.name,
    }


@router.post("/api/loras/upload")
def upload_lora(name: str, file: UploadFile):
    if file.filename is None or not file.filename.endswith(".safetensors"):
        raise HTTPException(400, "only .safetensors files are supported")
    safe_name = Path(file.filename).name
    if not safe_name.endswith(".safetensors") or safe_name in (".", ".."):
        raise HTTPException(400, "invalid filename")
    display_name = _sanitize_component(name) or Path(safe_name).stem[:80]
    existing = list(LORA_FILES_DIR.glob(f"*__{safe_name}"))
    if existing:
        dest = existing[0]
    else:
        dest = LORA_FILES_DIR / f"{display_name}__{safe_name}"
        written = 0
        fd, tmp = tempfile.mkstemp(dir=str(LORA_FILES_DIR), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                while chunk := file.file.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > MAX_LORA_UPLOAD_BYTES:
                        raise HTTPException(413, "file too large (max 8 GB)")
                    f.write(chunk)
            os.replace(tmp, dest)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    detected_base, detected_triggers = _inspect_safetensors(dest)
    if detected_base == "sdxl" and dest.parent != SDXL_LORA_DIR:
        sdxl_dest = SDXL_LORA_DIR / dest.name
        try:
            os.replace(dest, sdxl_dest)
            dest = sdxl_dest
        except Exception:
            pass

    _discover_local_loras()
    loras = _read_loras()
    resolved_path = str(dest.resolve())
    existing_meta = next((l for l in loras if l.get("path") == resolved_path or l.get("name") == display_name), {})
    entry = {
        **existing_meta,
        "name": display_name or dest.stem,
        "path": resolved_path,
        "scale": 1.0,
        "base_model": detected_base,
        "triggers": detected_triggers or existing_meta.get("triggers", []),
    }
    sync_lora_entry_with_civitai(entry)
    loras = [l for l in loras if l["name"] != entry["name"]]
    loras.append(entry)
    _write_loras(loras)
    return entry
