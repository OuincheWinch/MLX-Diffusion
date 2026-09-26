import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

import civitai_service
from state import (
    LORA_FILES_DIR,
    SDXL_LORA_DIR,
    UPLOADS_DIR,
    MAX_LORA_UPLOAD_BYTES,
    MAX_REFERENCE_UPLOAD_BYTES,
    SUPPORTED_REFERENCE_EXTENSIONS,
    _sanitize_component,
    _inspect_safetensors,
    _read_loras,
    _upsert_lora_entries,
    sync_lora_entry_with_civitai,
)

router = APIRouter(tags=["uploads"])

_MAX_IMAGE_PIXELS = 100_000_000
_MAX_IMAGE_SIDE = 16_384


def _safe_image_name(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200 or "\x00" in value or "/" in value or "\\" in value:
        raise HTTPException(400, "invalid filename")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise HTTPException(400, "invalid filename")
    if Path(value).name != value:
        raise HTTPException(400, "invalid filename")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:180]
    if not cleaned or cleaned in (".", ".."):
        raise HTTPException(400, "invalid filename")
    return cleaned


def _create_private_file(path: Path) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)


def _validate_image_file(path: Path):
    try:
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0 or image.width > _MAX_IMAGE_SIDE or image.height > _MAX_IMAGE_SIDE:
                raise HTTPException(400, "invalid image dimensions")
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise HTTPException(400, "image is too large")
            image.verify()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, "uploaded file is corrupted or not a valid image") from e


def _stream_upload(upload: UploadFile, destination: Path, limit: int, error_message: str):
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := upload.file.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, error_message)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    finally:
        try:
            destination.chmod(0o600)
        except OSError:
            pass
    return total


@router.get("/api/uploads/{filename}")
def get_uploaded_image(filename: str):
    safe = _safe_image_name(filename)
    if not safe.lower().endswith(SUPPORTED_REFERENCE_EXTENSIONS):
        raise HTTPException(404, "upload not found")
    path = (UPLOADS_DIR / safe).resolve()
    if not path.is_relative_to(UPLOADS_DIR.resolve()) or not path.is_file():
        raise HTTPException(404, "upload not found")
    media_type = "image/png"
    if safe.lower().endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif safe.lower().endswith(".webp"):
        media_type = "image/webp"
    elif safe.lower().endswith((".heic", ".heif")):
        media_type = "image/heic"
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
    if file.filename is None:
        raise HTTPException(400, "image filename is required")
    safe_name = _safe_image_name(file.filename)
    if not safe_name.lower().endswith(SUPPORTED_REFERENCE_EXTENSIONS):
        raise HTTPException(400, f"only {', '.join(SUPPORTED_REFERENCE_EXTENSIONS)} images are supported")
    prefix = uuid.uuid4().hex[:12]
    is_heic = safe_name.lower().endswith((".heic", ".heif"))
    if is_heic:
        raw_path = UPLOADS_DIR / f".raw-{prefix}-{safe_name}"
        final_path = UPLOADS_DIR / f"{prefix}_{Path(safe_name).stem}.png"
        convert_path = UPLOADS_DIR / f".convert-{prefix}.png"
        _create_private_file(raw_path)
        try:
            _stream_upload(file, raw_path, MAX_REFERENCE_UPLOAD_BYTES, "image file exceeds 50 MB limit")
            _validate_image_file(raw_path)
            _create_private_file(convert_path)
            converted = False
            try:
                with Image.open(raw_path) as image:
                    image.load()
                    if image.width > _MAX_IMAGE_SIDE or image.height > _MAX_IMAGE_SIDE or image.width * image.height > _MAX_IMAGE_PIXELS:
                        raise ValueError("image is too large")
                    converted_image = ImageOps.exif_transpose(image) or image
                    if converted_image.mode not in ("RGB", "RGBA"):
                        converted_image = converted_image.convert("RGB")
                    converted_image.save(convert_path, format="PNG")
                    converted = True
            except Exception:
                try:
                    convert_path.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["/usr/bin/sips", "-s", "format", "png", str(raw_path), "--out", str(convert_path)],
                        capture_output=True,
                        timeout=15,
                    )
                    converted = result.returncode == 0 and convert_path.is_file()
                except Exception:
                    converted = False
            if not converted:
                raise HTTPException(400, "failed to decode or convert HEIC/HEIF image")
            try:
                convert_path.chmod(0o600)
            except OSError:
                pass
            _validate_image_file(convert_path)
            os.replace(convert_path, final_path)
            return {"path": str(final_path.resolve()), "url": f"/api/uploads/{final_path.name}", "name": final_path.name}
        finally:
            for path in (raw_path, convert_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
    destination = UPLOADS_DIR / f"{prefix}_{safe_name}"
    _create_private_file(destination)
    try:
        _stream_upload(file, destination, MAX_REFERENCE_UPLOAD_BYTES, "image file exceeds 50 MB limit")
        _validate_image_file(destination)
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return {"path": str(destination.resolve()), "url": f"/api/uploads/{destination.name}", "name": destination.name}


@router.post("/api/loras/upload")
def upload_lora(name: str, file: UploadFile):
    if file.filename is None or not file.filename.lower().endswith(".safetensors"):
        raise HTTPException(400, "only .safetensors files are supported")
    safe_name = _sanitize_component(file.filename)
    if not safe_name.lower().endswith(".safetensors") or len(safe_name) > 180:
        raise HTTPException(400, "invalid filename")
    display_name = _sanitize_component(name) or Path(safe_name).stem[:80]
    if not display_name:
        raise HTTPException(400, "invalid LoRA name")
    fd, tmp_name = tempfile.mkstemp(dir=str(LORA_FILES_DIR), prefix=".upload-", suffix=".part")
    tmp_path = Path(tmp_name)
    dest = None
    try:
        with os.fdopen(fd, "wb") as output:
            fd = None
            written = 0
            while chunk := file.file.read(8 * 1024 * 1024):
                written += len(chunk)
                if written > MAX_LORA_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large (max 8 GB)")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not civitai_service.is_valid_safetensors(tmp_path):
            raise HTTPException(400, "uploaded file is not a valid .safetensors archive")
        detected_base, detected_triggers = _inspect_safetensors(tmp_path)
        if not detected_base:
            raise HTTPException(400, "could not inspect LoRA architecture")
        target_dir = SDXL_LORA_DIR if detected_base == "sdxl" else LORA_FILES_DIR
        dest = (target_dir.resolve() / f"{display_name}__{safe_name}").resolve()
        if not dest.is_relative_to(target_dir.resolve()):
            raise HTTPException(400, "invalid LoRA destination")
        os.replace(tmp_path, dest)
        tmp_path = None
    finally:
        if fd:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    resolved_path = str(dest.resolve())
    existing = next(
        (entry for entry in _read_loras() if entry.get("path") == resolved_path or entry.get("name") == display_name),
        {},
    )
    entry = {
        **existing,
        "name": display_name,
        "path": resolved_path,
        "scale": 1.0,
        "base_model": detected_base,
        "triggers": detected_triggers or existing.get("triggers", []),
        "source": existing.get("source", "upload"),
    }
    sync_lora_entry_with_civitai(entry)
    _upsert_lora_entries([entry])
    return entry
