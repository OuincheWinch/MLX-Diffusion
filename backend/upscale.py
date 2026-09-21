import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from PIL import Image

from image_meta import save_image_with_metadata, extract_image_metadata, _artist_fallback

DATA_DIR = Path(__file__).resolve().parent / "data"
GENERATED_DIR = DATA_DIR / "generated"


def upscale_image(image_id: str, scale: int = 2) -> dict:
    """Upscale image using fast Lanczos resampling with unsharp masking."""
    src_img = GENERATED_DIR / f"{image_id}.png"
    if not src_img.exists():
        src_img = GENERATED_DIR / f"{image_id}.jpeg"
    if not src_img.exists():
        src_img = GENERATED_DIR / f"{image_id}.jpg"
    src_json = GENERATED_DIR / f"{image_id}.json"
    if not src_img.exists():
        raise FileNotFoundError(f"image not found: {image_id}")

    parent_meta = {}
    if src_json.exists():
        try:
            parent_meta = json.loads(src_json.read_text())
        except Exception:
            pass

    scale = max(2, min(4, int(scale)))
    t0 = time.time()
    actual_method = f"lanczos_{scale}x"

    with Image.open(src_img) as img:
        img = img.convert("RGB")
        w, h = img.size
        new_w, new_h = w * scale, h * scale

        # High quality Lanczos resampling
        upscaled = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Unsharp mask for crisp high-frequency edge definition
        from PIL import ImageFilter
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))

    new_id = uuid.uuid4().hex
    fmt = parent_meta.get("format", "png")
    if fmt not in ("png", "jpeg", "jpg"):
        fmt = "jpeg" if src_img.suffix.lower() in (".jpeg", ".jpg") else "png"
    dest_path = GENERATED_DIR / f"{new_id}.{fmt}"
    elapsed = round(time.time() - t0, 2)
    new_w, new_h = upscaled.size

    meta = dict(parent_meta)
    meta.update({
        "id": new_id,
        "width": new_w,
        "height": new_h,
        "upscaled_from": image_id,
        "upscale_factor": scale,
        "upscale_method": actual_method,
        "generation_time": elapsed,
        "created_at": time.time(),
        "software": "MLX-DIFFUSION",
        "generator": "MLX-DIFFUSION",
        "artist": _artist_fallback(parent_meta.get("artist")),
        "file": dest_path.name,
        "format": fmt,
        "tags": list(set((parent_meta.get("tags") or []) + ["upscaled", actual_method])),
    })

    stealth = bool(parent_meta.get("stealth", False))
    save_image_with_metadata(
        image=upscaled,
        dest_path=dest_path,
        meta=meta,
        output_format=fmt,
        stealth=stealth,
    )
    (GENERATED_DIR / f"{new_id}.json").write_text(json.dumps(meta, indent=2))
    try:
        thumbnail_path(new_id)
    except Exception:
        pass
    return meta


def thumbnail_path(image_id: str, force_recreate: bool = False) -> Path:
    thumb = GENERATED_DIR / f"{image_id}_thumb.png"
    src = GENERATED_DIR / f"{image_id}.png"
    if not src.exists():
        src = GENERATED_DIR / f"{image_id}.jpeg"
    if not src.exists():
        src = GENERATED_DIR / f"{image_id}.jpg"

    needs_create = force_recreate or not thumb.exists()

    if needs_create and src.exists():
        meta = None
        json_path = GENERATED_DIR / f"{image_id}.json"
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not meta:
            meta = extract_image_metadata(src)

        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.thumbnail((512, 512))
                stealth = bool(meta.get("stealth", False)) if meta else False
                if meta and not stealth:
                    save_image_with_metadata(
                        image=img,
                        dest_path=thumb,
                        meta=meta,
                        output_format="png",
                        stealth=False,
                    )
                else:
                    fd, tmp = tempfile.mkstemp(dir=str(GENERATED_DIR), suffix=".png.tmp")
                    try:
                        with os.fdopen(fd, "wb") as f:
                            img.save(f, format="PNG")
                        os.replace(tmp, thumb)
                    except Exception:
                        if os.path.exists(tmp):
                            try:
                                os.unlink(tmp)
                            except OSError:
                                pass
                        raise
        except Exception:
            pass
    return thumb
