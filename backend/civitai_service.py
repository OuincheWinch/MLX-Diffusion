import hashlib
import json
import os
import re
import struct
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Any

CIVITAI_API_BASE = "https://civitai.com/api/v1"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 MLX-Diffusion/0.1.1"


class CivitaiAuthError(RuntimeError):
    """Raised when Civitai requires an authenticated account or API key for download."""
    pass


def is_valid_safetensors(path: Path | str) -> bool:
    """Verify that path points to a readable and structurally valid .safetensors file."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        size = p.stat().st_size
        if size < 16:
            return False
        with open(p, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return False
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or header_len > size - 8 or header_len > 100 * 1024 * 1024:
                return False
            header_prefix = f.read(min(header_len, 512))
            if header_prefix.startswith(b"<!doctype") or header_prefix.startswith(b"<html"):
                return False
            if header_len <= 4096:
                f.seek(8)
                header = json.loads(f.read(header_len).decode("utf-8"))
                if not isinstance(header, dict):
                    return False
        return True
    except Exception:
        return False


def get_civitai_api_key(api_key: str | None = None) -> str | None:
    """Resolve Civitai API key from argument, environment, or persistent token file."""
    if api_key and api_key.strip():
        return api_key.strip()
    env_key = os.environ.get("CIVITAI_API_KEY") or os.environ.get("CIVITAI_TOKEN")
    if env_key and env_key.strip():
        return env_key.strip()
    token_file = Path(__file__).resolve().parent / "data" / "civitai_token.txt"
    if token_file.is_file():
        try:
            token = token_file.read_text("utf-8").strip()
            if token:
                return token
        except Exception:
            pass
    return None


def compute_file_sha256(path: Path | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute full SHA-256 hex digest for a model/LoRA file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest().upper()


def _request_json(url: str, timeout: int = 8, api_key: str | None = None) -> dict | None:
    """Send an HTTP GET request to Civitai API with appropriate user agent and optional auth."""
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
    }
    resolved_key = get_civitai_api_key(api_key)
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"[civitai] HTTP {e.code} querying {url}: {e.reason}", flush=True)
    except Exception as e:
        print(f"[civitai] Error querying {url}: {e}", flush=True)
    return None


def fetch_by_hash(sha256: str, api_key: str | None = None) -> dict | None:
    """Query Civitai API to look up model version details by SHA-256 hash."""
    if not sha256 or len(sha256) < 10:
        return None
    url = f"{CIVITAI_API_BASE}/model-versions/by-hash/{sha256.upper()}"
    return _request_json(url, api_key=api_key)


def fetch_by_version_id(version_id: int | str, api_key: str | None = None) -> dict | None:
    """Query Civitai API to look up model version details by modelVersionId."""
    try:
        vid = int(version_id)
    except (ValueError, TypeError):
        return None
    url = f"{CIVITAI_API_BASE}/model-versions/{vid}"
    return _request_json(url, api_key=api_key)


def fetch_by_model_id(model_id: int | str, api_key: str | None = None) -> dict | None:
    """Query Civitai API to look up model details by modelId."""
    try:
        mid = int(model_id)
    except (ValueError, TypeError):
        return None
    url = f"{CIVITAI_API_BASE}/models/{mid}"
    return _request_json(url, api_key=api_key)


def parse_civitai_input(input_str: str, api_key: str | None = None) -> int | None:
    """Extract modelVersionId from Civitai URL or raw numeric ID string, supporting both Model ID and Version ID."""
    cleaned = input_str.strip()
    if not cleaned:
        return None

    # Case 1: URL with modelVersionId query param (e.g. ?modelVersionId=135867)
    m = re.search(r"modelVersionId=(\d+)", cleaned)
    if m:
        return int(m.group(1))

    # Case 2: Direct API download URL (e.g. /api/download/models/135867)
    m = re.search(r"/api/download/models/(\d+)", cleaned)
    if m:
        return int(m.group(1))

    # Case 3: Direct API URL (e.g. /model-versions/135867)
    m = re.search(r"/model-versions/(\d+)", cleaned)
    if m:
        return int(m.group(1))

    # Case 4: Model URL without version query (e.g. civitai.com/models/12345 or /models/12345/slug)
    m = re.search(r"/models/(\d+)", cleaned)
    if m:
        model_id = int(m.group(1))
        model_data = fetch_by_model_id(model_id, api_key=api_key)
        if model_data and "modelVersions" in model_data and len(model_data["modelVersions"]) > 0:
            return model_data["modelVersions"][0].get("id")

    # Case 5: Pure numeric ID — could be Version ID or Model ID
    if cleaned.isdigit():
        num_id = int(cleaned)
        # Test if it is a valid version ID
        if fetch_by_version_id(num_id, api_key=api_key):
            return num_id
        # Fallback: test if it is a model ID and pick its primary version
        model_data = fetch_by_model_id(num_id, api_key=api_key)
        if model_data and "modelVersions" in model_data and len(model_data["modelVersions"]) > 0:
            return model_data["modelVersions"][0].get("id")
        return num_id

    return None


def normalize_base_model(civitai_base: str | None) -> str:
    """Normalize Civitai base model string to internal base_model tag (sdxl, flux2, krea2, z-image).
    Raises ValueError for known incompatible architectures."""
    if not civitai_base:
        return "sdxl"
    b = str(civitai_base).lower().strip()

    # Incompatible architectures: reject early with actionable guidance
    if any(k in b for k in ("sd 1.5", "sd 1.4", "sd 2.0", "sd 2.1", "sd15", "sd14", "sd21")):
        raise ValueError(
            f"Civitai model architecture '{civitai_base}' is SD 1.5/2.x, which is not supported by MLX-DIFFUSION. "
            f"Supported architectures: SDXL (Lightning/Pony/Illustrious), FLUX.2-klein, Krea 2, Z-Image Turbo."
        )
    if any(k in b for k in ("flux.1", "flux1", "flux dev", "flux schnell")):
        raise ValueError(
            f"Civitai model architecture '{civitai_base}' is FLUX.1 (12B), which is incompatible with FLUX.2-klein (4B)."
        )
    if any(k in b for k in ("cascade", "pixart", "auraflow", "hunyuan")):
        raise ValueError(
            f"Civitai model architecture '{civitai_base}' is not supported by MLX-DIFFUSION."
        )

    # Supported architectures
    if any(k in b for k in ("sdxl", "pony", "illustrious", "sd xl")):
        return "sdxl"
    if any(k in b for k in ("flux.2", "flux2", "klein")):
        return "flux2"
    if "krea" in b:
        return "krea2"
    if any(k in b for k in ("z-image", "zit", "zimage")):
        return "z-image"

    return "sdxl"


def extract_civitai_metadata(version_info: dict) -> dict:
    """Convert Civitai API model-version JSON into standardized internal metadata."""
    model_obj = version_info.get("model", {})
    model_type = str(model_obj.get("type") or version_info.get("type") or "LORA").upper()

    if model_type not in ("LORA", "LOCON", "DORA"):
        if model_type in ("CHECKPOINT", "TEXTUALINVERSION", "CONTROLNET", "HYPERNETWORK", "UPSCALER"):
            raise ValueError(
                f"The Civitai resource is a '{model_type}', not a LoRA. "
                f"Only LoRA, LoCon, or DoRA models can be imported here."
            )

    trained_words = version_info.get("trainedWords") or []
    files = version_info.get("files") or []

    # Find the primary safetensors file
    safetensor_file = next(
        (f for f in files if str(f.get("name", "")).endswith(".safetensors")),
        files[0] if files else {},
    )

    base_model = normalize_base_model(version_info.get("baseModel"))

    return {
        "civitai_version_id": version_info.get("id"),
        "civitai_model_id": version_info.get("modelId"),
        "civitai_model_name": model_obj.get("name") or version_info.get("name", "Civitai-LoRA"),
        "civitai_version_name": version_info.get("name", "v1.0"),
        "base_model": base_model,
        "triggers": [w.strip() for w in trained_words if w and w.strip()],
        "download_url": safetensor_file.get("downloadUrl"),
        "filename": safetensor_file.get("name"),
        "size_kb": safetensor_file.get("sizeKB"),
    }


def _emit_progress(progress_cb: Callable | None, status_text: str, frac: float, downloaded: int, total: int, speed: float):
    if not progress_cb:
        return
    payload = {
        "status": status_text,
        "progress": frac,
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "speed_mb_s": round(speed, 2),
    }
    try:
        progress_cb(payload)
    except TypeError:
        try:
            progress_cb(status_text, frac)
        except Exception:
            pass


def download_civitai_lora(
    version_id: int,
    dest_dir: Path,
    custom_name: str | None = None,
    progress_cb: Callable | None = None,
    api_key: str | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, dict]:
    """Download a LoRA .safetensors from Civitai, compute hash, and return (filepath, metadata)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    version_info = fetch_by_version_id(version_id, api_key=api_key)
    if not version_info:
        raise ValueError(f"Could not find model version {version_id} on Civitai")

    meta = extract_civitai_metadata(version_info)
    resolved_key = get_civitai_api_key(api_key)
    download_url = meta.get("download_url")
    if not download_url:
        download_url = f"https://civitai.com/api/download/models/{version_id}"

    if resolved_key:
        delimiter = "&" if "?" in download_url else "?"
        download_url = f"{download_url}{delimiter}token={resolved_key}"

    raw_filename = meta.get("filename") or f"civitai_lora_{version_id}.safetensors"
    if not raw_filename.endswith(".safetensors"):
        raw_filename += ".safetensors"

    display_name = custom_name or meta.get("civitai_model_name") or f"lora_{version_id}"
    safe_display = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name).strip("._")[:60]
    safe_fname = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_filename).strip("._")[:80]
    final_filename = f"{safe_display}__{safe_fname}" if safe_display not in safe_fname else safe_fname
    dest_path = dest_dir / final_filename

    # If already downloaded, verify file integrity
    if dest_path.is_file():
        if is_valid_safetensors(dest_path):
            file_size = dest_path.stat().st_size
            _emit_progress(progress_cb, "LoRA already present on disk", 1.0, file_size, file_size, 0.0)
            sha = compute_file_sha256(dest_path)
            meta["sha256"] = sha
            meta["path"] = str(dest_path.resolve())
            return dest_path, meta
        else:
            print(f"[civitai] Removing corrupted cached LoRA file: {dest_path}", flush=True)
            try:
                dest_path.unlink()
            except OSError:
                pass

    # Download with progress
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    req = urllib.request.Request(
        download_url,
        headers=headers,
    )

    _emit_progress(progress_cb, f"Connecting to Civitai for {meta['civitai_model_name']}...", 0.05, 0, 0, 0.0)

    fd, tmp_file = tempfile.mkstemp(dir=str(dest_dir), suffix=".download")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, os.fdopen(fd, "wb") as out_f:
            final_url = str(resp.url).lower()
            content_type = str(resp.headers.get("Content-Type", "")).lower()
            if "auth.civitai.com" in final_url or "reason=download-auth" in final_url or content_type.startswith("text/html"):
                raise CivitaiAuthError(
                    f"Civitai model '{meta['civitai_model_name']}' requires authentication (download-auth). "
                    f"Please configure your Civitai API Key in settings or download the .safetensors manually."
                )

            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            start_time = time.time()
            last_speed_time = start_time
            last_speed_bytes = 0
            current_speed_mb = 0.0

            while True:
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Download cancelled by user")

                chunk = resp.read(chunk_size)
                if not chunk:
                    break
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
                    f"Downloading {meta['civitai_model_name']}: "
                    f"{downloaded // (1024*1024)}MB / {total_bytes // (1024*1024)}MB "
                    f"({frac*100:.0f}%) - {current_speed_mb:.1f} MB/s"
                ) if total_bytes > 0 else f"Downloading: {downloaded // (1024*1024)}MB - {current_speed_mb:.1f} MB/s"

                _emit_progress(progress_cb, status_text, frac, downloaded, total_bytes, current_speed_mb)

            out_f.flush()
            os.fsync(out_f.fileno())

        if not is_valid_safetensors(tmp_file):
            raise ValueError(
                f"Downloaded file for '{meta['civitai_model_name']}' is not a valid .safetensors archive. "
                f"Civitai may have returned an incomplete or erroneous payload."
            )

        os.replace(tmp_file, dest_path)
    except CivitaiAuthError:
        if os.path.exists(tmp_file):
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
        raise
    except Exception as e:
        if os.path.exists(tmp_file):
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
        raise RuntimeError(f"Failed downloading LoRA from Civitai: {e}") from e

    _emit_progress(progress_cb, "Computing SHA-256 digest...", 0.98, downloaded, total_bytes, 0.0)

    sha = compute_file_sha256(dest_path)
    meta["sha256"] = sha
    meta["path"] = str(dest_path.resolve())

    _emit_progress(progress_cb, "LoRA ready!", 1.0, downloaded, total_bytes, 0.0)

    return dest_path, meta
