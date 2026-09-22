import os
import re
import time
import tempfile
from pathlib import Path
from typing import Callable, Any
from urllib.parse import urlparse
import requests
from huggingface_hub import HfApi

import civitai_service


DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 MLX-Diffusion/0.1.1"


class HFAuthError(RuntimeError):
    """Raised when Hugging Face requires an authenticated token or agreement for gated/private models."""
    pass


def get_hf_token(token: str | None = None) -> str | None:
    """Resolve Hugging Face API token from argument, environment, local data file, or HF cache."""
    if token and token.strip():
        return token.strip()
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    token_file = Path(__file__).resolve().parent / "data" / "hf_token.txt"
    if token_file.is_file():
        try:
            val = token_file.read_text("utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    # Fallback to default huggingface-cli cached token
    hf_cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache_token.is_file():
        try:
            val = hf_cache_token.read_text("utf-8").strip()
            if val:
                return val
        except Exception:
            pass
    return None


def parse_hf_input(input_str: str) -> dict | None:
    """Parse a user input string into Hugging Face repo_id, filename, and revision.
    
    Supports:
    - https://huggingface.co/<user>/<repo>/resolve/<revision>/<path/to/file.safetensors>[?download=true]
    - https://huggingface.co/<user>/<repo>/blob/<revision>/<path/to/file.safetensors>
    - https://huggingface.co/<user>/<repo>[/tree/<revision>]
    - https://hf.co/<user>/<repo>/...
    - <user>/<repo>
    - <user>/<repo>:<filename>
    - <user>/<repo>/<filename.safetensors>
    """
    cleaned = input_str.strip()
    if not cleaned:
        return None

    # Handle URLs
    if cleaned.startswith(("http://", "https://")):
        parsed = urlparse(cleaned)
        netloc = parsed.netloc.lower()
        if "huggingface.co" not in netloc and "hf.co" not in netloc:
            return None

        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) < 2:
            return None

        user, repo = path_parts[0], path_parts[1]
        repo_id = f"{user}/{repo}"
        revision = "main"
        filename = None

        if len(path_parts) >= 4 and path_parts[2] in ("resolve", "blob", "raw"):
            revision = path_parts[3]
            if len(path_parts) >= 5:
                filename = "/".join(path_parts[4:])
        elif len(path_parts) >= 4 and path_parts[2] == "tree":
            revision = path_parts[3]
            if len(path_parts) >= 5 and path_parts[-1].endswith(".safetensors"):
                filename = "/".join(path_parts[4:])
        elif len(path_parts) == 3 and path_parts[2].endswith(".safetensors"):
            filename = path_parts[2]

        return {
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            "raw_input": cleaned,
        }

    # Handle direct string identifiers: <user>/<repo> or <user>/<repo>:<file> or <user>/<repo>/<file.safetensors>
    # Disqualify if it looks like a Civitai input (all digits) or local path (/...)
    if cleaned.isdigit() or cleaned.startswith(("/", "~", "\\")):
        return None

    if ":" in cleaned:
        repo_part, file_part = cleaned.split(":", 1)
        repo_parts = [p for p in repo_part.strip("/").split("/") if p]
        if len(repo_parts) == 2:
            return {
                "repo_id": f"{repo_parts[0]}/{repo_parts[1]}",
                "filename": file_part.strip(),
                "revision": "main",
                "raw_input": cleaned,
            }

    parts = [p for p in cleaned.strip("/").split("/") if p]
    if len(parts) == 2:
        return {
            "repo_id": f"{parts[0]}/{parts[1]}",
            "filename": None,
            "revision": "main",
            "raw_input": cleaned,
        }
    if len(parts) >= 3 and parts[-1].endswith(".safetensors"):
        return {
            "repo_id": f"{parts[0]}/{parts[1]}",
            "filename": "/".join(parts[2:]),
            "revision": "main",
            "raw_input": cleaned,
        }

    return None


def fetch_hf_metadata(
    repo_id: str,
    filename: str | None = None,
    revision: str = "main",
    token: str | None = None,
) -> dict:
    """Query Hugging Face API to inspect model information, resolve safetensors file, base model, and triggers."""
    resolved_token = get_hf_token(token)
    api = HfApi(token=resolved_token)

    try:
        info = api.model_info(repo_id, revision=revision)
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg or "gated" in err_msg:
            raise HFAuthError(
                f"Hugging Face repository '{repo_id}' requires authentication or acceptance of license terms. "
                f"Please configure your Hugging Face API token in settings."
            ) from e
        if "404" in err_msg or "not found" in err_msg:
            raise ValueError(f"Hugging Face repository '{repo_id}' (revision: {revision}) was not found.") from e
        raise RuntimeError(f"Failed querying Hugging Face repository '{repo_id}': {e}") from e

    # Sibling files
    siblings = [s.rfilename for s in (info.siblings or []) if s.rfilename]
    safetensors_files = [f for f in siblings if f.endswith(".safetensors")]

    resolved_filename = None
    if filename:
        # Match exact or basename
        norm_fn = filename.strip("/")
        match = next((f for f in safetensors_files if f == norm_fn or f.endswith(f"/{norm_fn}")), None)
        if match:
            resolved_filename = match
        else:
            resolved_filename = norm_fn
    else:
        if not safetensors_files:
            raise ValueError(f"No .safetensors files found in Hugging Face repository '{repo_id}'.")
        if len(safetensors_files) == 1:
            resolved_filename = safetensors_files[0]
        else:
            # Pick best candidate: contains 'lora', or matches repo name, or first
            repo_name = repo_id.split("/")[-1].lower()
            best = next((f for f in safetensors_files if "lora" in f.lower() or repo_name in f.lower()), safetensors_files[0])
            resolved_filename = best

    # Base model resolution
    tags = [t.lower() for t in (info.tags or [])]
    base_model_candidate = None
    for t in tags:
        if t.startswith("base_model:"):
            base_model_candidate = t.split(":", 1)[1]
            break

    # Map to internal base_model
    if not base_model_candidate:
        if any("sdxl" in t for t in tags) or "sdxl" in repo_id.lower() or "sdxl" in resolved_filename.lower():
            base_model_candidate = "sdxl"
        elif any("flux" in t for t in tags) or "flux" in repo_id.lower():
            base_model_candidate = "flux2"
        elif "krea" in repo_id.lower():
            base_model_candidate = "krea2"
        elif "z-image" in repo_id.lower() or "zit" in repo_id.lower():
            base_model_candidate = "z-image"

    base_model = civitai_service.normalize_base_model(base_model_candidate)

    # Extract triggers from model card / instance_prompt / widget
    triggers: list[str] = []
    card_data = getattr(info, "card_data", None)
    if card_data:
        inst_prompt = getattr(card_data, "instance_prompt", None)
        if inst_prompt:
            if isinstance(inst_prompt, list):
                triggers.extend([str(p).strip() for p in inst_prompt if str(p).strip()])
            elif isinstance(inst_prompt, str):
                for part in inst_prompt.split(","):
                    if part.strip() and part.strip() not in triggers:
                        triggers.append(part.strip())

        widget = getattr(card_data, "widget", None)
        if isinstance(widget, list):
            for w in widget:
                if isinstance(w, dict) and "text" in w:
                    for part in str(w["text"]).split(","):
                        p_clean = part.strip()
                        if p_clean and len(p_clean) > 1 and p_clean not in triggers:
                            triggers.append(p_clean)

    model_name = repo_id.split("/")[-1]
    raw_download_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{resolved_filename}"

    return {
        "source": "huggingface",
        "repo_id": repo_id,
        "filename": resolved_filename,
        "revision": revision,
        "model_name": model_name,
        "base_model": base_model,
        "triggers": triggers[:5],
        "download_url": raw_download_url,
        "author": repo_id.split("/")[0],
    }


def download_hf_lora(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    revision: str = "main",
    custom_name: str | None = None,
    token: str | None = None,
    progress_cb: Callable | None = None,
    cancel_event: Any | None = None,
) -> tuple[Path, dict]:
    """Stream download a LoRA .safetensors from Hugging Face with progress tracking and validation."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    meta = fetch_hf_metadata(repo_id, filename=filename, revision=revision, token=token)
    resolved_token = get_hf_token(token)

    download_url = meta["download_url"]
    basename = Path(filename).name
    if not basename.endswith(".safetensors"):
        basename += ".safetensors"

    display_name = custom_name or meta.get("model_name") or repo_id.replace("/", "__")
    safe_display = re.sub(r"[^A-Za-z0-9._-]+", "_", display_name).strip("._")[:60]
    safe_fname = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")[:80]
    final_filename = f"{safe_display}__{safe_fname}" if safe_display not in safe_fname else safe_fname
    dest_path = dest_dir / final_filename

    # If already downloaded and valid, return early
    if dest_path.is_file():
        if civitai_service.is_valid_safetensors(dest_path):
            file_size = dest_path.stat().st_size
            civitai_service._emit_progress(progress_cb, "LoRA already present on disk", 1.0, file_size, file_size, 0.0)
            sha = civitai_service.compute_file_sha256(dest_path)
            meta["sha256"] = sha
            meta["path"] = str(dest_path.resolve())
            return dest_path, meta
        else:
            try:
                dest_path.unlink()
            except OSError:
                pass

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    civitai_service._emit_progress(progress_cb, f"Connecting to Hugging Face for {meta['model_name']}...", 0.05, 0, 0, 0.0)

    fd, tmp_file = tempfile.mkstemp(dir=str(dest_dir), suffix=".download")
    try:
        with requests.get(download_url, headers=headers, stream=True, allow_redirects=True, timeout=30) as resp:
            if resp.status_code in (401, 403):
                raise HFAuthError(
                    f"Hugging Face repository '{repo_id}' requires authentication or access permission (HTTP {resp.status_code}). "
                    f"Please configure your Hugging Face API token in settings."
                )
            if resp.status_code == 404:
                raise ValueError(f"File '{filename}' not found in Hugging Face repository '{repo_id}' (HTTP 404).")
            resp.raise_for_status()

            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

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
                        f"Downloading {meta['model_name']}: "
                        f"{downloaded // (1024*1024)}MB / {total_bytes // (1024*1024)}MB "
                        f"({frac*100:.0f}%) - {current_speed_mb:.1f} MB/s"
                    ) if total_bytes > 0 else f"Downloading: {downloaded // (1024*1024)}MB - {current_speed_mb:.1f} MB/s"

                    civitai_service._emit_progress(progress_cb, status_text, frac, downloaded, total_bytes, current_speed_mb)

                out_f.flush()
                os.fsync(out_f.fileno())

        if not civitai_service.is_valid_safetensors(tmp_file):
            raise ValueError(
                f"Downloaded file for '{meta['model_name']}' from Hugging Face is not a valid .safetensors archive. "
                f"The server may have returned an incomplete or non-safetensors payload."
            )

        os.replace(tmp_file, dest_path)
    except HFAuthError:
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
        raise RuntimeError(f"Failed downloading LoRA from Hugging Face: {e}") from e

    civitai_service._emit_progress(progress_cb, "Computing SHA-256 digest...", 0.98, downloaded, total_bytes, 0.0)

    sha = civitai_service.compute_file_sha256(dest_path)
    meta["sha256"] = sha
    meta["path"] = str(dest_path.resolve())

    civitai_service._emit_progress(progress_cb, "LoRA ready!", 1.0, downloaded, total_bytes, 0.0)
    return dest_path, meta
