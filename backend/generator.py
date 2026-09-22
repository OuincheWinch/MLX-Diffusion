import collections
import gc
import json
import os
import select
import subprocess
import threading
import time
import uuid
from pathlib import Path

from collections import OrderedDict
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

from image_meta import (
    save_image_with_metadata,
    extract_image_metadata,
)
from upscale import upscale_image, thumbnail_path

import app_settings

# Anti-SIGABRT Safety Patch for mflux ImageUtil:
# By default, mflux calls np.array(images) on an un-evaluated MLX array.
# If an allocation/Metal failure occurs during eval_impl inside Python's C buffer protocol
# (getbufferproc), the unhandled C++ exception causes std::terminate() -> SIGABRT (Abort trap 6).
# Forcing mx.eval(images) in Python scope before calling np.array(images) ensures any Metal error
# is raised as a standard, catchable Python RuntimeError.
try:
    import mlx.core as mx
    import numpy as np
    import mflux.utils.image_util as _mflux_img_util

    _orig_mflux_to_numpy = _mflux_img_util.ImageUtil._to_numpy

    @staticmethod
    def _safe_mflux_to_numpy(images: mx.array) -> np.ndarray:
        if len(images.shape) == 5:
            images = mx.squeeze(images, axis=2)
        images = mx.transpose(images, (0, 2, 3, 1))
        images = mx.array.astype(images, mx.float32)
        mx.eval(images)
        return np.array(images)

    _mflux_img_util.ImageUtil._to_numpy = _safe_mflux_to_numpy
except Exception as _patch_err:
    print(f"[generator] WARN failed to patch ImageUtil._to_numpy: {_patch_err}", flush=True)

try:
    from mflux.models.krea2 import Krea2
    _orig_krea2_decode = Krea2._decode_latents

    def _safe_krea2_decode(self, *args, **kwargs):
        import mlx.core as _mx
        _mx.clear_cache()
        decoded = _orig_krea2_decode(self, *args, **kwargs)
        _mx.eval(decoded)
        _mx.clear_cache()
        return decoded

    Krea2._decode_latents = _safe_krea2_decode
except Exception:
    pass

DATA_DIR = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = DATA_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "mlx-community/flux2-klein-4b-4bit"
DEFAULT_MODEL_ID = "flux2-klein-4b"

MODELS = {
    "flux2-klein-4b": {
        "id": "flux2-klein-4b",
        "label": "FLUX.2-klein 4B",
        "repo": DEFAULT_MODEL,
        "civitai_model_id": 2387000,
        "civitai_version_id": 2684089,
        "civitai_model_name": "FLUX.2-klein 4B",
        "civitai_version_name": "4-bit",
        "sha256": "D9055F16B1",
        "ecosystem": "FLUX.2",
        "default_steps": 4,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": False,
        "supports_loras": True,
        "supports_multi_reference": True,
        "max_reference_images": 10,
        "supports_fast_vae": True,
        "lora_format": "FLUX.2",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft (512×768)", "width": 512, "height": 768, "steps": 4},
            {"id": "fast", "label": "⚡ Fast (~90s)", "width": 768, "height": 768, "steps": 4},
            {"id": "quality", "label": "✦ Quality (~2.5min)", "width": 1024, "height": 1024, "steps": 4},
        ],
    },
    "flux2-klein-9b": {
        "id": "flux2-klein-9b",
        "label": "FLUX.2-klein 9B",
        "repo": "mlx-community/flux2-klein-9b-4bit",
        "civitai_model_id": 2387000,
        "civitai_version_id": 2684090,
        "civitai_model_name": "FLUX.2-klein 9B",
        "civitai_version_name": "4-bit",
        "ecosystem": "FLUX.2",
        "default_steps": 4,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": False,
        "supports_loras": True,
        "supports_multi_reference": True,
        "max_reference_images": 10,
        "supports_fast_vae": True,
        "lora_format": "FLUX.2",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft (512×768)", "width": 512, "height": 768, "steps": 4},
            {"id": "fast", "label": "⚡ Fast (~110s)", "width": 768, "height": 768, "steps": 4},
            {"id": "quality", "label": "✦ Quality (~3.5min)", "width": 1024, "height": 1024, "steps": 4},
        ],
    },
    "juggernaut-xl-lightning": {
        "id": "juggernaut-xl-lightning",
        "label": "Juggernaut XL Lightning (MLX)",
        "repo": "RunDiffusion/Juggernaut-XL-Lightning",
        "model_dir": DATA_DIR / "models" / "juggernaut-xl-lightning",
        "engine": "sdxl",
        "civitai_version_id": 357609,
        "civitai_model_id": 133005,
        "civitai_model_name": "Juggernaut XL",
        "civitai_version_name": "V9+RDPhoto2-Lightning_4S",
        "sha256": "357609",
        "ecosystem": "SDXL",
        "default_steps": 4,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_fast_vae": True,
        "samplers": ["euler_trailing", "dpmpp_2m_karras", "euler_a_substep", "euler_a", "euler", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": True,
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~5s)", "width": 512, "height": 768, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 2},
            {"id": "photo", "label": "✦ Carré 1024×1024 (~15s)", "width": 1024, "height": 1024, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (~20s)", "width": 832, "height": 1216, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "cinematic", "label": "✦ Cinématique 1216×832 (~20s)", "width": 1216, "height": 832, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "fast", "label": "⚡ Fast 4-step DeepCache (~10s)", "width": 1024, "height": 1024, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 2},
        ],
    },
    "realvis-xl-v5-lightning": {
        "id": "realvis-xl-v5-lightning",
        "label": "RealVisXL V5.0 Lightning (MLX)",
        "repo": "SG161222/RealVisXL_V5.0_Lightning",
        "model_dir": DATA_DIR / "models" / "realvis-xl-v5-lightning",
        "engine": "sdxl",
        "civitai_version_id": 361593,
        "civitai_model_id": 139562,
        "civitai_model_name": "RealVisXL V5.0",
        "civitai_version_name": "V5.0 Lightning",
        "sha256": "B620C6B8D3",
        "ecosystem": "SDXL",
        "default_steps": 6,
        "default_guidance": 1.5,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_fast_vae": True,
        "samplers": ["euler_trailing", "dpmpp_2m_karras", "euler_a_substep", "euler_a", "euler", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": True,
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~7s)", "width": 512, "height": 768, "steps": 6, "sampler": "euler_trailing", "guidance": 1.5, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (~22s)", "width": 1024, "height": 1024, "steps": 6, "sampler": "euler_trailing", "guidance": 1.5, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (~28s)", "width": 832, "height": 1216, "steps": 6, "sampler": "euler_trailing", "guidance": 1.5, "cache_interval": 1},
        ],
    },
    "realvis-xl-v5": {
        "id": "realvis-xl-v5",
        "label": "RealVisXL V5.0 Standard / Hyper-SD",
        "repo": "SG161222/RealVisXL_V5.0",
        "model_dir": DATA_DIR / "models" / "realvis-xl-v5",
        "engine": "sdxl",
        "civitai_version_id": 361592,
        "civitai_model_id": 139562,
        "civitai_model_name": "RealVisXL V5.0",
        "civitai_version_name": "V5.0",
        "sha256": "D7E84FE269",
        "ecosystem": "SDXL",
        "default_steps": 25,
        "default_guidance": 4.5,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_fast_vae": True,
        "samplers": ["dpmpp_2m_karras", "euler_a", "euler", "euler_trailing", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": False,
        "presets": [
            {"id": "draft", "label": "⚡ Hyper-SD 8-step Draft (512×768)", "width": 512, "height": 768, "steps": 8, "sampler": "euler_trailing", "guidance": 2.0, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.5, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (25 steps)", "width": 832, "height": 1216, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.5, "cache_interval": 1},
        ],
    },
    "juggernaut-xi": {
        "id": "juggernaut-xi",
        "label": "Juggernaut XI v11 (MLX)",
        "repo": "RunDiffusion/Juggernaut-XI-v11",
        "model_dir": DATA_DIR / "models" / "juggernaut-xi",
        "engine": "sdxl",
        "civitai_version_id": 782002,
        "civitai_model_id": 133005,
        "civitai_model_name": "Juggernaut XL",
        "civitai_version_name": "XI",
        "sha256": "782002",
        "ecosystem": "SDXL",
        "default_steps": 25,
        "default_guidance": 4.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_fast_vae": True,
        "samplers": ["euler_trailing", "dpmpp_2m_karras", "euler_a_substep", "euler_a", "euler", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": False,
        "presets": [
            {"id": "draft", "label": "⚡ Hyper-SD 8-step Draft (512×768)", "width": 512, "height": 768, "steps": 8, "sampler": "euler_trailing", "guidance": 2.0, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.0, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (25 steps)", "width": 832, "height": 1216, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.0, "cache_interval": 1},
        ],
    },
    "z-image-turbo": {
        "id": "z-image-turbo",
        "label": "Z-Image Turbo 6B",
        "repo": "filipstrand/Z-Image-Turbo-mflux-4bit",
        "civitai_model_id": 2168935,
        "civitai_version_id": 2442439,
        "civitai_model_name": "Z Image Turbo",
        "civitai_version_name": "Turbo",
        "sha256": "2407613050",
        "ecosystem": "ZImageTurbo",
        "default_steps": 8,
        "default_guidance": None,
        "supports_guidance": False,
        "supports_negative": False,
        "supports_loras": True,
        "supports_fast_vae": True,
        "lora_format": "Z-Image",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft (512×768, 6s)", "width": 512, "height": 768, "steps": 6},
            {"id": "turbo", "label": "⚡ Turbo (~35s)", "width": 1024, "height": 1024, "steps": 8},
            {"id": "wide", "label": "✦ Wide HD", "width": 1280, "height": 720, "steps": 8},
        ],
    },
    "krea2-turbo": {
        "id": "krea2-turbo",
        "label": "Krea 2 Turbo 13B",
        "repo": "local:krea2-turbo-q4",
        "civitai_model_id": 2726029,
        "civitai_version_id": 3064584,
        "civitai_model_name": "Krea 2 Turbo Official Comfy-Org Checkpoints (Krea2)",
        "civitai_version_name": "krea2_turbo_bf16",
        "sha256": "78BBF8F416",
        "ecosystem": "Krea 2",
        "default_steps": 8,
        "default_guidance": 1.0,
        "supports_guidance": False,
        "supports_negative": False,
        "supports_loras": True,
        "supports_ref": True,
        "supports_fast_vae": True,
        "lora_format": "Krea 2",
        # 16GB M1 OOM guard: 3D causal VAE decode requires high activation memory.
        # Safe resolution: 512x512 / 512x768. Hard pixel budget: 512x768 (393k px), max side 768.
        "max_pixels": 393216,
        "max_side": 768,
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 4-step (512×768)", "width": 512, "height": 768, "steps": 4},
            {"id": "turbo", "label": "✦ 8-step Quality (~3min)", "width": 512, "height": 512, "steps": 8},
            {"id": "portrait", "label": "✦ Portrait (512×768)", "width": 512, "height": 768, "steps": 8},
            {"id": "fast", "label": "⚡ Fast 4-step (~1.5min)", "width": 512, "height": 512, "steps": 4},
        ],
    },
}


def get_model_info(model_id: str) -> dict:
    if not model_id:
        return MODELS[DEFAULT_MODEL_ID]
    if model_id in MODELS:
        return MODELS[model_id]
    for m in MODELS.values():
        if m.get("repo") == model_id or m.get("id") == model_id:
            return m
    return MODELS[DEFAULT_MODEL_ID]

# Wired memory hint (bytes) used during generation to keep Metal from
# swapping on 16GB machines. Set MLX_WIRED_LIMIT_GB=0 to disable.
# Capped at 45% of the GPU's recommended max working set — a too-high
# wired limit starves the VAE decode allocation and hangs the command.
# krea2 (13B q4) has its own larger budget: see _krea_wired_limit_bytes.
# Metal wired limit: Allow up to 70% of working memory (~11.2GB on 16GB)
# to give FLUX.2 and SDXL adequate room while preventing macOS swap thrashing.
try:
    _WIRED_LIMIT_GB = int(os.environ.get("MLX_WIRED_LIMIT_GB", "7"))
except ValueError:
    print("[generator] invalid MLX_WIRED_LIMIT_GB, using default 7", flush=True)
    _WIRED_LIMIT_GB = 7
_WIRED_LIMIT_GB *= (1 << 30)

# krea2 (13B q4) runs unbounded today; its transformer + TAEF2 decode need a
# larger wired allowance than the generic 45% cap, but still bounded so Metal
# does not page macOS to death. MLX_KREA_WIRED_LIMIT_GB=0 restores the legacy
# unbounded behavior.
try:
    _KREA_WIRED_LIMIT_GB = int(os.environ.get("MLX_KREA_WIRED_LIMIT_GB", "9"))
except ValueError:
    print("[generator] invalid MLX_KREA_WIRED_LIMIT_GB, using default 9", flush=True)
    _KREA_WIRED_LIMIT_GB = 9
_KREA_WIRED_LIMIT_GB *= (1 << 30)


def _env_wired_gb() -> float:
    return _WIRED_LIMIT_GB / (1 << 30) if _WIRED_LIMIT_GB > 0 else 0


def _env_krea_gb() -> float:
    return _KREA_WIRED_LIMIT_GB / (1 << 30) if _KREA_WIRED_LIMIT_GB > 0 else 0


def _wired_limit_gb() -> float:
    """Effective generic wired limit (GB): persisted setting > env var default."""
    v = app_settings.get_setting("memory_wired_limit_gb")
    return float(v) if v is not None else _env_wired_gb()


def _krea_wired_limit_gb() -> float:
    """Effective krea2 wired limit (GB): persisted setting > env var default."""
    v = app_settings.get_setting("memory_krea_wired_limit_gb")
    return float(v) if v is not None else _env_krea_gb()


def _wired_limit_bytes() -> int:
    limit = _wired_limit_gb()
    if limit <= 0:
        return 0
    try:
        import mlx.core as mx

        dev_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
        if dev_info:
            d = dev_info()
            cap = d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size") or 0
            mem = d.get("memory_size", 0)
            if mem > 0:
                # 45% of total unified memory on 16GB (~7.2GB), clamped to recommended working set
                budget = int(mem * 0.45)
                if cap > 0:
                    budget = min(budget, cap)
                return min(int(limit * (1 << 30)), budget)
            if cap > 0:
                return min(int(limit * (1 << 30)), int(cap * 0.45))
    except Exception:
        pass
    return min(int(limit * (1 << 30)), 7 * (1 << 30))


def _krea_wired_limit_bytes() -> int:
    """Wired budget for krea2 (13B q4): 68% of unified memory (vs 45% generic)
    so the transformer + TAEF decode allocations fit without the historical
    starved-decode hang, while still bounded to reduce macOS swap thrash."""
    limit = _krea_wired_limit_gb()
    if limit <= 0:
        return 0
    try:
        import mlx.core as mx

        dev_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
        if dev_info:
            d = dev_info()
            cap = d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size") or 0
            mem = d.get("memory_size", 0)
            if mem > 0:
                budget = int(mem * 0.68)
                if cap > 0:
                    budget = min(budget, cap)
                return min(int(limit * (1 << 30)), budget)
            if cap > 0:
                return min(int(limit * (1 << 30)), int(cap * 0.68))
    except Exception:
        pass
    return min(int(limit * (1 << 30)), 9 * (1 << 30))

_lock = threading.Lock()
_cancel_event = threading.Event()
_pipeline = None
_current_pipeline_key = None
_current_pipeline_model = None
_current_sdxl_model = None
_prompt_cache: OrderedDict[str, tuple] = OrderedDict()
_PROMPT_CACHE_MAX_SIZE = 32

_taef_models: dict = {}


def _get_taef_decoder(variant: str):
    """Lazy loader and in-memory cache for fast TAEF mini-VAEs."""
    global _taef_models
    if variant not in _taef_models:
        if variant == "taef2":
            from mlx_taef import TAEF2
            _taef_models["taef2"] = TAEF2.from_pretrained(include_encoder=False)
        elif variant == "taef1":
            from mlx_taef import TAEF1
            _taef_models["taef1"] = TAEF1.from_pretrained(include_encoder=False)
        elif variant == "krea2":
            from mlx_taef import Krea2
            _taef_models["krea2"] = Krea2.from_pretrained(include_encoder=False)
        else:
            raise ValueError(f"unknown TAEF variant: {variant}")
    return _taef_models[variant]


class GenerationCancelled(Exception):
    pass


def cancel_current():
    """Ask the running generation to abort at the next denoising step."""
    _cancel_event.set()


def model_download_repo(model_id: str, minfo: dict) -> str | None:
    """Resolve the Hugging Face repo whose weights are fetched when (re)installing a model.

    Returns None for models with no remote source (e.g. ``local:`` krea2 checkpoints).
    """
    if minfo.get("engine") == "sdxl":
        # diffusers-format HF repos; snapshot_download(local_dir=<model_dir>) installs them.
        return minfo.get("repo")
    if model_id == "krea2-turbo":
        return None  # local checkpoint bundle, no remote source
    if model_id == "flux2-klein-4b":
        # mflux's Flux2Klein is instantiated WITHOUT model_path, so the weights come from the
        # config default repo (black-forest-labs/FLUX.2-klein-4B), not the MODELS "repo" alias.
        return "black-forest-labs/FLUX.2-klein-4B"
    return minfo.get("repo")


def is_model_cached(model_id: str) -> bool:
    minfo = get_model_info(model_id)
    if not minfo:
        return True
    if resolve_local_model_path(model_id) is not None:
        return True
    if minfo.get("engine") == "sdxl":
        model_dir = minfo.get("model_dir")
        return bool(model_dir and Path(model_dir).exists())
    if model_id == "krea2-turbo":
        return (DATA_DIR / "models" / "krea2-turbo-q4").exists()
    repo = model_download_repo(model_id, minfo) or ""
    if "/" in repo:
        hf_dir_name = f"models--{repo.replace('/', '--')}"
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / hf_dir_name
        snapshots = hf_cache / "snapshots"
        try:
            if snapshots.exists() and any(snapshots.iterdir()):
                return True
        except Exception:
            pass
    return False


def _newest_hf_snapshot(repo_path: Path) -> Path | None:
    """Newest snapshot directory inside an HF cache repo folder, if any."""
    snaps = repo_path / "snapshots"
    if not snaps.is_dir():
        return None
    try:
        candidates = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _resolve_hf_cache_snapshot(path: Path) -> Path:
    """If ``path`` is an HF cache repo root (``models--org--name``), return its
    newest snapshot directory containing weights; otherwise return it unchanged."""
    snaps = path / "snapshots"
    if not snaps.is_dir():
        return path
    try:
        candidates = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return path
    for candidate in candidates:
        try:
            if any(candidate.rglob("*.safetensors")) or any(candidate.rglob("*.index.json")):
                return candidate
        except OSError:
            continue
    return candidates[0] if candidates else path


def resolve_local_model_path(model_id: str, path: Path | str | None = None) -> Path | None:
    """Normalized local weights directory registered for a model, or None.

    HF-cache repo roots are resolved to their snapshot directory so the returned
    path can be handed straight to mflux's ``model_path`` / the SDXL ``model_dir``.
    """
    if path is None:
        raw = app_settings.model_local_path(model_id)
        if not raw:
            return None
        path = Path(raw).expanduser()
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return _resolve_hf_cache_snapshot(p)


def validate_local_model_dir(model_id: str, path: Path | str) -> tuple[bool, str | None]:
    """Sanity-check that ``path`` plausibly holds this model's weights."""
    p = Path(path).expanduser()
    if not p.exists():
        return False, "path does not exist"
    if not p.is_dir():
        return False, "path is not a directory"
    resolved = _resolve_hf_cache_snapshot(p)
    minfo = get_model_info(model_id)
    if minfo.get("engine") == "sdxl":
        if (resolved / "model_index.json").exists() or any(resolved.rglob("*.safetensors")):
            return True, None
        return False, "no diffusers model_index.json or .safetensors found in directory"
    try:
        has_weights = any(resolved.rglob("*.safetensors")) or any(
            resolved.rglob("*.safetensors.index.json")
        )
    except OSError as e:
        return False, f"cannot read directory: {e}"
    if not has_weights:
        return False, "no .safetensors weights found in directory"
    return True, None


def list_local_model_sources(model_id: str | None = None) -> list[dict]:
    """Already-on-disk weight locations for the "install from local" UI.

    Lists every repo in the Hugging Face hub cache with a resolved snapshot path,
    annotating whether it looks like the requested model so the UI can rank it."""
    minfo = get_model_info(model_id) if model_id else None
    expected: set[str] = set()
    if minfo:
        for candidate in (
            model_download_repo(minfo.get("id"), minfo),
            minfo.get("repo"),
            minfo.get("id"),
        ):
            if candidate and "/" in str(candidate) and not str(candidate).startswith("local:"):
                expected.add(str(candidate))
    sources: list[dict] = []
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            snapshot = _newest_hf_snapshot(Path(r.repo_path))
            sources.append(
                {
                    "type": "hf_cache",
                    "repo_id": r.repo_id,
                    "path": str(snapshot or r.repo_path),
                    "size_bytes": int(getattr(r, "size_on_disk", 0) or 0),
                    "matches_model": r.repo_id in expected,
                }
            )
    except Exception as e:
        print(f"[generator] local source scan failed: {e}", flush=True)
    sources.sort(key=lambda s: (not s["matches_model"], -s["size_bytes"]))
    return sources


def is_pipeline_loaded(model_id: str, quantization: int = 4, loras: list | None = None, variant: str = "standard") -> bool:
    """Check if the requested model and weights are already warm & resident in unified memory."""
    minfo = get_model_info(model_id) or {}
    if minfo.get("engine") == "sdxl":
        model_dir = str(
            resolve_local_model_path(model_id)
            or minfo.get("model_dir")
            or (DATA_DIR / "models" / "juggernaut-xl-lightning")
        )
        return (
            _sdxl_daemon is not None
            and _sdxl_daemon.poll() is None
            and _current_sdxl_model == model_dir
        )
    key = _make_pipeline_key(model_id, quantization, loras or [], variant)
    return _pipeline is not None and _current_pipeline_key == key


def _dir_size(path) -> int:
    total = 0
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"


def model_disk_usage(model_id: str, minfo: dict | None = None) -> int:
    """Installed-on-disk size in bytes for a model (SDXL local dir, krea2 bundle,
    or the HF hub cache snapshot for mflux repos)."""
    try:
        minfo = minfo or get_model_info(model_id)
    except Exception:
        return 0
    if minfo:
        local = resolve_local_model_path(model_id)
        if local is not None:
            return _dir_size(local)
        engine = minfo.get("engine")
        if engine == "sdxl":
            d = minfo.get("model_dir")
            if not d:
                return 0
            return _dir_size(d)
        if model_id == "krea2-turbo":
            return _dir_size(DATA_DIR / "models" / "krea2-turbo-q4")
        repo = model_download_repo(model_id, minfo) or ""
        if "/" in repo:
            return _dir_size(_hf_repo_cache_dir(repo))
    return 0


def uninstall_model(model_id: str) -> tuple[bool, str | None]:
    """Remove a model's weights from disk. Refuses (False, reason) while the model
    is queued/generating or a weights download is in-flight for it."""
    minfo = get_model_info(model_id)
    try:
        import state

        with state._JOBS_LOCK:
            for j in state.JOBS.values():
                if j.get("status") not in ("generating", "queued"):
                    continue
                req = j.get("request")
                if req is None:
                    continue
                rid = getattr(req, "model", "") if not isinstance(req, dict) else req.get("model", "")
                if rid == model_id or rid == minfo.get("repo") or get_model_info(rid).get("id") == model_id:
                    return False, f"{model_id} has an active generation job"
        with state._MODEL_DOWNLOAD_LOCK:
            for t in state.MODEL_DOWNLOAD_TASKS.values():
                if t.get("model_id") == model_id and t.get("status") == "downloading":
                    return False, f"{model_id} is currently downloading"
    except Exception as e:
        print(f"[generator] uninstall guard error: {e}", flush=True)

    local_raw = app_settings.model_local_path(model_id)
    unlinked = False
    try:
        removed_any = False
        if local_raw:
            # Pointer install: unlink only; never delete the user's own files.
            app_settings.update_settings({"model_paths": {model_id: ""}})
            removed_any = True
            unlinked = True
        elif minfo.get("engine") == "sdxl":
            d = minfo.get("model_dir")
            if d and Path(d).exists():
                import shutil

                shutil.rmtree(Path(d), ignore_errors=True)
                removed_any = not Path(d).exists()
                if not removed_any:
                    return False, "failed to remove model directory"
        elif model_id == "krea2-turbo":
            import shutil

            d = DATA_DIR / "models" / "krea2-turbo-q4"
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed_any = not d.exists()
                if not removed_any:
                    return False, "failed to remove local model bundle"
        else:
            repo = model_download_repo(model_id, minfo) or ""
            if "/" not in repo:
                return False, "model has no removable weights"
            removed_any = _delete_hf_repo_cache(repo)
    except Exception as e:
        return False, f"uninstall error: {e}"

    with _lock:
        if _current_pipeline_model == model_id:
            _drop_mflux_pipeline()
        if minfo.get("engine") == "sdxl" and _sdxl_daemon is not None and _sdxl_daemon.poll() is None:
            _kill_sdxl_daemon()

    if unlinked:
        return True, "local path unlinked; files left on disk"
    return True, None


def _delete_hf_repo_cache(repo_id: str) -> bool:
    """Remove one repo from ~/.cache/huggingface/hub (snapshots + blobs)."""
    removed_any = False
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for r in info.repos:
            if r.repo_id == repo_id:
                r.delete()
                removed_any = True
                break
    except Exception as e:
        print(f"[generator] hf cache delete failed: {e}", flush=True)
    # belt & braces: drop a half-removed / stale directory (e.g. interrupted downloads)
    stale = _hf_repo_cache_dir(repo_id)
    try:
        if stale.exists():
            import shutil

            shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        pass
    return removed_any or not stale.exists()


def get_engine_status() -> dict:
    """Live snapshot of the Apple Silicon / MLX runtime, resident pipelines and
    disk state — drives the ⚙️ Parameters > Engine & GPU section."""
    status: dict = {
        "meta": {},
        "metal": {},
        "wired": {},
        "mflux": {},
        "sdxl": {},
        "taef": [],
        "storage": {},
    }
    try:
        import platform

        status["meta"] = {
            "engine": "MLX",
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    except Exception:
        pass

    try:
        import mlx.core as mx

        dev_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
        if dev_info:
            d = dev_info()
            status["metal"]["memory_size"] = d.get("memory_size")
            status["metal"]["recommended_max_working_set_size"] = (
                d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size")
            )
            status["metal"]["name"] = d.get("name") or d.get("model")
    except Exception:
        pass

    try:
        import mlx.core as mx

        peak = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
        if peak:
            status["metal"]["peak_memory"] = peak()
        cache = getattr(mx, "get_cache_memory", None) or getattr(mx.metal, "get_cache_memory", None)
        if cache:
            status["metal"]["cache_memory"] = cache()
    except Exception:
        pass

    status["wired"] = {
        "generic_limit_gb": _wired_limit_gb(),
        "generic_budget_bytes": _wired_limit_bytes(),
        "krea_limit_gb": _krea_wired_limit_gb(),
        "krea_budget_bytes": _krea_wired_limit_bytes(),
    }

    now = time.time()
    mflux_idle = _mflux_idle_kill_s()
    status["mflux"] = {
        "resident": _pipeline is not None,
        "model": _current_pipeline_model,
        "idle_kill_s": mflux_idle,
        "watchdog_armed": _mflux_watchdog is not None and _mflux_watchdog.is_alive(),
        "idle_since": _mflux_idle_since,
        "seconds_until_release": (
            max(0, mflux_idle - (now - _mflux_idle_since)) if _mflux_idle_since else None
        ),
        "prompt_cache_size": len(_prompt_cache),
    }
    sdxl_idle = _sdxl_idle_kill_s()
    status["sdxl"] = {
        "resident": _sdxl_daemon is not None and _sdxl_daemon.poll() is None,
        "model": Path(_current_sdxl_model).name if _current_sdxl_model else None,
        "idle_kill_s": sdxl_idle,
        "watchdog_armed": _sdxl_watchdog is not None and _sdxl_watchdog.is_alive(),
        "idle_since": _sdxl_idle_since,
        "seconds_until_release": (
            max(0, sdxl_idle - (now - _sdxl_idle_since)) if _sdxl_idle_since else None
        ),
        "stderr_tail": "".join(_sdxl_stderr_tail)[-1200:],
    }
    status["taef"] = sorted(_taef_models.keys())

    try:
        images = [f for f in GENERATED_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpeg", ".jpg", ".webp", ".heic")]
        status["storage"]["generated_dir"] = str(GENERATED_DIR)
        status["storage"]["image_count"] = len(images)
        status["storage"]["image_bytes"] = sum(f.stat().st_size for f in images)
        status["storage"]["settings_file"] = str(app_settings.SETTINGS_FILE)
        status["storage"]["settings_exists"] = app_settings.SETTINGS_FILE.exists()
    except Exception as e:
        status["storage"]["error"] = str(e)
    return status


class _ProgressCallback:
    def __init__(self, cb, cancel_event=None, phase_cb=None):
        self._cb = cb
        self._cancel_event = cancel_event
        self._phase_cb = phase_cb

    def call_before_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("compiling", "Compiling Metal graph & encoding prompt...")

    def call_in_loop(self, t, **kwargs):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise GenerationCancelled()
        if self._phase_cb:
            self._phase_cb("generating", f"Denoising step {t + 1}...")
        self._cb(t)

    def call_after_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("saving", "Decoding VAE latents & saving image...")


def _make_pipeline_key(model_id: str, quantization: int, loras: list[dict], variant: str = "standard") -> tuple:
    return (
        model_id,
        quantization,
        tuple(sorted(l["path"] for l in loras)),
        variant,
    )


def _update_pipeline_lora_scales(pipe, scales: list[float]):
    if not scales or not hasattr(pipe, "transformer"):
        return
    try:
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear

        for mod in pipe.transformer.modules():
            if isinstance(mod, LoRALinear):
                mod.scale = scales[0]
            elif isinstance(mod, FusedLoRALinear):
                for idx, l in enumerate(mod.loras):
                    if idx < len(scales):
                        l.scale = scales[idx]
        if hasattr(pipe, "lora_scales"):
            pipe.lora_scales = scales
    except Exception as e:
        print(f"[generator] note on dynamic lora scaling: {e}", flush=True)


def _apply_prompt_cache(pipe):
    """Cache text-encoder output keyed by prompt with true LRU eviction.
    Encoding runs a Qwen3 forward pass which is a significant fraction of a short generation."""
    import mlx.core as mx

    orig_encode = pipe._encode_prompt_pair

    def cached_encode(*, prompt, negative_prompt, guidance):
        if prompt in _prompt_cache:
            _prompt_cache.move_to_end(prompt)
            prompt_embeds, text_ids = _prompt_cache[prompt]
        else:
            prompt_embeds, text_ids = orig_encode(
                prompt=prompt,
                negative_prompt=None,
                guidance=1.0,  # encode only the positive prompt here
            )[0:2]
            mx.eval(prompt_embeds, text_ids)
            _prompt_cache[prompt] = (prompt_embeds, text_ids)
            if len(_prompt_cache) > _PROMPT_CACHE_MAX_SIZE:
                _prompt_cache.popitem(last=False)
        negative_embeds, negative_ids = None, None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            neg_key = f"\x00neg:{negative_prompt}"
            if neg_key in _prompt_cache:
                _prompt_cache.move_to_end(neg_key)
                negative_embeds, negative_ids = _prompt_cache[neg_key]
            else:
                negative_embeds, negative_ids = orig_encode(
                    prompt=negative_prompt,
                    negative_prompt=None,
                    guidance=1.0,
                )[0:2]
                mx.eval(negative_embeds, negative_ids)
                _prompt_cache[neg_key] = (negative_embeds, negative_ids)
                if len(_prompt_cache) > _PROMPT_CACHE_MAX_SIZE:
                    _prompt_cache.popitem(last=False)
        return prompt_embeds, text_ids, negative_embeds, negative_ids

    pipe._encode_prompt_pair = cached_encode


def _apply_prompt_cache_txt(pipe):
    """Shared LRU for Z-Image / Krea2 text encoders.

    Both mflux engines expose ``_encode_prompts(prompt, negative_prompt,
    guidance) -> (embeds, neg_embeds)`` and re-encode on every generate_image
    call even though the pipe (with its TextEncoder) is cached in memory.
    prompt/negative/guidance fully determine the embeddings, so memoizing the
    raw arrays (never mutated) is bit-identical. Uses _PROMPT_CACHE_MAX_SIZE;
    tuple keys cannot collide with the string keys used by the FLUX wrapper.
    """
    orig_encode = pipe._encode_prompts

    def cached_encode(*, prompt, negative_prompt=None, guidance=1.0):
        import mlx.core as mx

        key = (prompt, negative_prompt or "", float(guidance))
        if key in _prompt_cache:
            _prompt_cache.move_to_end(key)
            embeds, neg = _prompt_cache[key]
            return embeds, neg
        embeds, neg = orig_encode(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
        mx.eval(embeds)
        if neg is not None:
            mx.eval(neg)
        if len(_prompt_cache) >= _PROMPT_CACHE_MAX_SIZE:
            _prompt_cache.popitem(last=False)
        _prompt_cache[key] = (embeds, neg)
        return embeds, neg

    pipe._encode_prompts = cached_encode


def _install_rope_cache(model_id: str):
    """Memoize each engine's position-embedding cos/sin across denoise steps.

    cos/sin depend only on position ids, which are constant for one generation
    (FLUX.2 img_ids/txt_ids; Z-Image x_pos_ids/cap_pos_ids from patchify;
    Krea2 pos from image/text lengths) but every mflux engine recomputes them on
    each forward. ids are never mutated after creation, so caching per
    (shape, dtype, flat values) is bit-identical while skipping the per-step
    trig work. Class-level patch — mlx dispatches __call__ via the class, so
    instance monkeypatching never fires. Idempotent per rope class.
    """
    rope_classes = {
        "flux2-klein-4b": ("mflux.models.flux2.model.flux2_transformer.pos_embed", "Flux2PosEmbed"),
        "flux2-klein-9b": ("mflux.models.flux2.model.flux2_transformer.pos_embed", "Flux2PosEmbed"),
        "z-image-turbo": ("mflux.models.z_image.model.z_image_transformer.rope_embedder", "RopeEmbedder"),
        "krea2-turbo": ("mflux.models.krea2.model.krea2_transformer.rope_embedder", "Krea2RopeEmbedder"),
    }
    if os.environ.get("MLX_DISABLE_ROPE_CACHE") == "1":
        return
    spec = rope_classes.get(model_id)
    if spec is None:
        return
    import importlib
    try:
        mod = importlib.import_module(spec[0])
        cls = getattr(mod, spec[1])
    except Exception as e:
        print(f"[generator] rope cache unavailable for {model_id}: {e}", flush=True)
        return
    if spec in _ROPE_CACHE_INSTALLED:
        return
    orig = cls.__call__
    cache = {}

    def _call(self, ids):
        key = (ids.shape, str(ids.dtype), tuple(ids.reshape(-1).tolist()))
        hit = cache.get(key)
        if hit is None:
            hit = orig(self, ids)
            if len(cache) > 8:
                cache.clear()
            cache[key] = hit
        return hit

    cls.__call__ = _call
    _ROPE_CACHE_INSTALLED.add(spec)
    print(f"[generator] installed rope memo cache for {model_id} ({spec[1]}, lossless; ids constant across steps)", flush=True)


_ROPE_CACHE_INSTALLED = set()


def _get_pipeline(model_id: str, quantization: int, loras: list[dict], variant: str = "standard", phase_cb=None):
    global _pipeline, _current_pipeline_key, _current_pipeline_model
    key = _make_pipeline_key(model_id, quantization, loras, variant)
    if _pipeline is not None and _current_pipeline_key == key:
        return _pipeline
    info = get_model_info(model_id)
    if phase_cb is not None:
        if is_model_cached(model_id):
            phase_cb("loading_model", f"Loading {info['label']} weights into Apple Silicon unified memory...")
        else:
            phase_cb("downloading", f"Downloading {info['label']} weights from repository...")
    # exclusive residency: kill the SDXL daemon and completely drop previous mflux model & Metal cache
    _kill_sdxl_daemon()
    _drop_mflux_pipeline()
    scales = [l.get("scale", 1.0) for l in loras]
    local_path = resolve_local_model_path(model_id)
    local_arg = str(local_path) if local_path is not None else None
    if model_id == "flux2-klein-4b":
        if variant == "edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

            _pipeline = Flux2KleinEdit(
                model_path=local_arg,
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        else:
            from mflux.models.flux2 import Flux2Klein

            _pipeline = Flux2Klein(
                model_path=local_arg,
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
    elif model_id == "flux2-klein-9b":
        from mflux.models.common.config.model_config import ModelConfig

        if variant == "edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

            _pipeline = Flux2KleinEdit(
                model_config=ModelConfig.flux2_klein_9b(),
                model_path=local_arg or info["repo"],
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        else:
            from mflux.models.flux2 import Flux2Klein

            _pipeline = Flux2Klein(
                model_config=ModelConfig.flux2_klein_9b(),
                model_path=local_arg or info["repo"],
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
    elif model_id == "z-image-turbo":
        from mflux.models.z_image import ZImageTurbo

        _pipeline = ZImageTurbo(
            quantize=quantization,
            model_path=local_arg or info["repo"],
            lora_paths=[l["path"] for l in loras] or None,
            lora_scales=scales or None,
            bake_lora=False,  # runtime fp16 adapters — baking erodes deltas in q4 weights
        )
    elif model_id == "krea2-turbo":
        from mflux.models.krea2 import Krea2
        import mlx.core as mx
        import mlx.nn as nn

        local = DATA_DIR / "models" / "krea2-turbo-q4"
        _pipeline = Krea2(
            quantize=4,
            model_path=local_arg or str(local),
            lora_paths=[l["path"] for l in loras] or None,
            lora_scales=scales or None,
            bake_lora=False,
        )
        # Explicitly quantize the Qwen3-VL text encoder (7.5GB bf16 -> ~1.9GB q4).
        # mflux skips it by default (skip_quantization=True), leaving 7.5GB of raw bf16 weights
        # which causes resident memory to exceed 15GB and trigger Metal CommandBuffer OOM on 16GB Macs.
        if hasattr(_pipeline, "text_encoder") and _pipeline.text_encoder is not None:
            nn.quantize(_pipeline.text_encoder, bits=4, group_size=64)
            mx.eval(_pipeline.text_encoder)
            mx.clear_cache()
            gc.collect()
    else:
        raise ValueError(f"unknown model: {model_id}")
    _current_pipeline_key = key
    _current_pipeline_model = model_id
    _prompt_cache.clear()
    if model_id in ("flux2-klein-4b", "flux2-klein-9b"):
        if os.environ.get("MLX_DISABLE_ROPE_CACHE") != "1":
            _install_rope_cache(model_id)
            _apply_prompt_cache(_pipeline)
    elif model_id in ("z-image-turbo", "krea2-turbo"):
        _install_rope_cache(model_id)
        if os.environ.get("MLX_DISABLE_ROPE_CACHE") != "1":
            _apply_prompt_cache_txt(_pipeline)
    return _pipeline


def _enrich_loras_with_registry(loras: list[dict] | None) -> list[dict]:
    """Enrich LoRA list with civitai_version_id, civitai_model_name, and triggers from loras.json."""
    if not loras:
        return []
    loras_file = DATA_DIR / "loras.json"
    registry = []
    if loras_file.exists():
        try:
            registry = json.loads(loras_file.read_text())
        except Exception:
            registry = []

    enriched = []
    for lora in loras:
        if not isinstance(lora, dict):
            continue
        entry = dict(lora)
        path_str = entry.get("path", "")
        name_str = entry.get("name", "")
        match = next(
            (r for r in registry if (path_str and r.get("path") == path_str) or (name_str and r.get("name") == name_str)),
            None
        )
        if match:
            entry.setdefault("name", match.get("name"))
            if match.get("sha256"):
                entry["sha256"] = match["sha256"]
            if match.get("civitai_version_id"):
                entry["modelVersionId"] = match["civitai_version_id"]
                entry["civitai_version_id"] = match["civitai_version_id"]
            if match.get("civitai_model_id"):
                entry["modelId"] = match["civitai_model_id"]
                entry["civitai_model_id"] = match["civitai_model_id"]
            if match.get("civitai_model_name"):
                entry["civitai_model_name"] = match["civitai_model_name"]
            if match.get("civitai_version_name"):
                entry["civitai_version_name"] = match["civitai_version_name"]
            if match.get("triggers") and not entry.get("triggers"):
                entry["triggers"] = match["triggers"]
        enriched.append(entry)
    return enriched

def generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance: float | None = 1.0,
    seed: int | None = None,
    quantization: int = 4,
    loras: list[dict] | None = None,
    progress_cb=None,
    phase_cb=None,
    model: str = DEFAULT_MODEL_ID,
    cancel_event: threading.Event | None = None,
    negative_prompt: str = "",
    sampler: str | None = None,
    cache_interval: int = 1,
    reference_images: list[str] | None = None,
    image_strength: float | None = None,
    output_format: str = "png",
    stealth: bool = False,
    fast_vae: bool = True,
) -> dict:
    """Blocking generation. Caller must hold no other heavy work."""
    loras = _enrich_loras_with_registry(loras)
    minfo = get_model_info(model)
    if minfo.get("engine") == "sdxl":
        chosen_sampler = sampler or (minfo.get("samplers")[0] if minfo.get("samplers") else "euler_trailing")
        return _generate_sdxl(
            prompt=prompt, width=width, height=height, steps=steps,
            guidance=guidance, seed=seed, loras=loras, progress_cb=progress_cb,
            phase_cb=phase_cb,
            model=model, cancel_event=cancel_event,
            negative_prompt=negative_prompt, sampler=chosen_sampler,
            cache_interval=cache_interval,
            output_format=output_format, stealth=stealth,
            fast_vae=fast_vae)

    ref_paths: list[str] = []
    if reference_images:
        for p in reference_images:
            if p and isinstance(p, str) and p.strip():
                ref_paths.append(p.strip())

    if ref_paths:
        if minfo["id"] not in ("flux2-klein-4b", "flux2-klein-9b", "z-image-turbo", "krea2-turbo"):
            raise ValueError(
                f"Reference images are not supported on {minfo['label']} (FLUX.2-klein / Z-Image / Krea2 only)")
        if len(ref_paths) > 1 and minfo["id"] not in ("flux2-klein-4b", "flux2-klein-9b"):
            raise ValueError(
                f"Multi-reference images (up to 10) are currently supported on FLUX.2-klein only"
            )
        resolved_ref_paths: list[str] = []
        for p in ref_paths:
            cand = Path(p).expanduser()
            if cand.is_absolute() and cand.exists():
                resolved_ref_paths.append(str(cand.resolve()))
            elif (GENERATED_DIR / p).exists():
                resolved_ref_paths.append(str((GENERATED_DIR / p).resolve()))
            elif (GENERATED_DIR / f"{p}.png").exists():
                resolved_ref_paths.append(str((GENERATED_DIR / f"{p}.png").resolve()))
            elif (GENERATED_DIR / f"{p}.jpeg").exists():
                resolved_ref_paths.append(str((GENERATED_DIR / f"{p}.jpeg").resolve()))
            elif (GENERATED_DIR / f"{p}.jpg").exists():
                resolved_ref_paths.append(str((GENERATED_DIR / f"{p}.jpg").resolve()))
            elif (GENERATED_DIR / f"{p}.webp").exists():
                resolved_ref_paths.append(str((GENERATED_DIR / f"{p}.webp").resolve()))
            elif (GENERATED_DIR / f"{p}.heic").exists():
                resolved_ref_paths.append(str((GENERATED_DIR / f"{p}.heic").resolve()))
            elif (GENERATED_DIR / f"{p}.heif").exists():
                resolved_ref_paths.append(str((GENERATED_DIR / f"{p}.heif").resolve()))
            elif (DATA_DIR / "uploads" / p).exists():
                resolved_ref_paths.append(str((DATA_DIR / "uploads" / p).resolve()))
            elif (DATA_DIR / "uploads" / f"{p}.png").exists():
                resolved_ref_paths.append(str((DATA_DIR / "uploads" / f"{p}.png").resolve()))
            elif (DATA_DIR / "uploads" / f"{p}.webp").exists():
                resolved_ref_paths.append(str((DATA_DIR / "uploads" / f"{p}.webp").resolve()))
            elif (DATA_DIR / "uploads" / f"{p}.heic").exists():
                resolved_ref_paths.append(str((DATA_DIR / "uploads" / f"{p}.heic").resolve()))
            elif (DATA_DIR / "uploads" / f"{p}.heif").exists():
                resolved_ref_paths.append(str((DATA_DIR / "uploads" / f"{p}.heif").resolve()))
            elif (ROOT / p).exists():
                resolved_ref_paths.append(str((ROOT / p).resolve()))
            else:
                raise ValueError(f"reference image not found: {p}")

        # If any reference image is raw HEIC/HEIF, normalize to PNG to prevent codec issues in downstream MLX
        normalized_ref_paths: list[str] = []
        for p in resolved_ref_paths:
            p_obj = Path(p)
            if p_obj.suffix.lower() in (".heic", ".heif"):
                uploads_dir = DATA_DIR / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                cached_png = uploads_dir / f"converted_{p_obj.stem}.png"
                if not cached_png.exists() or cached_png.stat().st_mtime < p_obj.stat().st_mtime:
                    try:
                        from PIL import ImageOps
                        with Image.open(p_obj) as img:
                            img.load()
                            trans = ImageOps.exif_transpose(img) or img
                            if trans.mode not in ("RGB", "RGBA"):
                                trans = trans.convert("RGB")
                            trans.save(cached_png, format="PNG")
                    except Exception:
                        try:
                            subprocess.run(
                                ["/usr/bin/sips", "-s", "format", "png", str(p_obj), "--out", str(cached_png)],
                                capture_output=True,
                                timeout=15,
                            )
                        except Exception:
                            pass
                if cached_png.exists():
                    normalized_ref_paths.append(str(cached_png.resolve()))
                else:
                    normalized_ref_paths.append(p)
            else:
                normalized_ref_paths.append(p)
        ref_paths = normalized_ref_paths

    # Strength-based img2img (Z-Image / Krea2) quietly falls back to txt2img in
    # mflux when strength is None (init_time_step = 0). Default to the same 0.6
    # the frontend uses for Z-Image so omitted strength never silently ignores
    # the reference. FLUX.2 edit conditioning ignores strength.
    if ref_paths and image_strength is None:
        image_strength = 0.6

    if loras and not minfo.get("supports_loras"):
        raise ValueError(
            f"LoRAs are not supported on {minfo['label']} (needs {minfo['lora_format']}-format LoRAs)"
        )
    if not minfo["supports_guidance"]:
        guidance = None
    # OOM guard for memory-heavy models (16GB M1): clamp to the model's
    # pixel budget instead of letting Metal thrash / crash the daemon.
    max_side = minfo.get("max_side")
    max_pixels = minfo.get("max_pixels")
    if max_side:
        width = min(width, max_side)
        height = min(height, max_side)
    if max_pixels and width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width = max(256, int(width * scale) // 16 * 16)
        height = max(256, int(height * scale) // 16 * 16)
    with _lock:
        _cancel_event.clear()
        _cancel_mflux_watchdog()
        seed = seed if seed is not None else int(time.time())

        variant = "standard"
        is_flux2 = minfo["id"] in ("flux2-klein-4b", "flux2-klein-9b")
        if is_flux2 and ref_paths:
            variant = "edit"

        actual_loras = list(loras or [])[:16]
        if model == "krea2-turbo" and steps <= 4:
            krea_distill_path = DATA_DIR / "lora_files" / "krea2_turbo_4step_rank_64_lora_latest.safetensors"
            if not krea_distill_path.exists():
                krea_distill_path = DATA_DIR / "lora_files" / "krea2_turbo_4step_rank_64_lora.safetensors"
            if krea_distill_path.exists():
                has_krea_distill = any(
                    "krea2_turbo_4step" in str(l.get("path", "") if isinstance(l, dict) else l)
                    for l in actual_loras
                )
                if not has_krea_distill:
                    actual_loras.append({
                        "path": str(krea_distill_path),
                        "scale": 1.0,
                        "name": "krea2_turbo_4step_rank_64_lora_latest",
                    })
            actual_loras = _enrich_loras_with_registry(actual_loras)

        is_already_loaded = (
            _pipeline is not None
            and _current_pipeline_key == _make_pipeline_key(model, quantization, actual_loras, variant)
        )
        if is_already_loaded and phase_cb is not None:
            phase_cb("preparing", "Preparing prompt conditioning & latents...")

        t_load_start = time.time()
        pipe = _get_pipeline(model, quantization, actual_loras, variant=variant, phase_cb=phase_cb)
        load_time = round(time.time() - t_load_start, 2)
        lora_scales = [
            float(item.get("scale", 1.0)) if isinstance(item, dict) else 1.0
            for item in actual_loras
        ]
        _update_pipeline_lora_scales(pipe, lora_scales)
        if progress_cb is not None:
            pipe.callbacks.register(
                _ProgressCallback(progress_cb, cancel_event=cancel_event or _cancel_event, phase_cb=phase_cb)
            )
        import mlx.core as mx

        prev_wired = None
        restore_callbacks = []
        try:
            if model == "krea2-turbo":
                limit = _krea_wired_limit_bytes()
            else:
                limit = _wired_limit_bytes()
            if limit > 0:
                    try:
                        set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
                        prev_wired = set_limit(limit)
                    except Exception:
                        prev_wired = None

            # Acceleration Hooks: Fast VAE
            if is_flux2:
                if fast_vae and hasattr(pipe, "vae") and hasattr(pipe.vae, "decode_packed_latents"):
                    taef2_dec = _get_taef_decoder("taef2")
                    orig_decode_packed = pipe.vae.decode_packed_latents

                    def _taef2_decode_packed(packed_latents, tiling_config=None, *args, **kwargs):
                        # Ollin's TAEF2 expects raw unpatchified latents without VAE BatchNorm
                        # denormalization (identity BN). unpack_packed_latents multiplies by bn_std
                        # (~1.77) which oversaturates and crushes shadows.
                        if packed_latents.ndim == 5:
                            packed_latents = packed_latents[:, :, 0, :, :]
                        unpacked = pipe.vae._unpatchify_latents(packed_latents)
                        nhwc = mx.transpose(unpacked, (0, 2, 3, 1))
                        decoded_nhwc = taef2_dec.decode(nhwc)
                        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

                    pipe.vae.decode_packed_latents = _taef2_decode_packed
                    restore_callbacks.append(lambda: setattr(pipe.vae, "decode_packed_latents", orig_decode_packed))
                elif not fast_vae and hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
                    if not getattr(pipe.vae.decoder, "_compiled", False):
                        try:
                            pipe.vae.decoder = mx.compile(pipe.vae.decoder)
                            pipe.vae.decoder._compiled = True
                        except Exception:
                            pass

            elif model == "krea2-turbo":
                if fast_vae and hasattr(pipe, "_decode_latents"):
                    krea_dec = _get_taef_decoder("krea2")
                    orig_krea_decode = pipe._decode_latents

                    def _taew_decode_krea(*, latents, prompt, seed, **kwargs):
                        nhwc = mx.transpose(latents, (0, 2, 3, 1))
                        decoded_nhwc = krea_dec.decode(nhwc)
                        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

                    pipe._decode_latents = _taew_decode_krea
                    restore_callbacks.append(lambda: setattr(pipe, "_decode_latents", orig_krea_decode))

            elif model == "z-image-turbo":
                if fast_vae and hasattr(pipe, "_decode_latents"):
                    taef1_dec = _get_taef_decoder("taef1")
                    orig_zit_decode = pipe._decode_latents

                    def _taef1_decode_zit(*, latents, config, prompt, seed, **kwargs):
                        from mflux.models.z_image.latent_creator import ZImageLatentCreator
                        unpacked = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
                        nhwc = mx.transpose(unpacked, (0, 2, 3, 1))
                        decoded_nhwc = taef1_dec.decode(nhwc)
                        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

                    pipe._decode_latents = _taef1_decode_zit
                    restore_callbacks.append(lambda: setattr(pipe, "_decode_latents", orig_zit_decode))

            t_infer_start = time.time()
            if variant == "edit":
                out = pipe.generate_image(
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance if guidance is not None else 1.0,
                    image_paths=[Path(p) for p in ref_paths],
                )
            else:
                gen_kwargs = {
                    "seed": seed,
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "height": height,
                    "width": width,
                    "guidance": guidance if guidance is not None else 1.0,
                    "image_path": ref_paths[0] if ref_paths else None,
                    "image_strength": image_strength,
                }
                out = pipe.generate_image(**gen_kwargs)
            infer_time = round(time.time() - t_infer_start, 2)
        finally:
            for _cb in reversed(restore_callbacks):
                try:
                    _cb()
                except Exception:
                    pass
            if progress_cb is not None:
                pipe.callbacks.in_loop = [
                    c for c in pipe.callbacks.in_loop
                    if not isinstance(c, _ProgressCallback)
                ]
                pipe.callbacks.before_loop = [
                    c for c in getattr(pipe.callbacks, "before_loop", [])
                    if not isinstance(c, _ProgressCallback)
                ]
                pipe.callbacks.after_loop = [
                    c for c in getattr(pipe.callbacks, "after_loop", [])
                    if not isinstance(c, _ProgressCallback)
                ]
            if prev_wired is not None:
                try:
                    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
                    set_limit(prev_wired)
                except Exception:
                    pass
            _arm_mflux_watchdog()
        if phase_cb is not None:
            phase_cb("saving", "Decoding VAE latents & saving image...")
        elapsed = infer_time
        if load_time > 0.5:
            print(f"[{model}] Model load/quantize: {load_time}s | Pure inference: {elapsed}s", flush=True)
        image_id = uuid.uuid4().hex
        fmt = "jpeg" if str(output_format).lower() in ("jpeg", "jpg") else "png"
        image_path = GENERATED_DIR / f"{image_id}.{fmt}"
        meta = {
            "id": image_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "sampler": sampler or "Euler",
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance if guidance is not None else minfo.get("default_guidance", 1.0),
            "seed": seed,
            "quantization": quantization,
            "loras": actual_loras,
            "generation_time": elapsed,
            "load_time": load_time,
            "created_at": time.time(),
            "software": "MLX-DIFFUSION",
            "generator": "MLX-DIFFUSION",
            "artist": app_settings.metadata_artist(),
            "tags": [],
            "file": image_path.name,
            "model": minfo["repo"],
            "reference_images": [Path(p).name for p in ref_paths],
            "stealth": stealth,
            "format": fmt,
            "fast_vae": bool(fast_vae),
        }
        # Strength is meaningful only for strength-based single-ref conditioning
        # (Z-Image / Krea2). FLUX.2 in-context edit ignores strength entirely.
        if ref_paths and variant == "standard":
            meta["reference_strength"] = round(float(image_strength if image_strength is not None else 0.6), 3)
        if minfo.get("civitai_version_id"):
            meta["modelVersionId"] = minfo["civitai_version_id"]
        if minfo.get("civitai_model_id"):
            meta["modelId"] = minfo["civitai_model_id"]
        if minfo.get("civitai_model_name"):
            meta["model_label"] = minfo["civitai_model_name"]
        if minfo.get("civitai_version_name"):
            meta["modelVersionName"] = minfo["civitai_version_name"]

        save_image_with_metadata(
            image=out.image,
            dest_path=image_path,
            meta=meta,
            output_format=fmt,
            stealth=stealth,
        )
        (GENERATED_DIR / f"{image_id}.json").write_text(json.dumps(meta, indent=2))
        gc.collect()
        return meta


_sdxl_daemon = None
_sdxl_watchdog = None
_sdxl_idle_since = None
SDXL_IDLE_KILL_S = 300


def _arm_sdxl_watchdog():
    """Kill the SDXL daemon after `idle_kill_s_sdxl` idle seconds; it is
    respawned lazily on the next generate call. A setting of 0 disables."""
    global _sdxl_watchdog, _sdxl_idle_since
    idle_s = _sdxl_idle_kill_s()
    if idle_s <= 0:
        _cancel_sdxl_watchdog()
        return
    if _sdxl_watchdog is not None:
        _sdxl_watchdog.cancel()
    _sdxl_idle_since = time.time()
    proc = _sdxl_daemon

    def _kill():
        global _sdxl_daemon, _sdxl_watchdog, _current_sdxl_model
        with _lock:
            try:
                if proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                    print("[sdxl] watchdog: idle daemon terminated", flush=True)
            except Exception:
                pass
            if _sdxl_daemon is proc:
                _sdxl_daemon = None
                _current_sdxl_model = None
            _sdxl_watchdog = None
            _sdxl_idle_since = None

    _sdxl_watchdog = threading.Timer(idle_s, _kill)
    _sdxl_watchdog.daemon = True
    _sdxl_watchdog.start()


def _cancel_sdxl_watchdog():
    global _sdxl_watchdog, _sdxl_idle_since
    if _sdxl_watchdog is not None:
        _sdxl_watchdog.cancel()
        _sdxl_watchdog = None
    _sdxl_idle_since = None


_mflux_watchdog = None
_mflux_idle_since = None
MFLUX_IDLE_KILL_S = 300


def _mflux_idle_kill_s() -> int:
    """Effective mflux idle-kill seconds: persisted setting > built-in 300. 0 = never release."""
    v = app_settings.get_setting("idle_kill_s_mflux")
    if v is None:
        return MFLUX_IDLE_KILL_S
    return max(0, int(v))


def _sdxl_idle_kill_s() -> int:
    """Effective SDXL idle-kill seconds: persisted setting > built-in 300. 0 = never release."""
    v = app_settings.get_setting("idle_kill_s_sdxl")
    if v is None:
        return SDXL_IDLE_KILL_S
    return max(0, int(v))


def _arm_mflux_watchdog():
    """Release the resident mflux pipeline after `idle_kill_s_mflux` idle seconds;
    reloaded lazily on the next generate call. A setting of 0 disables the watchdog."""
    global _mflux_watchdog, _mflux_idle_since
    idle_s = _mflux_idle_kill_s()
    if idle_s <= 0:
        _cancel_mflux_watchdog()
        return
    if _mflux_watchdog is not None:
        _mflux_watchdog.cancel()
    _mflux_idle_since = time.time()
    pipe_ref = _pipeline

    def _release():
        global _mflux_watchdog, _mflux_idle_since
        with _lock:
            try:
                if _pipeline is pipe_ref and _pipeline is not None:
                    _drop_mflux_pipeline()
                    print("[mflux] watchdog: idle pipeline released", flush=True)
            except Exception:
                pass
            _mflux_watchdog = None
            _mflux_idle_since = None

    _mflux_watchdog = threading.Timer(idle_s, _release)
    _mflux_watchdog.daemon = True
    _mflux_watchdog.start()


def _cancel_mflux_watchdog():
    global _mflux_watchdog, _mflux_idle_since
    if _mflux_watchdog is not None:
        _mflux_watchdog.cancel()
        _mflux_watchdog = None
    _mflux_idle_since = None


def rearm_engine_watchdogs():
    """Reschedule any armed idle watchdogs with the current persisted settings.

    Safe mid-flight: never touches a running generation or resident pipeline —
    it only re-arms an already-armed watchdog that fires after a generation ends.
    If a generation is currently running the lock is busy, so re-arming is
    skipped; the persisted value still applies automatically on the next idle
    arm (generations always re-arm watchdogs with the current settings).
    """
    global _mflux_watchdog, _mflux_idle_since, _sdxl_watchdog, _sdxl_idle_since
    if not _lock.acquire(blocking=False):
        return
    try:
        if _mflux_watchdog is not None and _mflux_watchdog.is_alive():
            _mflux_watchdog.cancel()
            pipe_ref = _pipeline
            since = _mflux_idle_since
            idle_s = _mflux_idle_kill_s()

            def _release():
                global _mflux_watchdog, _mflux_idle_since
                with _lock:
                    try:
                        if _pipeline is pipe_ref and _pipeline is not None:
                            _drop_mflux_pipeline()
                            print("[mflux] watchdog: idle pipeline released", flush=True)
                    except Exception:
                        pass
                    _mflux_watchdog = None
                    _mflux_idle_since = None

            if idle_s > 0:
                _mflux_watchdog = threading.Timer(idle_s, _release)
                _mflux_watchdog.daemon = True
                _mflux_watchdog.start()
                _mflux_idle_since = since
            else:
                _mflux_watchdog = None
                _mflux_idle_since = None

        if _sdxl_watchdog is not None and _sdxl_watchdog.is_alive():
            _sdxl_watchdog.cancel()
            proc = _sdxl_daemon
            since = _sdxl_idle_since
            idle_s = _sdxl_idle_kill_s()

            def _kill():
                global _sdxl_daemon, _sdxl_watchdog, _current_sdxl_model
                with _lock:
                    try:
                        if proc.poll() is None:
                            proc.kill()
                            try:
                                proc.wait(timeout=5.0)
                            except subprocess.TimeoutExpired:
                                pass
                            print("[sdxl] watchdog: idle daemon terminated", flush=True)
                    except Exception:
                        pass
                    if _sdxl_daemon is proc:
                        _sdxl_daemon = None
                        _current_sdxl_model = None
                    _sdxl_watchdog = None
                    _sdxl_idle_since = None

            if idle_s > 0:
                _sdxl_watchdog = threading.Timer(idle_s, _kill)
                _sdxl_watchdog.daemon = True
                _sdxl_watchdog.start()
                _sdxl_idle_since = since
            else:
                _sdxl_watchdog = None
                _sdxl_idle_since = None
    finally:
        _lock.release()


def _drop_mflux_pipeline():
    """Free the resident mflux pipeline (Krea2/FLUX ~10GB). Needed before an
    SDXL run or between model switches: two resident engines = 20GB+ on a 16GB Mac = thrash."""
    global _pipeline, _current_pipeline_key, _current_pipeline_model, _prompt_cache
    _cancel_mflux_watchdog()
    _pipeline = None
    _current_pipeline_key = None
    _current_pipeline_model = None
    _prompt_cache.clear()
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _kill_sdxl_daemon():
    """Terminate the SDXL engine process so its ~11GB are freed before an
    mflux model loads."""
    global _sdxl_daemon, _current_sdxl_model, _current_pipeline_model
    _cancel_sdxl_watchdog()
    _current_sdxl_model = None
    _current_pipeline_model = None
    if _sdxl_daemon is not None:
        try:
            if _sdxl_daemon.poll() is None:
                _sdxl_daemon.kill()
                try:
                    _sdxl_daemon.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            pass
        _sdxl_daemon = None
        try:
            import mlx.core as mx
            mx.clear_cache()
            gc.collect()
        except Exception:
            pass


_sdxl_stderr_tail = collections.deque(maxlen=100)


def _drain_stderr(proc):
    try:
        for line in iter(proc.stderr.readline, ""):
            _sdxl_stderr_tail.append(line)
    except Exception:
        pass


def _get_sdxl_daemon():
    """Persistent SDXL engine process — model loads once, stays resident."""
    global _sdxl_daemon
    if _sdxl_daemon is not None and _sdxl_daemon.poll() is None:
        return _sdxl_daemon
    # Guarantee a single engine: kill any orphan daemon from a previous
    # backend instance before spawning ours (two resident engines could
    # otherwise run generations in parallel and thrash memory).
    subprocess.run(
        ["pkill", "-f", "sdxl_engine.py --serve"],
        capture_output=True)
    time.sleep(0.5)
    engine = Path(__file__).parent.parent / "venv-sdxl" / "bin" / "python"
    script = Path(__file__).parent / "sdxl_engine.py"
    _sdxl_stderr_tail.clear()
    _sdxl_daemon = subprocess.Popen(
        [str(engine), str(script), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
        cwd=str(Path(__file__).resolve().parent))
    threading.Thread(target=_drain_stderr, args=(_sdxl_daemon,), daemon=True).start()
    return _sdxl_daemon


def _generate_sdxl(prompt, width, height, steps, guidance, seed, loras,
                   progress_cb, model, cancel_event, negative_prompt, sampler,
                   cache_interval=1, output_format="png", stealth=False, fast_vae=True,
                   phase_cb=None):
    """Run F42 SDXL via the venv-sdxl subprocess engine (mlx-diffuser)."""
    with _lock:
        _cancel_sdxl_watchdog()
        try:
            # exclusive residency: drop the mflux pipeline before SDXL loads
            _drop_mflux_pipeline()
            minfo = get_model_info(model) or {}
            sampler = sampler or (minfo.get("samplers")[0] if minfo.get("samplers") else "euler_trailing")
            seed = seed if seed is not None else int(time.time())
            t0 = time.time()
            image_id = uuid.uuid4().hex
            fmt = "jpeg" if str(output_format).lower() in ("jpeg", "jpg") else "png"
            raw_image_path = GENERATED_DIR / f"{image_id}.raw.png"
            final_image_path = GENERATED_DIR / f"{image_id}.{fmt}"
            actual_loras = list(loras or [])
            lightning_path = str(DATA_DIR / "SDXL" / "sdxl_lightning_4step_lora.safetensors")
            # Only inject external Lightning LoRA if the base model is NOT already distilled (e.g. F42 SDXL)
            if not minfo.get("is_distilled"):
                if (steps <= 4 or sampler in ("euler_trailing", "trailing")) and os.path.exists(lightning_path):
                    has_lightning = any(Path(l.get("path", "")).name == "sdxl_lightning_4step_lora.safetensors" for l in actual_loras)
                    if not has_lightning:
                        actual_loras.append({"path": lightning_path, "scale": 1.0})
            tile_vae = (width * height >= 768 * 768)
            model_dir = str(
                resolve_local_model_path(model)
                or minfo.get("model_dir")
                or (DATA_DIR / "models" / "juggernaut-xl-lightning")
            )
            def_guidance = minfo.get("default_guidance")
            if def_guidance is None:
                def_guidance = 1.0 if minfo.get("is_distilled", True) else 2.4
            req = {
                "prompt": prompt, "negative_prompt": negative_prompt or "",
                "height": height - height % 8, "width": width - width % 8,
                "steps": steps, "guidance": guidance if guidance is not None else def_guidance,
                "seed": seed, "sampler": sampler, "image_id": image_id,
                "dest": str(raw_image_path),
                "loras": actual_loras,
                "cache_interval": max(1, int(cache_interval)),
                "tile_vae": tile_vae,
                "model_dir": model_dir,
                "fast_vae": bool(fast_vae),
            }
            global _current_sdxl_model
            is_daemon_alive = (
                _sdxl_daemon is not None
                and _sdxl_daemon.poll() is None
                and _current_sdxl_model == model_dir
            )
            if phase_cb is not None:
                if is_daemon_alive:
                    phase_cb("preparing", "Preparing prompt conditioning & latents...")
                else:
                    phase_cb("loading_model", "Loading SDXL UNet & text encoders into unified memory...")
            _cancel_event.clear()
            proc = _get_sdxl_daemon()
            _current_sdxl_model = model_dir
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            result = None
            while True:
                if (cancel_event is not None and cancel_event.is_set()) or _cancel_event.is_set():
                    # Abort mid-generation: the engine has no interrupt protocol,
                    # so kill it (watchdog/next call respawns it cleanly).
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    _kill_sdxl_daemon()
                    raise GenerationCancelled()
                r, _, _ = select.select([proc.stdout], [], [], 0.2)
                if not r:
                    if proc.poll() is not None:
                        err = "".join(_sdxl_stderr_tail)[-2000:] or "process exited unexpectedly"
                        raise RuntimeError(f"SDXL engine died: {err}")
                    continue
                line = proc.stdout.readline()
                if not line:
                    err = "".join(_sdxl_stderr_tail)[-2000:] or "EOF on stdout"
                    raise RuntimeError(f"SDXL engine died: {err}")
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    if "progress" in payload:
                        if progress_cb is not None:
                            progress_cb(payload["progress"]["step"] - 1)
                    elif "phase" in payload:
                        if phase_cb is not None:
                            phase_cb(payload["phase"], payload.get("detail", ""))
                    else:
                        result = payload
                        break
            if "error" in result:
                _kill_sdxl_daemon()
                raise RuntimeError(result["error"])
            if phase_cb is not None:
                phase_cb("saving", "Finalizing image and metadata...")
            elapsed = result["generation_time"]
            meta = {
                "id": image_id, "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "sampler": sampler,
                "width": width - width % 8, "height": height - height % 8,
                "steps": steps, "guidance": guidance,
                "seed": seed, "quantization": None,
                "loras": loras or [], "generation_time": elapsed,
                "load_time": 0.0,
                "created_at": time.time(),
                "software": "MLX-DIFFUSION",
                "generator": "MLX-DIFFUSION",
"artist": app_settings.metadata_artist(),
                "tags": [],
                "file": final_image_path.name, "model": minfo["repo"],
                "stealth": stealth,
                "format": fmt,
                "fast_vae": bool(fast_vae),
                "cache_interval": max(1, int(cache_interval)),
            }
            if minfo.get("civitai_version_id"):
                meta["modelVersionId"] = minfo["civitai_version_id"]
            if minfo.get("civitai_model_id"):
                meta["modelId"] = minfo["civitai_model_id"]
            if minfo.get("civitai_model_name"):
                meta["model_label"] = minfo["civitai_model_name"]
            if minfo.get("civitai_version_name"):
                meta["modelVersionName"] = minfo["civitai_version_name"]
            actual_w = width - width % 8
            actual_h = height - height % 8
            img_rgb = Image.frombytes("RGB", (actual_w, actual_h), raw_image_path.read_bytes())

            save_image_with_metadata(
                image=img_rgb,
                dest_path=final_image_path,
                meta=meta,
                output_format=fmt,
                stealth=stealth,
            )
            try:
                raw_image_path.unlink()
            except OSError:
                pass
            (GENERATED_DIR / f"{image_id}.json").write_text(json.dumps(meta, indent=2))
            gc.collect()
            return meta
        finally:
            _arm_sdxl_watchdog()

