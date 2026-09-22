"""Persistent user-configurable settings for MLX-DIFFUSION.

Settings are stored as plain JSON in ``data/settings.json`` next to the other
runtime state (LoRA registry, downloads, …). Loads lazily, writes atomically,
and rejects unknown/invalid keys so the rest of the app can trust the shape.
"""

import json
import os
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

_lock = threading.RLock()

_SAMPLERS = {
    "euler_trailing",
    "dpmpp_2m_karras",
    "euler_a_substep",
    "euler_a",
    "euler",
    "ddim",
}

DEFAULTS = {
    # Personalization: credit embedded in Civitai/EXIF metadata. Empty -> "MLX-DIFFUSION".
    "artist_name": "",
    # Default output format for new generations ("png" | "jpeg").
    "default_output_format": "png",
    # Default stealth mode (no metadata embedded) for new generations.
    "default_stealth": False,
    # Default fast-VAE (TAEF) toggle.
    "default_fast_vae": True,
    # Optional global sampler fallback (applies to models that support it).
    "default_sampler": "",
    # Default DeepCache interval for SDXL (1 = disabled).
    "default_cache_interval": 1,
    # Per-model overrides: model_id -> {steps, guidance, sampler, cache_interval, fast_vae, width, height}.
    "model_defaults": {},
    # Per-model local install overrides: model_id -> directory that already holds
    # the weights on disk (an HF cache snapshot or a folder downloaded elsewhere).
    # When set, generation loads from this path instead of re-downloading, and
    # "uninstall" only clears the pointer (the user's files are never deleted).
    "model_paths": {},
    # Per-engine custom system-prompt instructions for the prompt enhancer
    # (engine key -> custom "engine guidance" text; "" = use built-in profile).
    "prompt_enhancer": {
        "flux2": "",
        "sdxl": "",
        "krea2": "",
        "z-image-turbo": "",
    },
    # Same, but for the structured JSON output mode (independent override).
    "prompt_enhancer_json": {
        "flux2": "",
        "sdxl": "",
        "krea2": "",
        "z-image-turbo": "",
    },
    # Engine runtime tuning (None = fall back to env var / built-in default).
    # GB wired limits for the Metal allocator; 0 = unbounded (env MLX_WIRED_LIMIT_GB,
    # MLX_KREA_WIRED_LIMIT_GB). Idle kill seconds for the mflux/SDXL watchdogs;
    # 0 = never auto-release. Applied lazily (next generation / idle rearm).
    "memory_wired_limit_gb": None,
    "memory_krea_wired_limit_gb": None,
    "idle_kill_s_mflux": None,
    "idle_kill_s_sdxl": None,
}

# Engine keys used by prompt_enhancer.ENGINE_PROFILES.
PROMPT_ENHANCER_KEYS = {"flux2", "sdxl", "krea2", "z-image-turbo"}
_PROMPT_ENHANCER_MAX = 8000

_MODEL_DEFAULT_KEYS = {
    "steps": lambda v: isinstance(v, int) and 1 <= v <= 50,
    "guidance": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 30.0,
    "sampler": lambda v: isinstance(v, str) and (not v or v in _SAMPLERS),
    "cache_interval": lambda v: isinstance(v, int) and 1 <= v <= 10,
    "fast_vae": lambda v: isinstance(v, bool),
    "width": lambda v: isinstance(v, int) and 256 <= v <= 4096,
    "height": lambda v: isinstance(v, int) and 256 <= v <= 4096,
}

_VALIDATORS = {
    "artist_name": lambda v: isinstance(v, str) and len(v) <= 200,
    "default_output_format": lambda v: v in ("png", "jpeg"),
    "default_stealth": lambda v: isinstance(v, bool),
    "default_fast_vae": lambda v: isinstance(v, bool),
    "default_sampler": lambda v: isinstance(v, str) and (not v or v in _SAMPLERS),
    "default_cache_interval": lambda v: isinstance(v, int) and 1 <= v <= 10,
    "model_defaults": lambda v: isinstance(v, dict),
    "model_paths": lambda v: isinstance(v, dict),
    "prompt_enhancer": lambda v: isinstance(v, dict),
    "prompt_enhancer_json": lambda v: isinstance(v, dict),
    "memory_wired_limit_gb": lambda v: v is None or (isinstance(v, (int, float)) and 0 <= v <= 128),
    "memory_krea_wired_limit_gb": lambda v: v is None or (isinstance(v, (int, float)) and 0 <= v <= 128),
    "idle_kill_s_mflux": lambda v: v is None or (isinstance(v, int) and 0 <= v <= 86400),
    "idle_kill_s_sdxl": lambda v: v is None or (isinstance(v, int) and 0 <= v <= 86400),
}


def _deep_merge(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load() -> dict:
    # Deep copy so callers (update_settings) can mutate nested dicts without
    # corrupting the module-level DEFAULTS template.
    settings = json.loads(json.dumps(DEFAULTS))
    try:
        if SETTINGS_FILE.exists():
            raw = json.loads(SETTINGS_FILE.read_text("utf-8"))
            if isinstance(raw, dict):
                settings = _deep_merge(settings, raw)
    except Exception as e:
        print(f"[settings] failed to load {SETTINGS_FILE}: {e}", flush=True)
    return settings


def _save(settings: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2), "utf-8")
    os.replace(tmp, SETTINGS_FILE)


def get_settings() -> dict:
    """Return a deep copy of the current settings."""
    with _lock:
        settings = _load()
        settings = _deep_merge(DEFAULTS, settings)
        for mid, overrides in list(settings.get("model_defaults", {}).items()):
            if not isinstance(overrides, dict):
                del settings["model_defaults"][mid]
        return json.loads(json.dumps(settings))


def get_setting(key: str, default=None):
    settings = get_settings()
    return settings.get(key, default)


def artist_name() -> str:
    name = (get_settings().get("artist_name") or "").strip()
    return name


def metadata_artist() -> str:
    """Artist credit embedded in image metadata/EXIF. Falls back to a neutral
    product name when the user has not personalized it."""
    return artist_name() or "MLX-DIFFUSION"


def update_settings(updates: dict) -> dict:
    """Validate and apply a partial update. Raises ValueError with the first
    validation error message on invalid input (nothing is persisted)."""
    if not isinstance(updates, dict) or not updates:
        return get_settings()

    with _lock:
        current = _load()
        errors = []

        for key, value in updates.items():
            if key not in _VALIDATORS:
                errors.append(f"unknown setting '{key}'")
                continue
            if not _VALIDATORS[key](value):
                errors.append(f"{key}: invalid value")
                continue
            if key == "model_defaults":
                for model_id, overrides in value.items():
                    if not isinstance(model_id, str) or len(model_id) > 120:
                        errors.append(f"model_defaults: invalid model id '{model_id}'")
                        continue
                    if not isinstance(overrides, dict):
                        errors.append(f"model_defaults.{model_id}: expected object")
                        continue
                    if not overrides:
                        # Empty overrides object -> clear saved defaults for this model.
                        current.setdefault("model_defaults", {}).pop(model_id, None)
                        continue
                    for k, v in overrides.items():
                        if k not in _MODEL_DEFAULT_KEYS:
                            errors.append(f"model_defaults.{model_id}.{k}: unknown key")
                            continue
                        if not _MODEL_DEFAULT_KEYS[k](v):
                            errors.append(f"model_defaults.{model_id}.{k}: invalid value")
                    prev = current.get("model_defaults", {}).get(model_id, {})
                    merged = dict(prev)
                    for k, v in overrides.items():
                        if k in _MODEL_DEFAULT_KEYS and _MODEL_DEFAULT_KEYS[k](v):
                            merged[k] = v
                    current.setdefault("model_defaults", {})[model_id] = merged

            if key in ("prompt_enhancer", "prompt_enhancer_json"):
                for engine_key, text in value.items():
                    if engine_key not in PROMPT_ENHANCER_KEYS:
                        errors.append(f"{key}: unknown engine '{engine_key}'")
                        continue
                    if not isinstance(text, str) or len(text) > _PROMPT_ENHANCER_MAX:
                        errors.append(f"{key}.{engine_key}: invalid value")
                        continue
                    current.setdefault(key, {})[engine_key] = text

            if key == "model_paths":
                for model_id, path in value.items():
                    if not isinstance(model_id, str) or len(model_id) > 120:
                        errors.append(f"model_paths: invalid model id '{model_id}'")
                        continue
                    if not path:
                        # Empty/None -> unlink the local path for this model.
                        current.setdefault("model_paths", {}).pop(model_id, None)
                        continue
                    if not isinstance(path, str) or len(path) > 4096:
                        errors.append(f"model_paths.{model_id}: invalid path")
                        continue
                    current.setdefault("model_paths", {})[model_id] = path

        if errors:
            raise ValueError("; ".join(errors))

        for key, value in updates.items():
            if key not in (
                "model_defaults",
                "model_paths",
                "prompt_enhancer",
                "prompt_enhancer_json",
            ):
                current[key] = value

        _save(current)
        return get_settings()


def clear_model_defaults(model_id: str) -> dict:
    with _lock:
        current = _load()
        current.get("model_defaults", {}).pop(model_id, None)
        _save(current)
        return get_settings()


def prompt_enhancer_override(engine_key: str) -> str:
    """Custom system-prompt instructions for a prompt-enhancer engine, or ""."""
    return (get_settings().get("prompt_enhancer") or {}).get(engine_key, "") or ""


def prompt_enhancer_json_override(engine_key: str) -> str:
    """Custom JSON-mode system-prompt instructions for an engine, or ""."""
    return (get_settings().get("prompt_enhancer_json") or {}).get(engine_key, "") or ""


def model_local_path(model_id: str) -> str:
    """User-registered local weights directory for a model, or ""."""
    return (get_settings().get("model_paths") or {}).get(model_id, "") or ""


# Keys a model descriptor may carry natively vs which come from settings overrides.


def apply_model_defaults(model_id: str, model: dict) -> dict:
    """Merge persisted per-model + global defaults into a model descriptor.

    Returns a copy of ``model`` where ``default_*`` fields are overridden by the
    user's saved preferences (falling back to the model's built-in values)."""
    merged = dict(model)
    per_model = (get_settings().get("model_defaults") or {}).get(model_id, {}) or {}
    merged["has_model_override"] = bool(per_model)

    if per_model.get("steps") is not None:
        merged["default_steps"] = per_model["steps"]
    if per_model.get("guidance") is not None:
        merged["default_guidance"] = per_model["guidance"]
    if per_model.get("sampler"):
        merged["default_sampler"] = per_model["sampler"]
    if per_model.get("cache_interval") is not None:
        merged["default_cache_interval"] = per_model["cache_interval"]
    if per_model.get("fast_vae") is not None:
        merged["default_fast_vae"] = per_model["fast_vae"]
    if per_model.get("width") is not None:
        merged["default_width"] = per_model["width"]
    if per_model.get("height") is not None:
        merged["default_height"] = per_model["height"]

    # Global fallbacks only apply when the model has no explicit override and the
    # feature is supported (sampler list present for SDXL, engine-scoped cache).
    samplers = model.get("samplers") or []
    default_sampler = get_settings().get("default_sampler") or ""
    if "default_sampler" not in merged and default_sampler and default_sampler in samplers:
        merged["default_sampler"] = default_sampler
    if "default_cache_interval" not in merged and model.get("engine") == "sdxl":
        merged["default_cache_interval"] = get_settings().get("default_cache_interval", 1)
    if "default_fast_vae" not in merged:
        merged["default_fast_vae"] = get_settings().get("default_fast_vae", True)
    if "default_output_format" not in merged:
        merged["default_output_format"] = get_settings().get("default_output_format", "png")
    local_path = (get_settings().get("model_paths") or {}).get(model_id) or ""
    if local_path:
        merged["local_path"] = local_path
    return merged