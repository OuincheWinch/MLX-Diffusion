# AGENTS.md

## Overview
MLX-DIFFUSION: local image-generation playground for FLUX.2-klein 4B on Apple Silicon via mflux (MLX). FastAPI backend + React/Vite SPA.

- `backend/` — FastAPI (`main.py` entrypoint); generation logic in `generator.py`. Images + JSON sidecars live in `data/generated/`; LoRA registry in `data/loras.json`.
- `frontend/` — React 19 + Vite SPA (`src/App.jsx`, components in `src/components/`). Hardcodes `API_BASE = http://localhost:8001`.

## Commands

Backend (use repo venv — Python 3.10 at `venv/`; dependencies pinned in `backend/requirements.txt`; SDXL in `backend/requirements-sdxl.txt`):
```bash
./venv/bin/uvicorn main:app --reload --port 8001   # run from backend/ (or use ./run.sh from root)
```

Frontend:
```bash
cd frontend && npm run dev    # vite dev server
npm run lint                  # oxlint
npm run build                 # production build
```

## SDXL (Juggernaut XL Lightning) — second engine
- Juggernaut XL Lightning runs via `venv-sdxl/` (Python 3.14, mlx-diffuser from GitHub main — PyPI has NO SDXL).
- `backend/sdxl_engine.py` is invoked as subprocess by `generator._generate_sdxl`. Model converted to diffusers format at `backend/data/models/juggernaut-xl-lightning/`.
- Native 4-step distilled model: euler_trailing sampler, guidance 1.0, fast generation (~15-20s @ 1024x1024).
- Multi-LoRA: kohya/civitai keys mapped to diffusers names at load time; adapters rank-concat stacked. Triggers live in `data/loras.json` (`triggers`, `base_model` fields).
- Samplers: euler_trailing (default for 4-step lightning), dpmpp_2m_karras, euler_a_substep, euler_a, euler, ddim.
- torch/diffusers installed in venv-sdxl for conversion only; engine runtime is torch-free.

## Gotchas
- When the user asks for an improvement/optimization, do NOT just patch the obvious spot: actively research alternative approaches (different frameworks, engines, algorithms, architectures — e.g. CoreML vs MLX, driving external tools' APIs), benchmark the options when cheap, and PROPOSE them to the user before or alongside implementing.
- Apple Silicon only (MLX). Backend must be on port **8001** (8000 is Align-n-GIF's).
- First generation downloads model weights (~2–3 GB) to the HF cache; takes ~20 min.
- Model pipeline is cached in memory keyed by (quantization, LoRA paths); changing either reloads weights. Resident mflux pipeline (FLUX/Krea2 ~10GB) is released by a watchdog `MFLUX_IDLE_KILL_S=300` in generator.py 5 min after the last generation ends (same idle policy as the SDXL daemon); next mflux generation after eviction pays a cold reload (~20-40s).
- One generation at a time — worker thread + lock in `generator.py`/`main.py`.
- FLUX.2-klein 4B is guidance-distilled: **no negative prompt** support (mflux API has no such parameter).
- LoRAs must match FLUX.2 architecture; local paths are validated, HF repo ids download on first use.
- Thumbnails (`{id}_thumb.png`) are generated lazily by `/api/images/{id}/file?thumb=true`.
- krea2 wired-memory pin: `MLX_KREA_WIRED_LIMIT_GB` (default 9) — was unbounded for krea2 (previously excluded because a too-low limit starved VAE decode and hung); re-enabled via `_krea_wired_limit_bytes()` (68% fraction vs generic 45%). Bench 512×768/4-step warm: wired 155s vs unbounded 183s (~18% faster, identical output). Set env to 0 to restore legacy unbounded.
- SDXL perf notes (M1 16GB): UNet must run quantize_unet=4 — ~5x FASTER than fp16 (fused quant matmul) and halves memory; 8-bit is 7x SLOWER (dequant per matmul) — never enable. DeepCache cache_interval=2 available via request param (~1.5x claimed, quality tradeoff, default off). `release_text_encoders` must stay False in the daemon (library permanently drops the CLIP encoders — no lazy reload). Persistent daemon kills itself after 5 min idle (watchdog in generator.py), respawned lazily. LoRA set is cached; changing it reloads the pipeline clean (~20-30s). Samplers: euler_a_substep (DrawThings-style substepped ancestral, best quality/steps), dpmpp_2m_karras, euler_a, euler, ddim. Engine wired-limit 6GB during denoise (prevents Metal paging stalls). Bench: 42-step substep @512x768 q4 ≈ 353s (~4.2s/UNet call); DrawThings does ~0.65s/call (compiled CoreML kernels — gap is framework-level, not fixable in MLX). Check system load before benchmarking — load>5 doubles gen times.
