# QWENROUTE — Qwen-Image-2.1 Integration Plan (Q4, MLX/mflux)

Status: **PLAN** (approved by user on 2026-09-21) — implementation not started.
Hardware: M1 16GB · Backend `uvicorn main:app --port 8001` · Engine: mflux via `venv/` (py3.10)
Goal: run **Qwen-Image-2.1 at Q4** in the MLX-DIFFUSION app, done ONCE the q4 weights are built locally.

---

## 1. Research summary

### 1.1 Model: Qwen-Image-2.1 (2026-09-20)
- Unified text-to-image + image-editing model. License **`qwen-research` (non-commercial)**.
- HF repo `Qwen/Qwen-Image-2.1` (~33 GB bf16 on disk).
- Architecture (differs from Qwen-Image-2512 — the engine our installed `qwen` module targets):
  - Transformer: **7.1B visual**, 32 x single-stream DiT layers, 32 heads / 128 head-dim,
    `in/out_channels=64`, `patch_size=1`, `context_in_dim=4096`, `mlp_ratio=3`,
    `axes_dims_rope=[16,56,56]`, **block-causal** (`causal_condition`, text cond at t=0).
  - VAE: new **64-channel causal RGBA VAE** (`AutoencoderKLQwenImage21`), 16x spatial compression,
    1 latent token per 16x16 tile; alpha -> edit mask, decode returns RGB only.
  - Text encoder: **Qwen3-VL 8B** (`Qwen3VLForConditionalGeneration`, 36 layers, hidden 4096,
    32 heads / 8 KV heads, vocab 151936, interleaved mrope).
- Sample defaults: **40 steps, guidance 1.0** (guidance-distilled). Real CFG = negative prompt + guidance>1
  (2nd denoise pass). Native up to 2K/2048² / RGBA / up to 10 ref images (edit variant only).

### 1.2 Upstream mflux status
- Released 0.19.2 (2026-09-17): **zero** `qwen21` support (verified in wheel).
- **mflux git-main merged it yesterday**: `feat: Qwen-Image-2.1 support (txt2img, img2img, quantization) (#736)`,
  in `mflux.models.qwen21`:
  - `QwenImage21`, `Qwen21Initializer`, `Qwen21Transformer`, `Qwen21VAE`, `Qwen21TextEncoder`,
    `Qwen21LatentCreator`, `Qwen21WeightDefinition`/`Qwen21WeightMapping`.
  - Config `qwen-image-2.1` (priority 29, aliases `qwen-2.1`/`qwen-image-21`, model `Qwen/Qwen-Image-2.1`,
    `supports_guidance=True`). API: `QwenImage21(quantize=4, model_config=ModelConfig.qwen_image_21())`.
  - q4 works for transformer + VAE; **text encoder is `skip_quantization=True` (stays bf16)**.
  - **Not yet supported**: edit/instruction variant, LoRA mappings, PID decode.
  - Deps: darwin `mlx>=0.32.0,<0.33.0` — our 0.32.1 OK.
- `mlx-diffuser` 0.1.6 (SDXL venv): no qwen → ruled out. `mlx-community/Qwen-Image-2.1-MLX-4bit` (~10.5GB)
  is NOT mflux-native format → reference only.

### 1.3 Quantize-once strategy (the core of this plan)
- `nn.quantize()` is fast (seconds); the cost is I/O of ~33 GB bf16. Do it once, save to disk.
- mflux native-save writes `quantization_level` + `mflux_version` metadata per shard
  (`ModelSaver._save_weights`). On reload, `WeightLoader._try_load_mflux_format` detects `stored_q` and
  `WeightApplier` takes the **stored-predicate path** (reads already-quantized tensors, never re-quantizes).
- Proven by our existing `backend/data/models/krea2-turbo-q4/` (~15 GB) — cold load ~81 s via this path.
- **Text-encoder nuance**: krea2's saved copy stores the encoder **bf16** (mflux default), so
  generator.py:710-714 re-quantizes it at every load. For 2.1 we will **quantize the encoder BEFORE saving**
  so the on-disk copy is q4; target reload reads ~9 GB total (fast path). Fallback if the q4-encoder reload
  misbehaves: krea2-style runtime `nn.quantize(_pipeline.text_encoder, bits=4, group_size=64)`

### 1.4 Memory budget (M1 16GB, q4)
| Component          | bf16 disk | q4 resident (est.) |
|--------------------|-----------|--------------------|
| Transformer (7.1B) | 14.2 GB   | ~3.6–4 GB          |
| Text encoder (Qwen3-VL 8B) | 17.5 GB | ~4.4 GB (manual q4) |
| VAE                | 1.4 GB    | ~0.4 GB            |
| **Total**          | ~33 GB    | **~8.5 GB**        |
- + activations + Metal wired limit → cap resolution like krea2: **≤512×768 / ≤393k px, max side 768**.
- First build keeps 17.5 GB bf16 encoder until q4 → one heavy swap spike (~2–3 min load). Afterwards ~2–3 s/step
  ≈ **2–3 min per 40-step image**.

---

## 2. Option A (SELECTED) — Build local q4 model via git-main, keep pinnable runtime

`./venv/bin/pip install --force-reinstall "git+https://github.com/mflux-community/mflux.git"`

- Rationale: zero hand-porting; qwen21 rides upstream; future edit/LoRA support arrives via reinstall.
- mlx stays 0.32.1 (satisfies main's range).
- Update `backend/requirements.txt` note (mflux now dev-installed from git).
- Regression gate after install: re-smoke krea2 / flux2-klein / z-image / sdxl.

### Step A1 — Build script (one-time)
`backend/scripts/build_qwen21_q4.py` (run with `./venv/bin/python`):
1. `pipeline = QwenImage21(quantize=4, model_config=ModelConfig.qwen_image_21())`
   → streams ~33 GB from HF (first time) and quantizes transformer+VAE in-memory.
2. Manual `nn.quantize(pipeline.text_encoder, bits=4, group_size=64)` + `mx.eval` + `mx.clear_cache()` + `gc.collect()`.
3. `ModelSaver.save_model(pipeline, bits=4, base_path=<DATA/model config dir>, weight_definition=...)`
   → writes `backend/data/models/qwen-image-2.1-q4/` (transformer+vae q4, encoder q4 if save accepts it).
4. Save as mflux-native; log total size; last step completed here = "done once".

### Step A2 — Backend: `backend/generator.py`
- `MODELS["qwen-image-2.1"]` entry:
  - label `Qwen Image 2.1`, `repo: "Qwen/Qwen-Image-2.1"`, `ecosystem: "Qwen"`, `default_steps: 40`,
    `default_guidance: 1.0`, `supports_guidance: True`, `supports_negative: True`, `supports_loras: False`,
    `supports_fast_vae: False`, `max_pixels: 393216`, `max_side: 768`,
    presets: Fast Draft 512×768 / Quality 512×512 / Portrait 512×768 (all 40 steps).
- `_get_pipeline` branch: load from local q4 dir via `QwenImage21(quantize=4, model_path=<local q4 dir>)`,
  then optional runtime encoder-q4 fallback (avoids re-quantize each load).
- Add wired-limit budget (like krea2's ~9 GB) to stop Metal from swapping.
- **No** ROPE cache / prompt-cache / TAEF hooks for it (hooks are keyed to flux2/z-image/krea).

### Step A3 — Frontend
- No source change needed: `/api/models` flags drive `GenerateForm` (quantization already `4`,
  guidance/negative toggles appear from `supports_*`).

### Step A4 — Verification checklist
1. `./venv/bin/python -m compileall backend/` passes.
2. Cold restart `uvicorn main:app --port 8001`.
3. First generate: `qwen-image-2.1`, 512×512, 40 steps, fixed seed → no OOM (expect long first load/download).
4. Re-run krea2 + flux2-klein to confirm no regression after git-main install.
5. Confirm wired-limit applies (no `Metal CommandBuffer` hang on the 64-ch VAE decode).
6. Confirm metadata sidecar + EXIF + Civitai-style params embed correctly (`ecosystem: Qwen`).

---

## 3. Option B (backup) — Vendored backport into installed 0.19.2
Kept here for reference; only if Option A fails the regression gate.
- Copy `mflux/models/qwen21/` (~15 files) into site-packages; add `ModelConfig.qwen_image_21()` + entry
  to installed `model_config.py`; backport any shared main-only deps (verify qwen3_vl shared module exists);
  document patch set + upstream SHA (#736). Rollback = delete dir + revert config hunks.

---

## 4. Risks / open questions
- 17.5 GB bf16 encoder spike during first build on a swap-bound box (temporary heavy disk I/O).
- qwen-research license: non-commercial — OK for private playground; flag if it goes public.
- q4 quality on 64-ch block-causal transformer unverified here; fallback = transformer `-q 8` (still q4 encoder).
- LoRA / edit / 10-ref images unsupported upstream — separate task (needs Qwen3-VL vision tower port).
- Decision deferred: verify q4-encoder reload path in Step A1; if it fails, use runtime encoder-q4 fallback.

---

## 5. Timeline estimate
- Research: done. · Build script + backend changes: ~1 session. · Git-main install + regression + smoke: ~1 session.
- First qwen21 gen downloads ~33 GB — scale small on purpose.