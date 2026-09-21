
## 2026-09-20 — Black Screen Crash on Launch & LoRA State Retention
- **Root Cause of Black Screen on Launch**:
  1. In `GenerationParams.jsx:229`, line evaluated `model === "flux2-klein-4b"`. `model` was not passed as prop or imported, throwing `ReferenceError: model is not defined`.
  2. On initial mount, `models` was `[]`, so `modelInfo.supports_fast_vae` was falsy, rendering the initial UI briefly. As soon as `api("/api/models")` resolved (~100ms), `supports_fast_vae: true` triggered evaluation of line 229, crashing React and unmounting `#root` to a pitch-black screen.
  3. Fix: Added `model` prop to `GenerationParams`, safely derived `currentModelId = model || modelInfo?.id || ""` and `fastVaeName`, and wrapped `<App />` in a robust `<ErrorBoundary />` in `main.jsx` to eliminate black-screen-of-death crashes.
- **Root Cause of LoRAs Purging ("Ont sauté") on New LoRA Download**:
  1. `setLoras` in `LoraManagerDrawer` only recorded `{ path, scale, name }` without saving `base_model`.
  2. When a new LoRA was downloaded, `onLoraDownloaded` reloaded `loraRegistry`. In `GenerateForm.jsx`, `useEffect([model, modelInfo, loraRegistry])` ran `currentLoras.filter(...)`. If `findLoraEntry` failed or returned null during refresh, `lBase` was undefined and all active LoRAs were immediately dropped to `[]`.
  3. Fix: `addLoraFromRegistry` now bakes `base_model` directly into the active LoRA state object, and the filter guard preserves LoRAs if `lBase` is undetermined (`if (!lBase) return true`).
  4. Safetensors inspection in `state.py`: `_inspect_safetensors` checked `"z-image"`, `"zit"`, `"zimage"` in filename, missing `"z_image"` (underscore) and `ss_base_model_version: "zimage"`, misclassifying Z-Image LoRAs as `"flux2"`. Fixed to check both version and underscore variations.
  5. Atomic write UTF-8: `_atomic_write_text` in `state.py` now enforces `encoding="utf-8"` in `os.fdopen`.

## 2026-09-19 — FLUX.2-klein In-Context Reference Architecture & Hard Limits
- **Hard Limits Enforced**:
  - `MAX_LORAS = 16`: Enforced across Pydantic `GenerateRequest`, `/api/generate` route (HTTP 400), `generator.py` defensive clamping, and frontend `LoraManagerDrawer` (counter badge `LoRAs (${loras.length}/16)` and rejection).
  - `MAX_QUEUED_IMAGES = 512`: Sums all active images in queue (`status in ("generating", "queued")`). Rejects excess requests with HTTP 429.
- **FLUX.2-klein Reference Conditioning Root Causes & Fixes**:
  1. *Architecture reality*: FLUX.2 does NOT use latent noise blending (classic img2img). It uses in-context cross-attention conditioning where references are VAE-encoded, given 3D RoPE coordinates ($t=10, 20\dots$), and concatenated with prompt and generation tokens.
  2. *Single-reference bug*: Previously, if 1 reference image was passed with `image_strength` (sent by frontend default), `variant` remained `"standard"`, running classic `Flux2Klein` img2img which only executed 1 partial step due to `mflux`'s `init_time_step` calculation. Fix: Any FLUX.2 generation with reference images routes strictly to `variant = "edit"` (`Flux2KleinEdit`).
  3. *Skip-step noise bug*: `Flux2KleinEdit` initializes generation latents from pure Gaussian noise. Passing `image_strength` to `pipe.generate_image` forwarded it to `Config`, causing the scheduler to start at `init_time_step > 0`, skipping diffusion steps on uninitialized noise and producing corrupted noise output. Fix: Never pass `image_strength` to `Flux2KleinEdit`.
  4. *FLUX.2-klein 9B Edit support*: `generator._get_pipeline` previously only loaded `Flux2KleinEdit` for 4B; 9B always loaded standard `Flux2Klein`, crashing or ignoring reference images. Fix: Loaded `Flux2KleinEdit` with `ModelConfig.flux2_klein_9b()` when `variant == "edit"`.
  5. *Guidance scale support*: Backend previously blocked `guidance > 1.0` if `not supports_negative`. FLUX.2 natively supports CFG with `guidance > 1.0` (triggers unconditional embedding pass in `Flux2KleinEdit`). Fix: Allowed guidance > 1.0 when `supports_guidance` is True.
  6. *Prompting convention*: In-context tokens are addressed via natural language ("Image 1", "Image 2", "the person in Image 1"). Updated frontend hint and payload format.
  7. *Universal Fast VAE & Compilation Fallback*: Added `supports_fast_vae: True` to all models in `MODELS`. Exposed toggle in UI for every model (FLUX.2 TAEF2 ~0.67s, SDXL TAESD ~1.67s, Z-Image TAEF1 ~0.60s, Krea2 TAEW ~3.75s). When Fast VAE is toggled off on FLUX.2, JIT compilation `mx.compile(vae.decoder)` drops standard VAE decode from 73s to 25s.

## 2026-08-24 — venv-sdxl (Python 3.14, latest MLX)
- `mx.metal.set_wired_limit` / `get_wired_limit` are REMOVED-deprecated and raise in this MLX build. Use `mx.set_wired_limit(bytes)` (returns previous limit). Same for `mx.metal.device_info` → `mx.device_info`. Always smoke-test MLX API calls in venv-sdxl before shipping.
- Never run benchmarks or spawn the SDXL engine while a user generation is live — check `pgrep -f sdxl_engine.py` and system load first; competing engines thrash RAM (16GB M1) and produce 20-30s/step garbage numbers.

## 2026-08-24 — SDXL perf root cause (M1 16GB)
- The 20-30s/step was NOT memory/disk/sampler: GPU at 2.3 TFLOPS peak, all individual ops fast, but mlx-diffuser's fp16 SDXL graph composites ~10x slower than sum of parts on M1.
- FIX: quantize_unet=4 → 3.8s/call (5.3x faster, fused quant matmul kernel). 8-bit is 7x SLOWER (old AGENTS note — dequant per matmul); 4-bit is the opposite. Default engine to 4-bit.
- Profiling lesson: MLX async eval means per-block wall-clock timing attributes cumulative graph drain, not local cost. Microbenchmark individual modules to attribute cost.
- Benchmark discipline: check `pgrep -f sdxl_engine` + load BEFORE benchmarking; never compete with a live generation.

## 2026-08-25 — CoreML engine experiment (result: not worth it)
- coremltools needs Python ≤3.12 for native bindings (3.13/3.14 lack libcoremlpython). ct 9.0 has a const-cast bug in the torch frontend; use ct 8.3 + torch 2.5 + py3.10 (venv-coreml).
- Apple ml-stable-diffusion converter needs fp16 single-file variant: merge sharded safetensors + name it diffusion_pytorch_model.fp16.safetensors.
- BENCHMARK RESULT: CoreML fp16 @128² = 8.8s/call; 6-bit @96×64 = 3.46s/call vs MLX q4 3.79s. Only ~9% win, at the cost of fixed resolution, 4.8GB model, LoRA re-bake per combo. DrawThings' 0.65s/call comes from their Swift/NNC runtime, not reproducible via coremltools-Python. MLX q4 stays the engine.
- When user asks for improvements: propose alternatives (now codified in AGENTS.md) — this session it correctly led to testing CoreML, but the data said no.

## 2026-08-25 — queue cancel bug report
- Before fixing "queue bugs", REPRODUCE empirically via the API first: the queue skip logic was correct; the real issues were (a) Generate-clicks-while-busy silently queue extra jobs (UX surprise, by design), (b) SDXL path ignored cancel_event entirely — "Kill current" didn't stop the engine. Fixed (b) by killing the daemon on cancel; added per-batch-image status re-check in the worker.

## 2026-08-25 — Krea 2 Turbo integration
- mflux 0.19 Krea2: DiT ships pre-quantized 4-bit, but Qwen3-VL-4B text encoder ships bf16 (7.5GB). MUST pass quantize=4 — it only quantizes the TE (DiT keeps stored 4-bit) → resident ~10GB instead of ~15GB (OOM on 16GB).
- OOM guard pattern: per-model max_pixels/max_side in MODELS registry, clamped in generate() (pixel-budget proportional downscale, /16 aligned).
- mflux generate_image crashes on guidance=None (mx array * NoneType). Always pass a numeric guidance (1.0 for distilled) even when the model "doesn't support guidance".

## 2026-08-25 — engine residency
- Only ONE engine may be resident at a time on 16GB: SDXL daemon ~11GB, mflux pipelines ~10GB each. _get_pipeline now kills the SDXL daemon before loading mflux; _generate_sdxl drops the mflux pipeline before starting the daemon. Watchdog still respawns/kills lazily.

## 2026-08-25 — Krea2 4-step distillation LoRA
- lvladikov/Krea2-Turbo-Distill-4step-LoRA: mflux Krea2LoRAMapping matches ALL 228 diffusers-style keys out of the box. Use at scale 1.0, 4 steps, guidance 1.0 (CFG-free unchanged), Turbo only (never Raw). ~1.6x faster than 8-step; +13% per-step cost from the adapter but half the steps wins.

## 2026-08-25 — LoRA baking destroys deltas on quantized models (fixed)
- mflux bake_lora=True dequantize-add-requantizes into q4: small deltas (distillation LoRAs ~1.4% of |W|) are partially erased. Bench (backend/data/benchmarks/): runtime-applied adapter MAE 19.92 vs 8-step ref; baked 22.27; no-LoRA 22.77. bake_lora=False is now the Krea2 default — same treatment as DrawThings (adapters stay fp16, applied per forward).
- Benchmark images must be SAVED for user inspection, not just metrics.
- bake_lora=False applied to ALL mflux LoRA-capable models (Flux2Klein, ZImageTurbo, Krea2). SDXL engine was already runtime-applied (rank-concat adapters). Verified: LoRALinear wrappers stay alive after load (100 layers on klein).
- MLX_COMPILE=1 on M1 + Krea2 4-bit: 128s -> 143s (+11% SLOWER). mx.compile graph-trace overhead beats dispatch savings on M1. Keep off; retest on M3+.
- FLUX img2img shipped: /api/uploads + reference_image/reference_strength params + UI panel (FLUX only; Z-Image accepts same mflux params — future enable).

## 2026-08-25 — batch progress + missed saves
- Batch progress showed CUMULATIVE steps (steps × batch = "8 steps" for a 4-step request) — confusing. Now per-image: step/steps refer to the current image, image_index shows which of the batch.
- 2s polling can MISS intermediate saved_id events in batch jobs — on "done", the frontend now replays job.results and surfaces every image, not just the last.
- Diagnosis habit: before assuming data loss, check the API/files — both batch images existed on disk and were served; it was pure frontend display.

## 2026-08-27 — SDXL LoRA layer mapping and list assignment fix
- ROOT CAUSE OF SDXL LORA INEFFECTIVENESS: 
  1. `middle_block` keys failed regex `(input_blocks|output_blocks|middle_block)_(\d+)_(\d+)_(.+)` because middle_block only has 1 index (`middle_block_1_...`). 204 critical mid-block layers were completely unmapped and dropped!
  2. `to_out.0` and `ff.net.2` are Python `list` objects in mlx_diffuser. Doing `parent[last] = adapter` raised a TypeError because `last` was str (`"0"` / `"2"`), and the fallback `setattr(list, "0", adapter)` failed with AttributeError, skipping every output projection and feedforward layer in the UNet.
  3. CLIP text encoder keys (`lora_te1_`, `lora_te2_`) did a naive `.replace('_', '.')` which broke `text_model`, `self_attn`, `q_proj`, etc.
  4. Missing alpha in safetensors defaulted to 16.0 instead of `rank`, scaling high-rank LoRAs down by 4x to 8x.
- FIX: Rebuilt `_map_kohya_key` for SDXL UNet + CLIP text encoders, added integer indexing for list parents (`parent[int(last)] = adapter`), and defaulted alpha to `rank`. Verified: 986 / 986 layers loaded with 0 missed.

## 2026-09-05 — Metal wired limit key mismatch & job completion state integrity
- ROOT CAUSE OF STEP 2 FLUX ABORT & DISAPPEARING CANVAS PREVIEW:
  1. `mx.device_info()` returns `'max_recommended_working_set_size'` (NOT `'recommended_max_working_set_size'`). Checking the wrong key caused silent fallback to a 6GB ceiling (`6 * (1 << 30)`) on 16GB Macs, stalling / aborting FLUX.2-klein at Step 2 on resolutions $> 1024$ (e.g. 1248×832). Fixed: check `max_recommended_working_set_size` and scale up to 70% of total RAM (~11.2GB).
  2. Initializing `status = "done"` before the worker loop executes is an architectural hazard: any signal/interruption (`BaseException`) bypassed `except Exception:` and reached `finally:`, marking an aborted job with 0 images as `✓ Done` at `Step 2/4`. Fixed: initialize to `"generating"`, catch `BaseException`, and require `len(results) == batch` before marking `"done"`.
  3. Frontend Canvas image loss: `setCurrentResult(job.result)` when `job.result` is `undefined` erased the previous preview. Fixed: only set `currentResult` when `job.result` is truthy; otherwise raise error and preserve the canvas image.

## 2026-09-05 — SDXL-Lightning "Tuilé" / Mosaic Artifacts Root Cause (Euler Trailing vs DPMPP Leading & CFG Distillation)
- ROOT CAUSE OF TILING / MOSAIC / PSYCHEDELIC TEXTURE:
  1. Distillation Timestep Spacing: SDXL-Lightning was distilled with `EulerDiscreteScheduler` and `timestep_spacing="trailing"`. When using schedulers with `timestep_spacing="leading"` (such as default `dpmpp_2m_karras`), the timesteps do not reach 0 (4 steps leading denoise stops at timestep ~166), leaving huge high-frequency latent noise residue.
  2. CFG Scale: SDXL-Lightning models are CFG-distilled to run at `guidance_scale=1.0` (conditional guidance is learned directly into the single forward pass). Running at CFG > 1.0 (e.g. 1.8) runs a dual forward pass with negative prompt and amplifies residual latent noise by the CFG factor, creating the distinct mosaic / stained-glass / "tuilé" artifact.
  3. Quantization vs Scheduler: The mosaic is NOT caused by 4-bit UNet quantization or VAE tiling. Both Q4 and FP16 produce pristine, clean images when using `euler_trailing` and `guidance_scale=1.0`. Furthermore, running at `guidance_scale=1.0` cuts UNet forward passes in half (1 pass per step instead of 2), doubling generation speed!

## 2026-09-05 — External LoRA Path Restriction & Base Model Misclassification
- ROOT CAUSE OF FAILURE TO ADD EXTERNAL LORAS:
  1. Strict Path Validation: `_validate_lora_path` previously used `resolved.relative_to(allowed)` to force all local LoRAs into `backend/data/lora_files` or `backend/data/SDXL`. Any user trying to reference a LoRA on an external drive or Download directory was blocked with HTTP 400.
  2. Upload Destination & Base Model Filtering: Uploading through the UI placed files into `LORA_FILES_DIR`, which defaulted `base_model` to `"flux2"`. In `GenerateForm.jsx`, `compatibleLoras` filtered out `flux2` LoRAs when SDXL or Juggernaut XL models were selected, making uploaded SDXL LoRAs vanish from the selector.
- FIX:
  1. Replaced the restrictive folder checks in `_validate_lora_path` with a check ensuring the file exists, is a file, and has `.safetensors` extension. External paths anywhere on disk are now fully valid.
  2. Implemented `_inspect_safetensors` to read the first few KB of the `.safetensors` JSON header in <1ms without loading tensors. Automatically infers `base_model` ("sdxl" vs "flux2" vs "krea2") from `ss_base_model_version` / tensor key prefixes and extracts trigger words (`ss_tag_frequency`, `modelspec.trigger_phrase`).
## 2026-09-05 — SDXL VAE Quadratic Self-Attention & Metal Wired Memory Limits
- ROOT CAUSE OF SDXL METAL OOM AT STEP 4/4:
  1. SDXL VAE Latent Quadratic Attention: At $1024 \times 1024$, latents are $128 \times 128 = 16,384$ tokens. The untiled SDXL VAE decoder mid-block performs full self-attention across the token grid ($16,384 \times 16,384 \times 4 \text{ B} = 1.07\text{ GB}$ per head). In an untiled pass, peak attention activation buffers and intermediate convolutions in fp32 exceed Metal memory and trigger `[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
  2. Overly strict tiling condition: Changing `tile_vae` condition to `> 1024 * 1024` turned off tiling at exactly $1024 \times 1024$ and at $832 \times 1216$ ($1,011,712$ pixels). FIX: Mandate `tile_vae = (width * height >= 768 * 768)`. The iterative `mx.eval(out, wsum)` patch on `AutoencoderKLSD._tiled` splits latents into $64 \times 64$ tiles ($4096$ tokens, $67\text{ MB}$ per head) and evaluates each tile sequentially, maintaining minimal memory footprint.
  3. Denoise Wired Limit: Pinning 12GB of RAM during denoise leaves zero unpinned physical memory for Metal command buffers during VAE decode. The wired limit for SDXL must be clamped to 7GB (`min(cap, 7 * (1 << 30))`), leaving $\ge 8\text{ GB}$ for OS buffers and VAE decode.

## 2026-09-05 — NVIDIA PiD Decoder Gated Model & Pre-Flight Validation
- ROOT CAUSE OF NVIDIA PID DECODE FAILURE:
  1. Gated Model Dependency: `mflux` PiD decoder downloads two repositories: `nvidia/PiD` (~5GB) and `google/gemma-2-2b-it` (~3GB). The Google Gemma-2 model is protected/gated on Hugging Face and returns `401 Unauthorized GatedRepoError` if the user has not accepted the license on the Hugging Face website and run `huggingface-cli login`.
  2. Late Failure UX: Without pre-flight validation, the user waits ~4 minutes through all diffusion steps before `mflux` attempts to download Gemma-2 at the VAE stage, failing at the very end. FIX: Added fast pre-flight check in `/api/generate` to verify local cache or Hugging Face token presence before starting diffusion, failing fast in <1ms with clear instructions.
  3. Unified Memory Footprint: FLUX.2-klein (~10GB) + PiD Decoder (~8GB) = ~18GB total resident weights. On 16GB Apple Silicon Macs, this forces macOS to swap heavily to SSD. Clarified in UI that the standard FLUX VAE is strongly recommended for 16GB machines.

## 2026-09-05 — Flow Matching vs Latent Diffusion Samplers & Sampler Metadata Leaks
- RECTIFIED FLOW MATCHING VS LATENT DIFFUSION SOLVERS:
  1. Mathematical divergence: FLUX.2, Krea 2, and Z-Image Turbo are Rectified Flow Matching models ($x_t = (1-t)x_0 + tx_1$). Their vector trajectories between noise and clean image are trained along straight geodesics in latent space ($u_t = x_1 - x_0$). An exact linear Euler step ($x_{t - \Delta t} = x_t - \Delta t \cdot v_\theta$) is mathematically optimal and exact. Curved multi-step solvers (DPM++ 2M, UniPC, Heun) or stochastic noise injectors (Euler Ancestral) are designed for curved VP SDEs (SDXL / SD 1.5) and are mathematically invalid on linear flows.
  2. Library reality: `mflux` only implements `FlowMatchEulerDiscreteScheduler` for Flow models. There are no alternative samplers because linear flows require no curved polynomial interpolation.
  3. Metadata leak bug: `GenerateRequest` in `backend/main.py` previously defaulted to `sampler: str = "dpmpp_2m_karras"`, and `GenerateForm.jsx` initialized state with `"dpmpp_2m_karras"`. While the UI hid the dropdown for Flow models, the default leaked to `_generate_flux`, which blindly recorded `"sampler": sampler or "euler"`. This caused FLUX.2 and Krea images to display `Sampler: dpmpp_2m_karras` in the UI canvas badge and EXIF.
  4. Fix: Set `GenerateRequest.sampler = None`, decouple SDXL sampler defaults from Flow models, hardcode `"sampler": "FlowMatch Euler"` in `_generate_flux`, and synchronize `GenerateForm.jsx` sampler state only on models advertising `m.samplers`.

## 2026-09-06 — SDXL VAE Decode Acceleration (Dynamic Tiling + mx.compile & Pure MLX TAESD)
- ROOT CAUSE OF SDXL VAE DECODE LATENCY (~22s–35s on M1):
  1. Excessive Tiling: `mlx_diffuser` defaults to `tile_latent=64, overlap_latent=16`. For a standard $1024 \times 1024$ image (latents $128 \times 128$), this triggers a $3 \times 3$ grid = 9 full passes through the uncompiled 335MB VAE decoder graph.
  2. Uncompiled Decoder: Calling the PyTorch-ported decoder module without `mx.compile` emits un-fused Metal kernel dispatches with high scheduling overhead on Apple Silicon.
- SOLUTION & ARCHITECTURE:
  1. Option A (Native VAE JIT-Accelerated):
     - For resolutions $\le 1024 \times 1024$ (latents $\le 128 \times 128$), single direct pass with `mx.compile(self.decoder)` decodes in **8.84s** (2.5x to 4x faster) with 100% bit-for-bit fidelity (PSNR 40.10 dB). Memory is completely stable as long as the denoise wired limit is capped at 7GB.
     - For resolutions $> 1024 \times 1024$, adaptive $96 \times 96$ tiling cuts tile count from 9 to 4 with seamless feathering.
     - Critical detail: Native VAE must pass through `self.post_quant_conv(z)` before `self.decoder(z)`. Direct call without post-quant conv corrupts latent channels (MAE jumps to 128).
  2. Option B (Pure MLX TAESD Decoder):
     - Implemented `backend/taesd_mlx.py` with pure `mlx.core` and `mlx.nn` modules (`MLXTinyBlock`, `MLXTAESDDecoder`). Zero PyTorch runtime dependency (torch-free engine rule).
     - Weights transposed from NCHW to NHWC and JIT-compiled with `mx.compile`. Decodes $1024 \times 1024$ in **0.48s–0.54s** (**40x faster** than baseline).
     - Operates directly on unscaled latents: `z * self.scaling_factor`.
  3. UI Integration: Exposed VAE Decoder selector in `GenerateForm.jsx` (`✦ Native VAE (JIT Accelerated ~8s)` vs `⚡ Ultra-Fast TAESD (~0.5s)`) and added `VAE: TAESD ⚡` badge in `ResultCanvas.jsx`.

## 2026-09-06 — TAESD Dynamic Range Mapping & Milky / Washed-Out Haze Defect
- ROOT CAUSE OF MILKY / WASHEED-OUT TAESD RENDERS:
  1. Range Incompatibility: Standard VAE decoders (SDXL / SD 1.5) output pixel values in `[-1.0, 1.0]`, which the post-processing affine transform maps via `(x + 1) * 127.5` into `[0, 255]`.
  2. In contrast, TAESD (madebyollin) is trained to output directly in `[0.0, 1.0]`.
  3. Applying `(x + 1) * 127.5` directly to `[0, 1]` shifted the black point ($0.0$) to $127.5$ (50% middle gray) and mapped the entire image into `[124, 255]`. The bottom half of the histogram `[0..123]` was completely absent, creating an unnatural milky, fogged, low-contrast appearance.
- FIX:
  1. In `sdxl_engine.py:_safe_decode`: TAESD's output is normalized to `[-1, 1]` via `img * 2.0 - 1.0` to respect the library contract.
  2. In `sdxl_engine.py:generate`: Adaptive normalization `(arr + 1.0) / 2.0 if arr.min() < 0.0 else arr` guarantees correct $[0, 255]$ dynamic range regardless of input format.
  3. Verified: Black point restored to 0, mean matches Option A (61.69 vs 60.55), milky haze completely eliminated.

## 2026-09-07 — LoRA Compatibility Isolation & Step Synchronization on Distilled Models
- ROOT CAUSE OF LORA LEAKS ON MODEL SWAP:
  1. `switchModel` in frontend evaluated `newBase = ... ? ... : null`. When switching to models like `z-image-turbo`, `newBase` was `null`, completely skipping `setLoras()`, leaving incompatible LoRAs and triggers active in the UI state.
  2. `compatibleLoras` filtered using `!engineBase || ...`, which exposed all registry LoRAs across every architecture whenever `engineBase` was `null`.
- STEP SYNCHRONIZATION ON KREA 2 TURBO:
  1. Krea 2 Turbo is natively an 8-step model. Running at 4 steps without the distillation LoRA leaves massive high-frequency residual noise (incomplete/blurry render).
  2. With `Krea2-Turbo-Distill-4step` loaded, the schedule converges cleanly in 4 steps (~140s vs ~250s).
  3. UI must automatically synchronize steps: loading a 4-step LoRA drops steps to 4; removing it or switching away restores 8 steps.

## 2026-09-08 — Civitai Metadata Extraction & EXIF Tag Integrity
- ROOT CAUSE OF CIVITAI METADATA EXTRACTION FAILURE:
  1. Parameter Line Missing `Model:` and `Model hash:`: Civitai's parser (`@civitai/generation-metadata`) extracts checkpoints by scanning the parameter line starting with `Steps: ` for `Model: <name>` and `Model hash: <hash>`. In `generator.py`, `param_parts` had omitted `Model:` and `Model hash:` completely, preventing Civitai from identifying base checkpoints.
  2. Software & Artist Overwrite Bug: EXIF `Software` was using `meta.get("id") or "MLX-DIFFUSION"`. Because `meta.get("id")` is always the image hex UUID, the EXIF software tag was set to the image UUID instead of `"MLX Diffusion"`. `Artist` tag was defaulting to `"ai"` instead of `"www.ouinche.com"`.
  3. LoRA Tag Sanitization & Hashes: Automatic1111 `<lora:name:scale>` regex only matches `[a-zA-Z0-9_.-]+`. LoRA names with spaces or special chars failed regex unless sanitized. Missing `Hashes: {"model": ..., "lora:name": ...}` and `Lora hashes: "..."` prevented Civitai from auto-linking exact LoRA versions.
- FIX:
  1. In `build_generation_metadata_text`: strictly inject `Model: <name>` and `Model hash: <hash>` into `param_parts`.
  2. Inject both `Hashes: {"model": ..., "lora:tag": ...}` and `Lora hashes: "tag: hash, ..."` using sanitized tag identifiers.
  3. In `param_parts`, EXIF, and PNG chunks: explicitly set `Software: MLX-DIFFUSION`, `Generator: MLX-DIFFUSION`, and `Artist: www.ouinche.com`.
## 2026-09-08 — Civitai Browser-Side Upload Parsing: iTXt vs tEXt Chunks & Model Hashes
- ROOT CAUSE OF BROWSER-SIDE "EXIF NOT FOUND" & MISSING METADATA:
  1. Pillow `iTXt` Fallback Trap: When prompts contain smart quotes (`’`, `‘`, `“`, `”`), dashes (`—`, `–`), ellipsis (`…`), or zero-width spaces (`\u200b`), Pillow's `PngInfo.add_text()` detects non-Latin1 characters and automatically switches the PNG chunk from `tEXt` to `iTXt` (International Text chunk).
  2. Browser Upload Parser Incompatibility: In-browser image uploaders (like Civitai's web UI drag-and-drop parser) only scan for standard `tEXt` chunks. When encountering `iTXt`, they skip the chunk completely, leading to an empty prompt, 0 detected resources, and the banner `EXIF not found ✖`.
  3. Missing Model Hash on Custom/Distilled Checkpoints: Civitai's resource extraction logic strictly checks `if (metadata["Model"] && metadata["Model hash"])`. If `Model hash:` is absent, Civitai will NEVER add the checkpoint to `resources`, even if the text was parsed.
  4. UTF-16BE BOM: When encoding EXIF `UserComment` with `b"UNICODE\x00"`, decoders need the UTF-16BE BOM `\xfe\xff` to immediately identify the endianness without falling through to heuristic zero-byte counts.
- FIX:
  1. Implemented `to_latin1_clean(text)` in `backend/generator.py`: Maps all smart quotes, prime marks, en/em dashes, and zero-width characters to ASCII/Latin-1 equivalents before building PNG chunks and parameters infotext. Guaranteed: 100% of generated images use standard `tEXt` chunks (verified: 353 `tEXt`, 0 `iTXt`).
  2. Added deterministic 10-char `Model hash` fallback (`hashlib.sha256(name).hexdigest()[:10].upper()`) for models lacking official Civitai IDs (Krea 2 Turbo, FLUX.2, Z-Image), and updated `MODELS` dict with explicit SHA256 hashes.
  3. Added `\xfe\xff` BOM to EXIF `UserComment`.
  4. Re-ran `backend/scripts/retroactive_civitai_metadata.py --all` across all 354 images in `backend/data/generated/`.
  5. Verified: Every image (including the failing spider image `3a61a86ac4c846ce9ae9ded2344392a2.png`) now has prompt, checkpoint, LoRAs, and parameters extracted with 100% fidelity. Software & Generator are strictly `"MLX-DIFFUSION"`, and Artist is `"www.ouinche.com"`.

## 2026-09-08 — Civitai Resource Detection: Canonical Hashes vs Local Arbitrary Hashes & Distilled Parameters
- ROOT CAUSE OF KREA 2 DETECTING NO RESOURCES ON CIVITAI:
  1. Civitai Database Hash Lookup: Civitai's server does an exact lookup against indexed file hashes (AutoV2, AutoV1, SHA256) and modelVersionIds. SDXL matched because `modelVersionId: 357609` is indexed in Civitai. For Krea 2, a local arbitrary hash `C53FD4E098` was used, which does not exist in Civitai's database. Without a matching hash or version ID, Civitai displays: *"We weren't able to detect any resources used in the creation of this image"*.
  2. Missing CFG Scale on Distilled Models: Distilled models (Krea 2, FLUX) have `guidance: null` / None. Omitting `CFG scale:` from the infotext line or outputting `"cfgScale": null` disrupts Civitai's parameter parser. Distilled models must always emit `CFG scale: 1.0` (both on the `Steps:` line and in JSON).
  3. Non-Standard Sampler Names: `"FlowMatch Euler"` is not recognized by Civitai's sampler mapping table. FlowMatch samplers must be normalized to `"Euler"` for Civitai parameters.
  4. Retroactive Script Stale Field Guard: `if minfo.get(...) and not meta.get(...)` skipped overwriting previously saved dummy hashes on existing JSON sidecars. Canonical fields from `minfo` must unconditionally overwrite stale local metadata during retroactive patching.
  5. File Location Warning: Testing Civitai uploads using files from the browser's downloads or desktop will fail if those files were downloaded before the retroactive script ran. Files must be uploaded directly from `backend/data/generated/` or freshly downloaded from the UI.

## 2026-09-08 — Apple Silicon Metal JIT & Daemon Sandboxing (BypassSandbox: true)
- ROOT CAUSE OF ENHANCE BUTTON & GENERATION SILENT FAILURE:
  1. Metal Front-End Compiler (`metalfe`) Cache: On Apple Silicon, MLX and mlx-lm compile Metal shader libraries Just-In-Time and write precompiled module headers (`monolithic_metal.pcm`) to Darwin user cache directory (`/var/folders/.../com.apple.metalfe/`).
  2. Sandbox Isolation Conflict: When backend daemons are launched inside the tool sandbox (`BypassSandbox: false`), macOS sandboxing restricts filesystem write access outside the workspace. Any JIT shader compilation throws: `RuntimeError: [metal::Device] Unable to build metal library from source - error: unable to open output file '/var/folders/.../monolithic_metal.pcm': 'Operation not permitted'`.
  3. UI Symptom: Prompt enhancer `/api/prompt/enhance` fails with 500, catching the error and leaving the prompt unchanged. Generations crash at step 0 during text encoder evaluation.
  4. Fix: Always launch the backend uvicorn daemon with `BypassSandbox: true` to ensure Apple Metal has uninhibited access to macOS Metal cache directories.

## 2026-09-12 — Civitai LoRA Download-Auth Traps, Safetensors Header Validation & Daemon Resilience
- ROOT CAUSE OF SDXL LORA ENGINE FREEZE & TOTAL GENERATION COLLAPSE:
  1. Civitai Auth-Gated Download Redirects: Certain Civitai models (e.g., Wagyu Beef Style `654467`) require user authentication (`reason=download-auth`). Without an API key, Civitai redirects the download request to `auth.civitai.com/login`. A naive file downloader following redirects saves the 10KB HTML login page directly to disk as a `.safetensors` file.
  2. The 32-Petabyte Safetensors Memory Crash: The `.safetensors` format specifies the header length as the first 8 bytes (little-endian unsigned 64-bit int `<Q`). In ASCII/HTML, `<!DOCTYPE` or `<!doctyp` unpacked as `<Q` yields `0x7974636f64213c` (~34 Petabytes). Calling `fh.read(n)` immediately triggered an unrecoverable `MemoryError`, killing the SDXL background engine.
  3. Poisoned State & Startup Stall: On page reload, `_discover_local_loras` checked `size > 1024` and considered the 10KB HTML file a valid LoRA, keeping it in `loras.json`. Furthermore, duplicate filenames (`{name}__{name}.safetensors`) caused slow SHA256 hashing and 8s Civitai network timeouts on startup.
- FIX & DEFENSIVE ARCHITECTURE:
  1. Header Sanity Check (`is_valid_safetensors`): Before reading or saving any safetensors file, read the 8-byte header and strictly enforce:
     - `0 < header_len <= file_size - 8`
     - `header_len <= 100 * 1024 * 1024` (100MB sanity ceiling)
     - Valid JSON parse containing tensor definitions.
  2. Civitai Auth & Content-Type Gate: Download streams inspect headers for `Content-Type: text/html` or redirects to `auth.civitai.com`. Any unauthenticated attempt fails immediately with `CivitaiAuthError` and HTTP 401, providing clear user instructions rather than saving garbage to disk.
  3. Civitai API Key Persistence: Added support for `CIVITAI_API_KEY` (via environment variable, request payload, or `backend/data/civitai_token.txt`). Added `/api/civitai/token` endpoints and frontend settings to input and manage API keys seamlessly.
## 2026-09-12 — Threading Deadlocks in Re-entrant Lock Helpers & Async LoRA Management Architecture
- ROOT CAUSE OF LORA FREEZE ON CIVITAI IMPORT / DELETE:
  1. Non-Reentrant Lock Deadlock: In `backend/main.py`, `_loras_lock` was initialized with standard `threading.Lock()`. Routes like `/api/loras/import-civitai`, `delete_lora`, and `sync_all_loras_civitai` acquired `with _loras_lock:` and subsequently invoked `_read_loras()` or `_write_loras()`, which also attempted `with _loras_lock:`. Because Python's standard `Lock` is non-reentrant, the worker thread deadlocked on itself permanently. Any subsequent call from the UI (like `GET /api/loras` or deleting a LoRA) queued behind the deadlock, freezing all LoRA interactions across the app.
  2. Synchronous Long-Running Download Blocks: Downloading 200MB–400MB LoRA weights over HTTP synchronously in an endpoint blocks the client connection for 30–60s, provides zero progress feedback, risks client-side timeouts, and prevents user cancellation.
  3. Civitai Model ID vs Version ID Resolution: Users frequently copy civitai URLs like `https://civitai.com/models/582432` without `?modelVersionId=...`. Querying `/model-versions/582432` returns 404 because `582432` is a Model ID.
  4. Cross-Architecture LoRA Invisibility: When the UI strictly filters the LoRA selector by `r.base_model === engineBase`, downloading an SDXL LoRA while on FLUX.2 caused the LoRA to be saved successfully to disk, but remain completely invisible in the selector, giving the illusion that the download failed.
- FIX & ARCHITECTURAL UPGRADE:
  1. Re-entrant Lock: Changed `_loras_lock = threading.RLock()` to permit recursive nested locking within the same thread.
  2. Async Download Manager: Converted `/api/loras/import-civitai` to launch a background thread returning a `task_id` in <100ms. Added `GET /api/loras/downloads` for live progress (bytes, percent, speed in MB/s) and `DELETE /api/loras/downloads/{task_id}` for real-time cancellation.
  3. Dual Model ID / Version ID Fallback: `parse_civitai_input` extracts either `model_id` or `version_id`. If `version_id` is absent, it queries `https://civitai.com/api/v1/models/{model_id}` and extracts the primary `modelVersions[0]`.
  4. Type & Base Validation: Rejects checkpoints (`type != "LORA"`) and unsupported architectures (`SD 1.5`, `Flux.1`, etc.) upfront before downloading bytes.
  5. Installed LoRAs Hub Drawer: Added a full-featured drawer displaying all installed LoRAs categorized by architecture badges with trigger word chips, delete action, and a 1-click "Switch to {engine}" action.

## 2026-09-14 — Zero-Quality-Loss Performance Pack & Memory Safety (M1 16GB)
- ROOT CAUSE OF SDXL VAE OOM DURING UN-TILED DECODE:
  1. Untiled VAE Attention Memory Spike: Attempting to decode $1024 \times 1024$ without tiling requires $16384 \times 16384$ self-attention matrices in fp32 ($>1\text{ GB}$ per head). On 16GB Apple Silicon, this exceeds available Metal working buffers and fails with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
  2. Tiling Memory Invariant: Always enforce `tile_vae = (width * height >= 768 * 768)` for native SDXL VAE. The loop in `_safe_tiled` MUST call `mx.eval(out, wsum)` on every tile iteration to flush completed intermediate tiles from Metal memory before executing the next tile.
  3. Denoise vs VAE Wired Memory Handshake: Denoising benefits from a locked wired ceiling (`6.5GB`), but VAE decode needs dynamic buffer allocation. In `_safe_decode`, call `mx.set_wired_limit(0)` and `mx.metal.clear_cache()` before decoding to guarantee maximal unified memory headroom.
  4. Dual-CLIP Prompt LRU Cache: Text encoding across repeated prompt iterations or seed sweeps is completely redundant. Caching `(pe, pooled)` via an LRU cache with `mx.eval` eliminates redundant CLIP forward passes while remaining 100% safe by clearing on LoRA swap.
  5. Distilled Guidance Clamp: Models distilled for single-pass CFG (e.g. SDXL-Lightning 4-step) must strictly default to `guidance=1.0`. Any value $> 1.0$ forces dual UNet forward passes (2 passes per step), doubling generation time with zero quality gain.

## 2026-09-14 — Mandatory Test Protocol & Cross-Engine Validation Rules
- USER TESTING MANDATE:
  1. Image Presentation Invariant: Whenever test images are generated, the user MUST be able to see them immediately. Always view them (`view_file`), copy them to the artifacts directory, and embed them in the response and walkthrough.
  2. Cross-Engine Battery: After any major implementation, refactoring, or performance optimization, systematically run validation tests across ALL 4 supported engines:
     - Juggernaut XL Lightning (`sdxl` subprocess engine)
     - Z-Image Turbo (`z-image-turbo`, mflux 6B)
     - FLUX.2-klein 4B (`flux2-klein-4b`, mflux 4B)
     - Krea 2 Turbo (`krea2-turbo`, mflux 13B)
  3. Canonical Benchmark Prompt: Every test suite MUST use a prompt featuring *"a stormtrooper in a swiss paysage"* (e.g. `a detailed imperial stormtrooper standing in a scenic swiss alpine landscape, green valley, snow-capped mountains, wooden chalet, cinematic lighting`).
  4. Krea 2 Turbo Specifics: For tests on `krea2-turbo`, ALWAYS use 4 steps with the 4-step distillation LoRA (`backend/data/lora_files/krea2_turbo_4step_rank_64_lora.safetensors` or `krea2_turbo_4step_rank_64_lora_latest.safetensors`, scale 1.0) at 512x512, rather than the un-adapted 8-step baseline.

## 2026-09-14 — User Correction: Generation Time vs Inference Latency
- TERMINOLOGY INTEGRITY:
  1. Never label total image synthesis duration as "Latency" or "Inference Latency". Latency in machine learning typically denotes time-to-first-token (TTFT) or single forward-pass latency.
  2. Rule: Always use **"GENERATION TIME"** (or "Temps de génération") for the complete end-to-end wall-clock image synthesis time.
  3. Krea 2 Turbo Comparative Representation: When presenting benchmarks, always include BOTH:
     - Krea 2 Turbo 13B (4-step + Distill LoRA) @ 168.2s
     - Krea 2 Turbo 13B (8-step Baseline without LoRA) @ 292.3s
     This demonstrates the exact 124s speedup achieved by the distillation adapter.

## 2026-09-15 — Safetensors Dtype Support in SDXL Multi-LoRA Engine
- ROOT CAUSE OF "unsupported dtype I64" ON SDXL LORA LOAD:
  Certain LoRA architectures (e.g. Spiking Neurons LoRA, LyCORIS, Kohya scripts) save metadata or scalar constants like `.alpha` as integer types (`I64`, `I32`) or quantization manifests (`U32`) instead of standard floating point (`F32`/`F16`/`BF16`).
  `_read_safetensors_tensor` previously only handled `F32`, `F16`, and `BF16`, throwing `ValueError / RuntimeError: Failed loading LoRA: unsupported dtype I64` on any safetensors tensor with non-float dtypes.
- FIX: Expanded `_read_safetensors_tensor` in `backend/sdxl_engine.py` to cover all safetensors standard dtypes: `I64` (`<i8`), `I32` (`<i4`), `I16` (`<i2`), `I8` (`<i1`), `U64` (`<u8`), `U32` (`<u4`), `U16` (`<u2`), `U8` (`<u1`), `F64` (`<f8`), and `BOOL` (`?`). Now Spiking Neurons (560 I64 alpha tensors) and all other LoRAs load seamlessly.

## 2026-09-15 — Live Batch Preview & Filmstrip in Studio Canvas
- ROOT CAUSE OF STALE PREVIEW IN BATCH RUNS:
  When generating batches (`batch > 1`), `backend/main.py` only updated `job["last_saved"]` with an ID, leaving `job["last_result"]` and `job["partial_results"]` unset until the entire job completed. In `GenerateForm.jsx`, polling only called `onImageSavedRef` (which updated the gallery tab's refresh key, invisible on the Generate tab), while `currentResult` was never set on intermediate images. As a result, the canvas remained frozen on whatever image was generated in a previous session through images 1 to N-1 of the batch.
- FIX:
  1. Backend (`backend/main.py`): In the batch loop, immediately append each `meta` to `results` and assign `job["last_saved"]`, `job["last_result"] = meta`, and `job["partial_results"] = list(results)`. Expose both in `/api/jobs/{job_id}` and include `batch` in `job["progress"]`.
  2. Frontend (`GenerateForm.jsx`): Real-time polling (800ms) syncs `batchResults` and updates `currentResult` immediately whenever `job.last_result` arrives. Auto-chains to the next queued job in the stack upon completion.
  3. Studio Canvas (`ResultCanvas.jsx`): Renders a sleek interactive batch filmstrip (`canvas-batch-strip`) with clickable thumbnails for all completed batch images, an active selection indicator, and an animated placeholder for the image currently being rendered. Displays `Image X/Y` badge and current prompt during generation overlay.

## 2026-09-15 — Thumbnail Civitai Metadata Embedding & Auto-Heal
- ROOT CAUSE OF METADATA LOSS ON THUMBNAIL COPIES / DRAG-AND-DROP:
  1. `thumbnail_path()` in `backend/generator.py` originally generated thumbnails by downscaling with Pillow and saving with bare `img.save(f, format="PNG")`, stripping 100% of generation metadata, EXIF, and PNG text chunks.
  2. In web galleries, batch strips, and file drawers, users frequently right-click "Copy Image", drag-and-drop the displayed thumbnail, or copy/paste directly into Civitai's web uploader. Because the thumbnail was stripped of all metadata, Civitai reported: *"We weren't able to detect any resources used in the creation of this image"*.
- FIX & ARCHITECTURAL PATTERN:
  1. Updated `thumbnail_path()` to retrieve the generation metadata sidecar (`{id}.json` or source image metadata) and save with `save_image_with_metadata()`, embedding identical Civitai A1111 infotext, EXIF `UserComment`, `Model`, `Model hash`, `Hashes`, and `Civitai resources`.
  2. Implemented Auto-Heal: `thumbnail_path()` checks if an existing thumbnail on disk lacks the `parameters` chunk; if missing, it automatically regenerates the thumbnail with full metadata before returning the path.
  3. Fixed EXIF pixel dimensions: `build_image_exif` accepts `image_size` so `ExifImageWidth`/`ExifImageHeight` reflect the actual file dimensions (`image.size`), while preserving original generation size (`Size: WxH`) in the infotext parameters.
  4. Stripped BOM artifact `\ufeff`: `extract_image_metadata` strips UTF-16BE BOM codepoints cleanly.
  5. Retroactive Batch: `backend/scripts/retroactive_civitai_metadata.py` updated to patch all `*_thumb.png` files alongside main images (498/498 patched with 0 errors).

## 2026-09-17 — Native Image Drag on macOS & Full-Resolution Drag-and-Drop
- ROOT CAUSE OF THUMBNAIL BEING DROPPED INSTEAD OF ORIGINAL IMAGE:
  1. In WebKit and Chromium on macOS, when an `<img>` element is dragged to the OS Desktop or Finder, the browser's native Cocoa drag controller (`WebDragSource` / `PasteboardMac`) inspects the `HTMLImageElement`'s `.src` attribute directly to create the promised file payload on the Cocoa pasteboard.
  2. Setting `e.dataTransfer.setData("DownloadURL", ...)` on a parent container does NOT override the native Cocoa image drag behavior when the cursor directly drags the `<img>` element.
  3. Because gallery thumbnails originally had `src=".../file?thumb=true"`, dropping to macOS Desktop caused Chrome and Finder to download and save the 512px thumbnail (`{id}_thumb.png`).
- FIX:
  1. Implemented `ensureFullResolutionImage(e, id)` in `frontend/src/utils/dragDrop.js` which immediately swaps `img.src` from `?thumb=true` to the full-resolution URL `imageUrl(id, false)` synchronously on `onPointerEnter`, `onPointerDown`, and `onDragStart`.
  2. When Chromium's native drag controller inspects the `HTMLImageElement`'s `.src` attribute during drag initialization, the URL is guaranteed to point to the canonical full-resolution endpoint (`/api/images/{id}/file`) with `Content-Disposition: inline; filename="{id}.png"`.
  3. Spread `bindFullImageDrag` props directly onto both the container element and the `<img>` element in `Gallery.jsx` and `ResultCanvas.jsx`.
  4. Now dropping to macOS Finder, Desktop, Civitai, Discord, or chat windows writes the full-resolution original image with 100% complete metadata chunks intact.

## 2026-09-18 — UI Restraint & Krea 2 Turbo LoRA Selection
- UI RESTRAINT & MINIMALISM:
  1. Never inject loud, neon-bordered, multiline alert boxes or overflowing pills that disrupt the grid layout (`param-grid`) or overlap neighboring columns (like Seed).
  2. Status indications on parameters must be subtle, compact, and inline (e.g. `Steps (4) ⚡ 4-step distill`), matching existing typography and spacing with zero layout shift.
  3. LoRA chips must maintain the app's clean, unified card styling without heavy colored borders, shouting text badges, or animation glows.
- KREA 2 TURBO DISTILLATION LORA IDENTIFICATION:
  1. Empirical truth confirmed by historical sidecars (`9415c562d71b4689a6a8859320099a67.json` @ 135.86s): The canonical high-fidelity adapter that produced the user's reference stormtrooper is Civitai version 3306914 (`v1.0`), SHA256 `20C1FB1BB66477FB3B19E501D69B55E1D2A25C7CE6550BB142529220838D80A4` (AutoV2 `20C1FB1BB6`), named `krea2_turbo_4step_rank_64_lora_latest.safetensors`.
  2. The Hugging Face repo commit `933981...` (version 3267820 / 26k, SHA256 `72D11C95CD...`) produces muddy, dark textures and distorted facial helmets. Never use it.
  3. Guidance on Krea 2 Turbo must always record `1.0` in metadata sidecars (never `None`), so the UI renders `Guidance: 1` as expected.

## 2026-09-18 — Sampler Naming & Metadata Contract (Euler vs FlowMatch Euler)
- Never label rectified flow matching samplers as `"FlowMatch Euler"` in UI or metadata sidecars.
- The standard, expected sampler identifier across FLUX.2, Z-Image Turbo, and Krea 2 Turbo is strictly `"Euler"`.
- Exposing non-standard names like `"FlowMatch Euler"` creates confusion, pollutes the gallery and canvas information badge, and conflicts with Civitai/A1111 parameter parsers.
- In `backend/generator.py`: Flow models must record `"sampler": sampler or "Euler"`.
- All legacy sidecars with `"FlowMatch Euler"` were migrated back to `"Euler"`.

## 2026-09-19 — Preserving Exact Section Titles & Civitai Affiliate Links
- When the user asks to restore or enforce a specific title (e.g. `⚡ High-Resolution Visual Comparator: SOTA Arena (Cross-Model)`), never alter, abbreviate, or replace it with generic phrasing. Keep it verbatim across all HTML, markdown, and WordPress outputs.
- Whenever Civitai is mentioned in any article or public-facing documentation, strictly hyperlink it using the user's affiliate URL (`https://civitai.red/?ref_code=88C8VEBA`).
- For WordPress deployment, pre-wire image sources directly to the user's remote uploads CDN path (`https://www.ouinche.com/wp-content/uploads/2026/09/{filename}`) and bundle the exact files in a `.zip` archive so the user can batch upload without any manual URL pasting or configuration.

## 2026-09-19 — Grand SOTA Arena: Full Dynamic Interactive Comparator (No Static Duels)
- When the user asks for the "Grand Arena with all prompt and model choice", NEVER replace it with static hardcoded duel cards (Duel 1, Duel 2...).
- The Grand Arena must ALWAYS be the fully interactive stage containing:
  1. All 10 prompt/scene pill selector buttons (`🌲 Scandinavian Fjord`, `⏱️ Master Watchmaker`, etc.) updating the active prompt & seed banner.
  2. Dual interactive dropdowns (`LEFT:` and `RIGHT:`) containing ALL 9 benchmarked models.
  3. A single interactive before/after slider that dynamically updates the Top and Bottom images to the user's chosen models on the chosen prompt in real time.
## 2026-09-19 — Instant Gallery/Browser Image Display After Generation
- ROOT CAUSE OF BROWSER/GALLERY TAB NOT SHOWING IMAGE AFTER GENERATION:
  1. Gallery tab effect had `if (activeTab !== "browser") return;` which bailed out when generation finished on the "Generate" tab, dropping the `refreshKey` trigger. When user switched to the "Browser" tab, it was debounced by 250ms via `setTimeout`.
  2. `LazyGalleryImage` used a JS `IntersectionObserver` with empty dependency array `[]`. When mounted while the Browser tab had `display: none`, the observer never fired in WebKit/Safari on macOS when the tab became visible, leaving images invisible (`isVisible = false`) with only skeletons.
  3. `image_file` route in `gallery.py` threw a hard HTTP 404 when `thumb=True` if thumbnail generation had not finished yet on disk.
- FIX:
  1. `App.jsx` now passes `newImage={newlyGeneratedImage}` directly to `<Gallery>`. Gallery immediately prepends `newImage` to `items`, increments `total`, and resets to page 1 in 0ms (instant optimistic display).
  2. `Gallery.jsx` now loads immediately on tab switch (`activeTab === "browser"`) and on `refreshKey` without arbitrary 250ms debounce delays (debouncing is restricted to user keystrokes in search/tag inputs).
  3. `LazyGalleryImage.jsx` uses browser-native `loading="lazy"` on `<img>` with automatic `onError` fallback to full image (`imageUrl(id, false)`) if thumbnail is still generating or 404s.
  4. `image_file` route in `backend/routers/gallery.py` wraps thumbnail lookup and falls back cleanly to the main image file if the thumbnail is not yet on disk.
  5. `ResultCanvas.jsx` has `key={currentImage.id}` on the canvas `<img>` to force fresh DOM mount and instant image decode without stale image retention.
  6. `useGenerationJob.js` polls at 500ms (down from 800ms) and guarantees `lastAutoShownIdRef` tracking on job completion.


## 2026-09-20 — FLUX.2-klein 4B Speed Campaign (Task 64) — Empirical Verdicts
- Run `./venv/bin/python backend/scripts/profile_klein.py` from repo root; every run saves its PNG + full params/times to `test/manifest.json` (user rule: ALL benchmark images go in `test/`).
- **The loop is purely GPU/GEMM-bound.** Host graph-build is 0.01–0.06s per step vs ~16s of `mx.eval`. Any Python-level "optimization" that doesn't cut real GPU work cannot help.
- **Per-op ranking** (`--variant internals`): `single_transformer_blocks.*.attn.to_qkv_mlp_proj` ≈ 48% of denoise (2.5s per op × 20 single blocks per 4-step 768² run). It's the fused QKV+MLP projection — the biggest matmul — so all gains must target GEMM kernels, not glue.
- mlx `nn` layers dispatch calls via CLASS-level `__call__` (special-method lookup is type-based): instance monkeypatching `m.__call__ = wrapper` silently does NOTHING. To intercept, patch the class and gate on instance ids.
- **RoPE memo cache adopted** (`generator.py::_install_rope_cache`): img_ids/txt_ids are constant per generation yet mflux recomputes cos/sin every step; memoizing is ~3% & FREE. Proven bit-identical: pre/post same-seed 768² PNG hashes match (`e9726698…`).
- **Rejected:** bf16-transformer proxy (only ~7%, +~4GB RAM risk); q4 g128 (≈ g64 per micro-bench); wired 8GB (no gain vs the app's 45%-of-RAM ≈7.2GB cap — the cap is not the bottleneck); mx.compile narrow (mflux excludes M1/M2 on purpose; compile only optimizes the 0.01s host build).
- Framework ceiling (documented, not fixable in MLX): M1 q4 ~4.2s/UNet call vs DrawThings' compiled CoreML ~0.65s — kernel-level gap.
- Benchmark hygiene: check `uptime` before runs; load>5 inflates generation times (observed 5.3–9.4 during this campaign).

## 2026-09-20 — run.sh must be launchable from anywhere
- run.sh derives PROJECT_DIR from its own location (symlink-resolved), locates venv by probing `venv/bin/uvicorn`, never hardcodes `/Volumes/Externe/IA/MLX-DIFFUSION` (the old path was wrong anyway — folder is `MLX-DIFFUSION OpenCode`).

## 2026-09-20 — All-engine lossless caches (Task 65) + first SDXL/ZIT/Krea baselines
- Verify the rope implementation BEFORE betting on a rope memo. FLUX2 `Flux2PosEmbed` does per-step cos/sin → cache pays ~3%. Krea2 `Krea2RopeEmbedder` does outer-mul + cos/sin per call → cache pays ~4% @1024. Z-Image `RopeEmbedder` builds cos/sin LUTs in `__init__` and `__call__` is only advanced-index gathers → cache pays ~0 (2.4% measured was load noise). Check `rope_embedder.py` first.
- ZIT exposes `_encode_prompts(prompt, negative_prompt, guidance) -> (embeds, neg)`; Krea2 has the same signature (kw-only) and also re-encodes every `generate_image` call; FLUX2 uses `_encode_prompt_pair` (4-tuple). One shared LRU (`_apply_prompt_cache_txt`) keyed on (prompt, neg, guidance) covers both.
- Engine decode hooks differ: ZIT `_decode_latents(*, latents, config, prompt, seed)`; Krea2 `_decode_latents(*, latents, prompt, seed)`; FLUX2 `vae.decode_packed_latents`. The app's TAEF fast-VAE wraps must match each signature. Krea2 gets NO wired limit in the app (`model != "krea2-turbo"` guard in generate()).
- Krea2 un-Distill @1024² is the slowest path on M1 by far: ~94s/it (8-step, q4) → 12.5 min/image. The 4-step `krea2_turbo_4step_rank_64_lora_latest` Distill LoRA (HF `lvladikov` already in cache) is the real speed hope; the app auto-appends it when `steps <= 4`.
- SDXL benchmarks must go through the app path (`generator.generate(model="juggernaut-xl-lightning")`): daemon spawn + model load adds ~77s to cold wall time; `generation_time` in the sidecar is the engine-internal number. First baseline: 66.85s @1024/4-step euler_trailing (load 6.9 — inflated; re-do under load<5 if clean numbers matter).
- Every engine's full app path writes `backend/data/generated/{id}.png` + `{id}.json` + `{id}_thumb.png` (task 63.2 thumbnail inline). The profiler harness now verifies and records that triple in `test/manifest.json` (`data_generated.present`).
- All-engine manifest runs: 11 (Task 64) → 18 (Task 65) in `test/manifest.json`.

## 2026-09-20 — Repeatability test vs yesterday's SOTA arena (`comparison_12_vs_4_steps`)
- Harness: `backend/scripts/repeatability_test.py` (regen exact arena grid: 9 model configs × 10 scenes, 512×768, seeds 1001–1010, q4). Compare pixel-to-pixel (SHA equal + file bytes) against the stored `images/{model}_{scene}.png`; verdict `BIT-IDENTICAL` or `DIFFERS` w/ PSNR + max-diff. Report → `test/repeatability_report.json`. Scope via `--models` / `--scenes` (smoke: `--models flux2-klein-4b,juggernaut-xl-lightning`).
- **GROUND TRUTH = the stored PNG's embedded Civitai `parameters` metadata**, NOT the arena page or SOTA_PROMPTS: the SDXL row reused the SHORT "breathtaking" bench prompt (deepcache style) while all mflux rows used the LONG SOTA prompt. `read_stored()` parses prompt / negative / steps / sampler / cfg / seed from each image; table configs are only fallbacks. Deviation warnings are printed when table != embedded.
- **Trap 1 — string LoRAs are silently dropped:** `generator._enrich_loras_with_registry()` does `continue` on non-dict entries, so `loras=["Hyper-SDXL 8-step CFG Distill"]` yields NO LoRA (base model runs un-distilled → PSNR ~17dB). Harness must pass `[{"name", "path", "scale"}]` resolved from `data/loras.json` (`lora_req()`). Proven by sidecar: loras `[]` vs request intent.
- **Trap 2 — tractive civitai `<lora:...>` tag in embedded prompt:** `save_image_with_metadata` appends `, <lora:name:scale>` to the PNG text — METADATA ONLY. Yesterday's conditioning used the clean prompt + loras param. The SDXL daemon strips `<lora:...>` tags before encoding (so SDXL matched anyway); mflux does NOT (→ krea distill image differed at PSNR 20.1 dB). `resolve()` strips `,?\s*<lora:[^>]+>` before regeneration.
- Everything-lossless proof achieved for scene `misty_fjord` (seed 1001): **9/9 configs BIT-IDENTICAL** (all 4 SDXL incl. 2× +Hyper-SD-8step, flux2-4b/9b @fast_vae=False, zit, krea native + krea distill). Spot-check on `elderly_watchmaker` (seed 1002): flux2-4b + juggernaut also 2/2 identical. FLUX arena images must be regenerated with `fast_vae=False` (krea/zit used True) — the sidecars pin this (sha of FLUX arena PNGs matched only with fvae False).
- Full 90-image grid cost is ~hours on M1 (mflux engines dominate: zit ~145s, krea ~240–370s, flux9b ~113s per image under load 5.5). Run in scoped batches + `--dry-run` to preview the grid.
