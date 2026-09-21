# MLX-DIFFUSION

![MLX-DIFFUSION on MacBook Pro](frontend/src/assets/MLX-DIFFUSION_ON_MBP.png)

**A private, on-device image generation studio for Apple Silicon.**
Benchmarked and tuned on a **2021 MacBook Pro M1 with 16 GB of Unified Memory** — no cloud account, no upload, no GPU farm. Your prompts and your images never leave the machine.

---

## Local by design — your privacy on your silicon

Every part of the pipeline runs from your Mac's own memory:

- **Generation runs on the Metal GPU** via MLX (`mflux`), the Apple-silicon-native model framework. Model weights are quantized to 4-bit so a 3–13 billion parameter diffusion model fits in the 16 GB unified memory of an M1 MacBook Pro.
- **The prompt enhancer is a local LLM too** — a Qwen2.5-0.5B-Instruct 4-bit model (~350 MB RAM, sub-second latency) rewrites your prompts *on-device*.
- **No telemetry, no API keys, no network calls at generation time.** The studio even works fully offline once the weights are cached in your local Hugging Face cache.
- **Install from an already-downloaded copy** — register any model tree already on your disk (folder or HF cache) and it loads directly, nothing is re-downloaded or copied.

The M1 (2021) runs this comfortably because of unified memory: the CPU and GPU share one pooled 16 GB pool, so the whole quantized model stays resident and there is no CPU-GPU copying — exactly what MLX exploits.

---

## Engines included

| Engine | Size (quantized) | Sweet spot | Speed on M1 (16 GB) |
|---|---|---|---|
| **FLUX.2-klein 4B** | 4-bit | 4 steps, guidance 1.0 | ~90 s @ 768×768 · ~120–140 s @ 1024×1024 |
| **Juggernaut XL Lightning** (SDXL, distilled) | 4-bit | 4 steps, `euler_trailing` | ~15–20 s @ 1024×1024 (with TAESD) |
| **Krea 2 Turbo 13B** | 4-bit | 8 steps native · 4 steps with the distill LoRA | ~140 s (4-step) · ~250 s (8-step) @ 512×512 |
| **Z-Image Turbo 6B** | 4-bit | fast sketches, 16:9 | ~40 s @ 1024×1024 |

All four are quantized and memory-tuned for 16 GB unified memory — switching engines swaps weights on demand instead of keeping everything resident.

---

## Feature highlights

- **✨ Prompt Enhancer** — local 4-bit LLM that rewrites prompts per-engine, preserves your LoRA trigger words verbatim (with a deterministic re-insertion guarantee), and offers editable per-engine system prompts in **text or JSON mode**.
- **Multi-LoRA** — FLUX.2 and SDXL LoRA support, drag-and-drop `.safetensors`, rank-concat stacking for SDXL, and a built-in **Civitai importer** with live progress, speed, and cancel.
- **In-context multi-reference** — condition FLUX.2 on 1–10 reference images by referencing them as `Image 1`, `Image 2`, … in the prompt.
- **Exact color control** — strict `#HEX` palette matching with an in-app visual palette.
- **Civitai-compliant metadata** — every PNG embeds prompt, seed, steps, sampler, CFG, checkpoint and LoRA hashes in `tEXt` + EXIF, fully parseable by Civitai's uploader.
- **Split-screen studio & gallery** — left form / right interactive canvas, lazy-loaded gallery with tags, text search and one-click "Use as Reference", plus a local **2x/4x upscaler** (Lanczos) and **AI neural 2x** (SeedVR2).

---

## Quick start

From the repo root:

```bash
./run.sh        # one command: backend (8001) + frontend (5174), opens the browser
```

Development mode (2 terminals):

```bash
./dev-backend.sh                         # FastAPI, port 8001, hot reload
cd frontend && npm run dev               # Vite, port 5174, hot reload
```

Free stuck services: `lsof -ti :8001,5174 | xargs kill -9`
Ports are fixed — **8001** and **5174** (8000 and 5173 belong to other local tools).

> Full guide (models, tuning rules, troubleshooting): [`readme.txt`](readme.txt)

---

## Tech stack

- **Backend** — Python / FastAPI, worker thread + FIFO queue, MLX inference via `mflux` and a native MLX SDXL daemon
- **Frontend** — React 19 + Vite
- **Engines** — FLUX.2-klein 4B, Juggernaut XL Lightning (SDXL), Krea 2 Turbo, Z-Image Turbo
- **Cleanup** — idle watchdogs release the ~10 GB resident pipeline 5 min after the last generation

---

## License

[MIT](LICENSE) — covers this project's source code only. Model weights carry their own terms (e.g. FLUX.2-klein is Black Forest Labs Non-Commercial); checkpoints are not covered by this license.