# Codebase Review — Findings (2026-08-23)

> **Status: ALL FIXED (2026-08-23).** Backend rewritten with path validation, per-job
> cancel events, atomic writes, worker hardening; frontend races/polling/UI fixes applied.
> Verified: backend imports + endpoint smoke tests pass; `npm run lint` (pre-existing
> warnings only) and `npm run build` pass.

Reviewed: `backend/main.py`, `backend/generator.py`, `backend/bench.py`, frontend (`src/App.jsx`, components). Full detail from subagent reviews; this is the consolidated tracker.

## 🔴 HIGH — Bugs / Security

| # | Area | Issue | Fix |
|---|------|-------|-----|
| H1 | backend/main.py:222+ | **Path traversal** — `image_id` used unsanitized in file paths (read/delete arbitrary files via `/api/images/...`) | Validate `^[A-Za-z0-9_-]+$` or `resolve().is_relative_to(GENERATED_DIR)` |
| H2 | backend/main.py:333 | **Arbitrary LoRA registration** — any absolute path accepted; CORS wide open enables drive-by abuse | Restrict to `LORA_FILES_DIR`, tighten CORS to `localhost:5173` |
| H3 | backend/main.py:311-324 | **Upload validation gaps** — unsanitized `name` query param, `filename=None` crash, no size cap (disk-fill DoS) | Sanitize name, guard None, max-size limit |
| H4 | backend/main.py:165-176 | **Cancel race** — job state transitions unsynchronized between queue and worker; cancelled queued jobs leak in JOBS | Lock state transitions; re-check status in worker |
| H5 | backend/generator.py:77 | **Global cancel event** — cancelling kills the wrong job when jobs are queued serially | Per-job `threading.Event` passed to `generate()` |

## 🟠 MEDIUM

| # | Area | Issue | Fix |
|---|------|-------|-----|
| M1 | main.py:179-217 | Gallery rescans/parses every JSON per request; unvalidated `page`/`limit` | In-memory index; clamp params |
| M2 | main.py:242-250, 290 | Non-atomic JSON writes (tags, loras registry) → truncation/lost updates | Temp file + `os.replace` |
| M3 | main.py:53-117 | Worker thread can die on exception outside inner try → server stops processing forever | Wrap whole loop body |
| M4 | generator.py:336-343 | Thumbnail race + unclosed `Image.open` handle | Atomic replace + context manager |
| M5 | main.py:42 | `guidance > 1.0` accepted despite klein having no negative prompt support | Clamp/reject for non-negative models |
| M6 | main.py:135-138 | Relative LoRA paths bypass validation | Require path under LORA_FILES_DIR |
| M7 | main.py:269-287 | GET /api/loras mutates disk state (auto-import) → races with uploads | Discover at startup/upload only |
| M8 | frontend GenerateForm.jsx | Cancel-then-submit race; polling fragile; out-of-order gallery responses | AbortController per request; sequence guards |
| M9 | frontend App.jsx | Hardcoded `API_BASE = http://localhost:8001` | Centralize in env/config |

## 🟡 LOW (selected)

- JOBS dict grows forever (memory leak) — TTL/cap
- 8-hex-char IDs risk silent collision overwrites — use full uuid
- `/static` mounts entire `data/` incl. LoRA files — mount `generated/` only
- Tracebacks leaked in error field — strip in prod
- Seed auto-increment silently mutates user input after run
- `fmt()` can show "1m 60s"; progress bar NaN if steps=0
- Modal `<a><button>` invalid nesting; no clipboard feedback; TagEditor not keyed by image id
- No tests anywhere (frontend or backend) — polling logic most at risk
- Prompt cache: crude eviction, negative entries bypass cap — use LRU

## ✅ Good

- Atomic image saving with retry (generator.py)
- Wired-memory-limit handling with restore in finally
- Pydantic bounds on generation params

## Suggested fix order

1. H1 (path traversal) — one-line regex, huge security win
2. H2/H3 + CORS/static scope
3. H5/H4 (cancel correctness)
4. M2/M3 (data integrity + worker resilience)
5. Frontend polling/races (M8)

---

# Codebase Deep Audit — Comprehensive Findings (2026-09-05)
**Frameworks & Standards:** `Principles.md` & `ELON.md` (5-Step Engineering Algorithm)  
**Scope:** `backend/main.py`, `backend/generator.py`, `backend/sdxl_engine.py`, `frontend/src/`

## 🔴 P0 — Critical Vulnerabilities (Immediate Crashes, Metal OOM, Protocol Corruption)

| # | Area | File:Line | Issue | Root Cause & Impact | Recommended Fix |
|---|------|-----------|-------|---------------------|-----------------|
| **P0-1** | Generator | `backend/generator.py:461-462` | **NameError: name 'ROOT' is not defined** | `(ROOT / p).exists()` crashes immediately when relative reference images are submitted | Define `ROOT = Path(__file__).resolve().parent.parent` |
| **P0-2** | Generator | `backend/generator.py:269, 495` | **`_update_pipeline_lora_scales` is NEVER called** | Dynamic LoRA scaling in memory is 100% dead code; tweaking LoRA sliders has zero effect on weights | Call `_update_pipeline_lora_scales` right after `_get_pipeline` before `generate_image` |
| **P0-3** | Generator | `backend/generator.py:534, 653` | **8GB PiD Decoder Memory Leak** | `release_pid_decoder()` only called on model drop; running PiD then regular FLUX keeps 8GB in RAM alongside 10GB FLUX (18GB total -> Metal paging/crash on 16GB Mac) | Explicitly release PiD decoder immediately after VAE decode or before non-PiD runs |
| **P0-4** | Generator | `backend/generator.py:660-673` | **Memory collision on SDXL -> mflux model switch** | `_kill_sdxl_daemon()` sends SIGKILL without `proc.wait()`; mflux allocates 10GB while SDXL still holds 11GB in Metal wired memory (21GB peak) | Add `_sdxl_daemon.wait(timeout=5.0)` and `mx.clear_cache()` in `_kill_sdxl_daemon` |
| **P0-5** | SDXL Engine | `backend/sdxl_engine.py:139-140` | **Fatal crash on DDIM sampler (`init_noise_sigma` missing)** | `AttributeError: 'DDIMScheduler' object has no attribute 'init_noise_sigma'`; crashing the daemon immediately on DDIM selection | Add `DDIMScheduler.init_noise_sigma = property(lambda self: 1.0)` |
| **P0-6** | SDXL Engine | `backend/sdxl_engine.py:141-146` | **Trailing timesteps broken in mlx_diffuser (`linspace` fallback)** | `EulerDiscreteScheduler` in mlx_diffuser has no branch for `"trailing"`, falling back to `linspace` (`[999, 666, 333, 0]`), conditioning UNet on timestep 0 with sigma 0.01 | Patch `EulerDiscreteScheduler.set_timesteps` to implement exact trailing calculation `round(T/N)` |
| **P0-7** | SDXL Engine | `backend/sdxl_engine.py:413-421` | **Silent rejection of Diffusers/PEFT LoRAs** | LoRAs ending with `.lora_A.weight` or `.down.weight` are ignored (`applied 0 LoRA layers`), generating without requested adapters | Support `.lora_A.weight`, `.lora_B.weight`, `.down.weight`, `.up.weight` in `load_multilora` |
| **P0-8** | Backend API | `backend/main.py:149-156, 408-419` | **Cancel status overwrite race condition** | Worker loop overwrites `"cancelled"` status back to `"done"` in `finally`, and instantiates `cancel_event` late | Instantiate `cancel_event` atomically upon transition to `"generating"`; guard `finally` against overwriting `"cancelled"` |
| **P0-9** | Backend API | `backend/main.py:144-159, 254-257` | **Permanent zombie job leak in `"generating"`** | If exception occurs before inner `try: for i in range(batch):`, worker jumps to outer `except` and status stays `"generating"` forever | Move `try...finally` around the entire job lifecycle |
| **P0-10** | Backend API | `backend/main.py:335-360, 433` | **`RuntimeError: dictionary changed size during iteration`** | Iterating over `JOBS` or `GALLERY_INDEX` during FastAPI JSON serialization while worker thread mutates progress | Perform defensive copy `dict(j["progress"])` under lock before releasing `_JOBS_LOCK` |
| **P0-11** | Frontend | `frontend/src/components/GenerateForm.jsx:591` | **Crash White Screen on Regex in Trigger Chips** | `new RegExp(tw)` crashes with SyntaxError on triggers containing brackets/parentheses (`[trigger]`, `style+punk`) | Add `escapeRegExp` helper before building RegExp |
| **P0-12** | Frontend | `frontend/src/components/GenerateForm.jsx:1057` | **Crash UI on `scale.toFixed(2)`** | `TypeError: Cannot read properties of undefined (reading 'toFixed')` if LoRA scale is null/undefined | Fallback `(Number(l.scale) || 1.0).toFixed(2)` |
| **P0-13** | Frontend | `frontend/src/components/GenerateForm.jsx:912` | **PiD Decode Schema Mismatch (HTTP 422)** | Frontend slider allows up to 2.0 while backend Pydantic limits `le=1.0`, breaking all generations with sigma > 1.0 | Clamp frontend slider to `max="1.0"` |
| **P0-14** | Frontend | `frontend/src/components/GenerateForm.jsx:422, 475` | **Destructive `jobId` overwrite in `queueNext`** | Queuing a second job immediately replaces `jobId`, dropping tracking and canvas display of the active GPU job | Maintain active job queue or guard `setJobId` |

---

## 🟠 P1 — High Severity (Data Loss, Memory Leaks, Inconsistent State)

| # | Area | File:Line | Issue | Fix |
|---|------|-----------|-------|-----|
| **P1-1** | Backend API | `backend/main.py:584-598` | **Non-atomic `loras.json` transactions** — concurrent read-modify-write overwrites user configurations | Introduce atomic `_loras_transaction()` context manager |
| **P1-2** | Backend API | `backend/main.py:833, 657` | **Zombie resurrection of deleted LoRAs** — `delete_lora` only drops JSON entry, `_discover_local_loras` re-adds on restart | Delete physical file in `delete_lora` and purge missing files during scan |
| **P1-3** | Backend API | `backend/main.py:755-773` | **Uncapped upload DoS in `/api/uploads`** — no file size limit on reference images | Enforce `MAX_IMAGE_UPLOAD_BYTES = 50MB` and validate format via Pillow |
| **P1-4** | Backend API | `backend/main.py:756, 777` | **Blocking synchronous I/O in `async def` routes** — freezes FastAPI event loop during large file uploads | Change to standard `def` to run in threadpool or use `anyio.to_thread` |
| **P1-5** | Generator | `backend/generator.py:617-631` | **SDXL watchdog race condition** — `_kill()` runs outside `_lock`, killing daemon while new request writes to stdin | Wrap `_kill()` inside `with _lock:` |
| **P1-6** | Generator | `backend/generator.py:714, 816` | **SDXL watchdog not re-armed on error/cancel** — daemon stays in RAM forever | Ensure `_arm_sdxl_watchdog()` in `finally:` block |
| **P1-7** | Generator | `backend/generator.py:948-953` | **Pillow crash on RGBA to JPEG conversion** (`cannot write mode RGBA as JPEG`) | Convert to RGB before saving JPEG: `image.convert("RGB")` |
| **P1-8** | Generator | `backend/generator.py:457-458` | **JPEG reference images fail lookup** — only `.png` tested in `GENERATED_DIR` | Test `.jpeg`, `.jpg`, and `.png` |
| **P1-9** | SDXL Engine | `backend/sdxl_engine.py:80, 87` | **Stdout IPC pollution by plain text logs** — breaks parent JSON parser | Redirect all non-IPC logs to `sys.stderr` |
| **P1-10** | SDXL Engine | `backend/sdxl_engine.py:504, 579` | **Stdout thread collision (`_poll` vs result line)** | Guard `sys.stdout.write` with an IPC output lock |
| **P1-11** | SDXL Engine | `backend/sdxl_engine.py:552` | **Missing `mx.clear_cache()` on exception in `generate()`** | Add `mx.clear_cache()` and `gc.collect()` in `finally` |
| **P1-12** | SDXL Engine | `backend/sdxl_engine.py:545` | **Abusive Tiled VAE at 1024×1024** — runs 9 tiles instead of 1 pass, wasting 6.5s per image | Change condition to `(width * height > 1024 * 1024)` |
| **P1-13** | Frontend | `frontend/src/components/GenerateForm.jsx:339` | **Memory leak on reference image Blob URLs** | Call `URL.revokeObjectURL` on remove and unmount |
| **P1-14** | Frontend | `frontend/src/components/GenerateForm.jsx:236` | **Batch intermediate images hidden on Studio Canvas** | Display thumbnail filmstrip for batches in `ResultCanvas` |
| **P1-15** | Frontend | `frontend/src/components/GenerateForm.jsx:254` | **Partial batch results lost on cancel** (`prev ?? last`) | Use `last` if `partial_results` exists |
| **P1-16** | Frontend | `frontend/src/components/ResultCanvas.jsx:177` | **Hardcoded `.png` breaks Img2Img on JPEG images** | Use `currentImage.file || ...` |
| **P1-17** | Frontend | `frontend/src/components/GenerateForm.jsx:434` | **False positive `dirty` state during Variation** | Normalize JSON comparison keys |

---

## 🟡 P2 / P3 — Optimization & Cleanliness (ELON.md Steps 1 to 5)

- **Delete Waste (ELON Step 2)**:
  - Delete dead function `_save_image_resilient` in `generator.py:985-998`.
  - Delete unused imports `File`, `StaticFiles` in `main.py:12, 14`.
  - Delete dead CSS rules `.ref-preview`, `.ref-row`, `.trigger-reminder` in `App.css`.
  - Delete redundant local re-imports (`import mlx.core as mx`, `import math`) in inner loops.
- **Simplify & Accelerate (ELON Steps 3 & 4)**:
  - Factorize 4 duplicated image extension search loops into `find_image_file(image_id)`.
  - Enforce 16-pixel alignment in Pydantic schema for `width` and `height`.
  - Add fast CPU caching of raw LoRA weights in `sdxl_engine.py` to make scale tweaks take 50ms instead of 15s.
  - Implement PRNG `mx.random.split` in `EulerAncestralScheduler` to eliminate affine seed correlation.
  - Add `lifespan` handler in FastAPI to cleanly shut down GPU daemons on server termination.
