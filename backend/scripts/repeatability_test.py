#!/usr/bin/env python3
"""Repeatability test vs YESTERDAY'S SOTA-arena build (comparison_12_vs_4_steps).

Regenerates the exact arena grid (9 model configs x 10 canonical scenes,
512x768, seeds 1001-1010) with TODAY'S code and compares pixel-for-pixel
against the stored yesterday images. Per-config params are pinned from
yesterday's data/generated sidecars (their times byte-match the arena page):
  sdxl: euler_trailing, fast_vae=True; Hyper-SD 8-step LoRA for the two
        base-UNet configs (realvis-xl-v5, juggernaut-xi); lightning native=no lora.
  mflux: Euler (flow-match); FLUX saved with fast_vae=FALSE (sidecar proof),
         ZIT/Krea with fast_vae=True; Krea 4-step = auto krea2 distill LoRA.
Every regenerated image flows through generator.generate(), producing the
standard data/generated/{id}.png + .json + _thumb.png triple.

Usage:
  # fast authoritative smoke (2 engines, 1 scene) ~3 min
  ./venv/bin/python backend/scripts/repeatability_test.py --models flux2-klein-4b,juggernaut-xl-lightning
  # one scene across all mflux engines (zit/krea slow)
  ./venv/bin/python backend/scripts/repeatability_test.py --scenes misty_fjord --models all-mflux
  # full 90-image grid (several hours on M1)
  ./venv/bin/python backend/scripts/repeatability_test.py --models all --scenes all
"""
import argparse
import importlib.metadata
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
PROJECT_ROOT = BACKEND_DIR.parent
TEST_DIR = PROJECT_ROOT / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_ARENA = PROJECT_ROOT / "comparison_12_vs_4_steps"

from image_meta import atomic_write_json
import numpy as np
from PIL import Image
import mlx.core as mx

import generator

MODEL_CONFIGS = {
    "juggernaut-xl-lightning": {"model": "juggernaut-xl-lightning", "steps": 4, "guidance": 1.0, "sampler": "euler_trailing", "fast_vae": True, "loras": []},
    "realvis-xl-v5-lightning": {"model": "realvis-xl-v5-lightning", "steps": 6, "guidance": 1.5, "sampler": "euler_trailing", "fast_vae": True, "loras": []},
    "realvis-xl-v5":            {"model": "realvis-xl-v5", "steps": 8, "guidance": 2.0, "sampler": "euler_trailing", "fast_vae": True, "loras": ["Hyper-SDXL 8-step CFG Distill"]},
    "juggernaut-xi":            {"model": "juggernaut-xi", "steps": 8, "guidance": 2.0, "sampler": "euler_trailing", "fast_vae": True, "loras": ["Hyper-SDXL 8-step CFG Distill"]},
    "flux2-klein-4b":           {"model": "flux2-klein-4b", "steps": 4, "guidance": 1.0, "sampler": None, "fast_vae": False, "loras": []},
    "flux2-klein-9b":           {"model": "flux2-klein-9b", "steps": 4, "guidance": 1.0, "sampler": None, "fast_vae": False, "loras": []},
    "z-image-turbo":            {"model": "z-image-turbo", "steps": 8, "guidance": 1.0, "sampler": None, "fast_vae": True, "loras": []},
    "krea2-turbo":              {"model": "krea2-turbo", "steps": 4, "guidance": 1.0, "sampler": None, "fast_vae": True, "loras": ["krea2_turbo_4step_rank_64_lora_latest"]},
    "krea2-turbo-8step":        {"model": "krea2-turbo", "steps": 8, "guidance": 1.0, "sampler": None, "fast_vae": True, "loras": []},
}

# string names in MODEL_CONFIGS are resolved to request dicts (path+scale) here;
# generator._enrich_loras_with_registry() drops non-dict entries, and the SDXL
# daemon (sdxl_engine.load_multilora) raises without a resolvable "path".
LORA_SCALE = 1.0


def lora_req(names: list[str]) -> list[dict]:
    if not names:
        return []
    registry = json.loads((BACKEND_DIR / "data" / "loras.json").read_text())
    reqs = []
    for n in names:
        match = next((r for r in registry if r.get("name") == n or Path(r.get("path", "")).name == n), None)
        if not match:
            raise SystemExit(f"lora '{n}' not found in loras.json registry")
        path = Path(match["path"])
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        reqs.append({"name": n, "path": str(path.resolve()), "scale": LORA_SCALE})
    return reqs


def _dependency_versions() -> dict:
    names = ("mlx", "mflux", "mlx-lm", "mlx-diffuser", "numpy", "pillow")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _memory_snapshot() -> dict:
    snapshot = {}
    for key, name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        getter = getattr(mx, name, None) or getattr(mx.metal, name, None)
        snapshot[key] = int(getter()) if getter else None
    return snapshot


def _reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if reset:
        reset()


def _phase_timings(events: list[tuple[str, float]], start: float, end: float) -> dict:
    if not events:
        return {"unattributed": round(max(0.0, end - start), 3)}
    totals = {"unattributed_before_first_phase": round(max(0.0, events[0][1] - start), 3)}
    for index, (phase, timestamp) in enumerate(events):
        next_timestamp = events[index + 1][1] if index + 1 < len(events) else end
        duration = max(0.0, next_timestamp - timestamp)
        totals[phase] = round(totals.get(phase, 0.0) + duration, 3)
    return totals


def _pipeline_cache_state(cfg: dict, loras: list[dict]) -> str:
    if cfg["model"] in ("juggernaut-xl-lightning", "realvis-xl-v5-lightning", "realvis-xl-v5", "juggernaut-xi"):
        daemon = getattr(generator, "_sdxl_daemon", None)
        current = str(getattr(generator, "_current_sdxl_model", "") or "")
        return "warm" if daemon is not None and daemon.poll() is None and cfg["model"] in current else "cold"
    expected_key = (cfg["model"], 4, tuple(sorted(l["path"] for l in loras)), "standard")
    return "warm" if getattr(generator, "_current_pipeline_key", None) == expected_key else "cold"


def parse_scenes(arena: Path):
    """Extract SOTA_PROMPTS (id, title, seed, prompt) from the arena index.html."""
    html = (arena / "index.html").read_text(encoding="utf-8")
    block = html.split("const SOTA_PROMPTS", 1)[1].split("const DEEPCACHE_BENCH", 1)[0]
    scenes = []
    for m in re.finditer(r'"id"\s*:\s*"(\w+)"\s*,\s*"title":\s*"([^"]*)"\s*,\s*"icon":\s*"([^"]*)"\s*,\s*"seed"\s*:\s*(\d+)\s*,\s*"prompt":\s*"((?:[^"\\]|\\.)*)"', block):
        scenes.append({"id": m.group(1), "title": m.group(2), "icon": m.group(3),
                       "seed": int(m.group(4)), "prompt": json.loads('"' + m.group(5) + '"')})
    return scenes


def parse_model_times(arena: Path):
    """arena per-scene times by model key -> {scene_id: seconds} for speed reproducibility.

    SOTA_MODELS is JSON-with-trailing-comma; each scene block is
        "scene_id": {\n  "time": "12.3s",\n  "img": "images/model_scene.png"
    i.e. "time" and "img" live on separate lines, so per-model regex over the
    whole block is used instead of a line scanner.
    """
    html = (arena / "index.html").read_text(encoding="utf-8")
    block = html.split("const SOTA_MODELS", 1)[1].split("const SOTA_PROMPTS", 1)[0]
    times = {}
    model_re = re.compile(r'"([\w-]+)":\s*\{\s*"id":.*?(?="[\w-]+":\s*\{\s*"id":|\Z)', re.S)
    scene_re = re.compile(r'"(\w+)":\s*\{\s*"time":\s*"([\d.]+)s",\s*"img":\s*"images/([\w-]+\.png)"')
    for m in model_re.finditer(block):
        model = m.group(1)
        if model not in MODEL_CONFIGS:
            continue
        body = m.group(0)
        scenes = {}
        for s in scene_re.finditer(body):
            scene, secs, img = s.group(1), float(s.group(2)), s.group(3)
            assert img == f"{model}_{scene}.png", f"{model} scene block: img {img} != {model}_{scene}.png"
            scenes[scene] = secs
        times[model] = scenes
    return times


def pixels(im) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.int16)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def read_stored(img_path: Path):
    """"Ground truth" generated yesterday is embedded in each stored PNG's
    parameters text (prompt / negative / steps / sampler / cfg / seed)."""
    d = {}
    try:
        text = Image.open(img_path).info.get("parameters") or ""
    except Exception:
        return d
    if not text:
        return d
    lines = text.splitlines()
    if lines:
        d["prompt"] = lines[0]
    nm = re.search(r"Negative prompt:\s*(.*)", text)
    if nm:
        d["negative"] = nm.group(1)
    pm = re.search(r"Steps:\s*(\d+),\s*Sampler:\s*([^,]+),\s*CFG scale:\s*([\d.]+),\s*Seed:\s*(\d+)", text)
    if pm:
        d["steps"] = int(pm.group(1))
        d["sampler"] = pm.group(2).strip()
        d["guidance"] = float(pm.group(3))
        d["seed"] = int(pm.group(4))
    return d


def resolve(cfg: dict, scene: dict, arena_img: Path):
    emb = read_stored(arena_img)
    embedded_prompt = emb.get("prompt", scene["prompt"])
    # The stored PNG embeds a Civitai-style ", <lora:name:scale>" tag that the
    # app appends at SAVE time only; conditioning used the clean prompt + loras
    # param. mflux does NOT strip the tag (SDXL does), so strip it here to keep
    # conditioning identical to yesterday's run.
    clean_prompt = re.sub(r",?\s*<lora:[^>]+>", "", embedded_prompt)
    stripped = clean_prompt != embedded_prompt
    use = {
        "prompt": clean_prompt,
        "negative": emb.get("negative", ""),
        "steps": emb.get("steps", cfg["steps"]),
        "guidance": emb.get("guidance", cfg["guidance"]),
        "sampler": emb.get("sampler", cfg["sampler"]),
        "seed": emb.get("seed", scene["seed"]),
    }
    deviations = []
    if emb:
        for key in ("steps", "guidance", "sampler", "seed"):
            if key not in emb:
                continue
            table_v = cfg.get(key, scene.get(key))
            if key == "sampler" and table_v is None:
                continue  # mflux default = "Euler" (matches embedded)
            if emb[key] != table_v:
                deviations.append(f"{key} table={table_v} vs embedded={emb[key]}")
        if "prompt" in emb and emb["prompt"] != scene["prompt"]:
            deviations.append("prompt table!=embedded (SDXL rows reused short bench prompt)")
        if stripped:
            deviations.append("prompt had trailing <lora:> civitai tag stripped (metadata-only in yesterday's save)")
    return use, deviations


def run_one(cfg: dict, scene: dict, arena_img: Path, out_dir: Path, use: dict):
    step_ts = {}
    phase_events = []
    loras = lora_req(cfg["loras"])
    pipeline_cache = _pipeline_cache_state(cfg, loras)

    def _cb(step, *_a, **_k):
        step_ts[step] = time.perf_counter()

    def _phase(phase, detail=None):
        phase_events.append((phase, time.perf_counter()))

    _reset_peak_memory()
    memory_before = _memory_snapshot()
    t0 = time.perf_counter()
    meta = generator.generate(
        prompt=use["prompt"],
        negative_prompt=use["negative"],
        width=512,
        height=768,
        steps=use["steps"],
        guidance=use["guidance"],
        seed=use["seed"],
        quantization=4,
        model=cfg["model"],
        sampler=use["sampler"],
        fast_vae=cfg["fast_vae"],
        loras=loras,
        progress_cb=_cb,
        phase_cb=_phase,
    )
    gen_end = time.perf_counter()
    gen_s = round(gen_end - t0, 2)
    phases = _phase_timings(phase_events, t0, gen_end)
    memory_after = _memory_snapshot()
    s_per_iter = "—"
    ks = sorted(step_ts)
    if len(ks) >= 2:
        diffs = [step_ts[ks[i + 1]] - step_ts[ks[i]] for i in range(len(ks) - 1)]
        if diffs:
            s_per_iter = round(sum(diffs) / len(diffs), 2)
    today_p = out_dir / f"today_{meta['model'].replace('/','_')}_{scene['id']}_s{use['seed']}.png"
    shutil.copy2(BACKEND_DIR / "data" / "generated" / meta["file"], today_p)

    stored = pixels(Image.open(arena_img))
    fresh = pixels(Image.open(today_p))
    same = bool((stored == fresh).all())
    sdxl_model = cfg["model"] in (
        "juggernaut-xl-lightning",
        "realvis-xl-v5-lightning",
        "realvis-xl-v5",
        "juggernaut-xi",
    )
    res = {
        "model": cfg["model"], "scene": scene["id"], "seed": use["seed"],
        "steps": use["steps"], "guidance": use["guidance"], "sampler": use["sampler"] or "model-default",
        "fast_vae": cfg["fast_vae"], "prompt_used": use["prompt"][:90],
        "pipeline_cache": pipeline_cache,
        "dependencies": _dependency_versions(),
        "memory_scope": "api-parent-only; SDXL subprocess memory unavailable" if sdxl_model else "api-process",
        "memory": {"before": memory_before, "after": memory_after},
        "phase_sequence": [phase for phase, _ in phase_events],
        "phases": phases,
        "generation_time_s": gen_s, "s_per_iter": s_per_iter,
        "timing_scope": "generator.generate wall time with recorded pipeline cache state",
        "today_png": str(today_p), "data_generated": meta["file"],
        "verdict": "BIT-IDENTICAL" if same else "DIFFERS",
        "psnr_db": round(psnr(stored, fresh), 2) if not same else None,
        "max_abs_diff": int(np.abs(stored - fresh).max()) if not same else 0,
        "byte_identical": bool(open(arena_img, "rb").read() == open(today_p, "rb").read()),
    }
    return res


def _pct(today, arena) -> str:
    if arena is None:
        return "—"
    v = (today - arena) / arena * 100
    return f"{'+' if v >= 0 else ''}{v:.1f}%"


def summarize(results: list[dict], out_dir: Path, stamp: str):
    """Print + write a per-model speed table (today / arena / Δ% / s-it + mean)."""
    by_model = {}
    order = []
    for r in results:
        m = r["model"]
        if m not in by_model:
            by_model[m] = []
            order.append(m)
        by_model[m].append(r)
    lines = [f"# Repeatability speed summary — {stamp}", ""]
    for m in order:
        rs = sorted(by_model[m], key=lambda r: r["scene"])
        cfg_parts = [
            f"`{r['sampler']}` {r['steps']} steps · cfg {r['guidance']} · fvae {r['fast_vae']}"
        ]
        c0 = rs[0]
        print(f"\n== {m} | {cfg_parts[0]} ==", flush=True)
        lines.append(f"## {m} — {c0.get('sampler')} {c0.get('steps')} steps, cfg {c0.get('guidance')}, fast_vae {c0.get('fast_vae')}")
        lines.append("")
        lines.append("| scene | today (s) | cache | arena (s) | Δ% | s/it | verdict |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rs:
            line = (f"| {r['scene']} | {r['generation_time_s']} | {r.get('pipeline_cache', 'unknown')} | {r.get('arena_time_s', '—')} | "
                    f"{_pct(r['generation_time_s'], r.get('arena_time_s'))} | {r.get('s_per_iter', '—')} | {r['verdict']} |")
            print(f"  {r['scene']:22s} today {r['generation_time_s']:>7}s cache {r.get('pipeline_cache', 'unknown'):4s} arena {r.get('arena_time_s', '—')}  "
                  f"d% {_pct(r['generation_time_s'], r.get('arena_time_s'))}  s/it {r.get('s_per_iter', '—')}  {r['verdict']}", flush=True)
            lines.append(line)
        ok = [r for r in rs if r.get("arena_time_s") is not None]
        for cache in sorted({r.get("pipeline_cache", "unknown") for r in ok}):
            cache_rows = [r for r in ok if r.get("pipeline_cache", "unknown") == cache]
            mean_t = sum(r["generation_time_s"] for r in cache_rows) / len(cache_rows)
            mean_a = sum(r["arena_time_s"] for r in cache_rows) / len(cache_rows)
            it = [r.get("s_per_iter") for r in cache_rows if r.get("s_per_iter") not in (None, "—")]
            mean_it = f"{sum(it) / len(it):.2f}" if it else "—"
            lines.append(f"| **mean {cache}** | **{mean_t:.2f}** | **{cache}** | **{mean_a:.2f}** | **{_pct(mean_t, mean_a)}** | **{mean_it}** | |")
            print(f"  {'mean ' + cache:22s} today {mean_t:>7.2f}s arena {mean_a:.2f}  d% {_pct(mean_t, mean_a)}  s/it {mean_it}", flush=True)
        lines.append("")
    summary_md = out_dir / f"repeatability_summary_{stamp}.md"
    summary_md.write_text("\n".join(lines))
    print(f"\nsummary: {summary_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", type=Path, default=DEFAULT_ARENA)
    ap.add_argument("--models", default="flux2-klein-4b,juggernaut-xl-lightning",
                    help="comma list, or 'all' / 'all-mflux' / 'all-sdxl'")
    ap.add_argument("--scenes", default="misty_fjord", help="comma list, or 'all'")
    ap.add_argument("--dry-run", action="store_true", help="list the grid, do not generate")
    args = ap.parse_args()

    scenes = parse_scenes(args.arena)
    if args.scenes.lower() != "all":
        ids = [s.strip() for s in args.scenes.split(",")]
        scenes = [s for s in scenes if s["id"] in ids]
        missing = set(ids) - {s["id"] for s in scenes}
        if missing:
            print(f"[warn] unknown scene(s): {sorted(missing)}", flush=True)

    if args.models.lower() == "all":
        keys = list(MODEL_CONFIGS)
    elif args.models.lower() == "all-mflux":
        keys = ["flux2-klein-4b", "flux2-klein-9b", "z-image-turbo", "krea2-turbo", "krea2-turbo-8step"]
    elif args.models.lower() == "all-sdxl":
        keys = ["juggernaut-xl-lightning", "realvis-xl-v5-lightning", "realvis-xl-v5", "juggernaut-xi"]
    else:
        keys = [k.strip() for k in args.models.split(",")]
        missing = set(keys) - set(MODEL_CONFIGS)
        if missing:
            raise SystemExit(f"unknown model(s): {sorted(missing)}; choices: {list(MODEL_CONFIGS)}")

    times = parse_model_times(args.arena)
    grid = [(k, s) for k in keys for s in scenes]
    missing_img = []
    for k, s in grid:
        p = args.arena / "images" / f"{k}_{s['id']}.png"
        if not p.exists():
            missing_img.append(f"{k}::{s['id']}")
    if missing_img:
        print(f"[warn] {len(missing_img)} stored image(s) missing for the requested grid "
              f"(first: {missing_img[:3]})", flush=True)
        grid = [(k, s) for k, s in grid if (args.arena / "images" / f"{k}_{s['id']}.png").exists()]

    print(f"grid: {len(grid)} image(s) | scenes {len(scenes)} | models {keys}", flush=True)
    if args.dry_run:
        for k, s in grid:
            cfg = MODEL_CONFIGS[k]
            line = (f"  {k:28s} {s['id']:22s} seed {s['seed']:5d} steps {cfg['steps']:2d} "
                    f"cfg {cfg['guidance']} fvae {cfg['fast_vae']} loras {len(cfg['loras'])}")
            ago = times.get(k, {}).get(s["id"])
            if ago:
                line += f" | arena {ago}s"
            print(line, flush=True)
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = TEST_DIR / f"repeatability_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (k, s) in enumerate(grid, 1):
        cfg = MODEL_CONFIGS[k]
        arena_img = args.arena / "images" / f"{k}_{s['id']}.png"
        ago = times.get(k, {}).get(s["id"])
        use, deviations = resolve(cfg, s, arena_img)
        print(f"[{i}/{len(grid)}] {k} · {s['id']} · seed {use['seed']} · {use['steps']} steps · "
              f"cfg {use['guidance']} · {use['sampler'] or 'model-default'} · fvae {cfg['fast_vae']}"
              f" | arena {ago}s", flush=True)
        for d in deviations:
            print(f"  [meta] {d}", flush=True)
        print(f"  prompt: {use['prompt'][:78]}…", flush=True)
        try:
            r = run_one(cfg, s, arena_img, out_dir, use)
        except Exception as e:
            print(f"  !! failed: {e}", flush=True)
            r = {"model": cfg["model"], "scene": s["id"], "seed": use["seed"], "error": str(e), "verdict": "ERROR"}
        r["arena_time_s"] = ago
        r["deviations"] = deviations
        results.append(r)
        v = r["verdict"]
        extra = f" (psnr {r.get('psnr_db')} dB, max {r.get('max_abs_diff')}/255)" if r.get("psnr_db") else ""
        print(f"  => {v}{extra} | today {r.get('generation_time_s')}s", flush=True)

    report = {"created_at_utc": stamp, "grid": len(results),
              "identical": sum(1 for r in results if r["verdict"] == "BIT-IDENTICAL"),
              "differ": sum(1 for r in results if r["verdict"] == "DIFFERS"),
              "errors": sum(1 for r in results if r["verdict"] == "ERROR"),
              "results": results}
    latest = TEST_DIR / "repeatability_report.json"
    atomic_write_json(latest, report, ensure_ascii=False)
    atomic_write_json(TEST_DIR / f"repeatability_report_{stamp}.json", report, ensure_ascii=False)
    print(f"\n== VERDICT: {report['identical']}/{report['grid']} bit-identical | "
          f"{report['differ']} differ | {report['errors']} errors ==", flush=True)
    print(f"report: {latest}", flush=True)
    summarize(results, out_dir, stamp)


if __name__ == "__main__":
    main()