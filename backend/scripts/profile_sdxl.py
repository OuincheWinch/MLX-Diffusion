#!/usr/bin/env python3
"""SDXL (Juggernaut XL Lightning) first baseline — exercises the app's real
subprocess engine via generator.generate() so the standard data/generated
artifacts ({id}.png, {id}.json, {id}_thumb.png) are produced, then records the
run to test/manifest.json alongside a test/ PNG copy.

Usage:
  ./venv/bin/python backend/scripts/profile_sdxl.py
"""
import json
import os
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
MANIFEST = TEST_DIR / "manifest.json"

import generator

PROMPT = (
    "a detailed imperial stormtrooper standing in a scenic swiss alpine landscape, "
    "green valley, snow-capped mountains, wooden chalet, cinematic lighting"
)
W, H, STEPS = 1024, 1024, 4
SEED = 1337


def _sysload() -> float:
    try:
        return float(os.popen("uptime").read().split()[-3].strip(","))
    except Exception:
        return 0.0


if __name__ == "__main__":
    print(f"[sys] {os.popen('uptime').read().strip()}", flush=True)
    t0 = time.perf_counter()
    meta = generator.generate(
        prompt=PROMPT,
        width=W,
        height=H,
        steps=STEPS,
        guidance=1.0,
        seed=SEED,
        quantization=4,
        model="juggernaut-xl-lightning",
        sampler="euler_trailing",
        progress_cb=lambda *a, **k: None,
        fast_vae=True,
    )
    wall = time.perf_counter() - t0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    img_name = f"sdxl_lightning_{W}x{H}_q4_s{SEED}_{stamp}.png"
    img_path = TEST_DIR / img_name
    gen_dir = BACKEND_DIR / "data" / "generated"
    src = gen_dir / meta["file"]
    if src.exists():
        shutil.copy2(src, img_path)
    data_artifacts = {
        "png": f"{meta['id']}.{meta.get('format', 'png')}",
        "json": f"{meta['id']}.json",
        "thumb": f"{meta['id']}_thumb.png",
    }
    present = {k: (gen_dir / v).exists() for k, v in data_artifacts.items()}

    runs = []
    if MANIFEST.exists():
        try:
            runs = json.loads(MANIFEST.read_text())
            if not isinstance(runs, list):
                runs = []
        except Exception:
            runs = []
    runs.append({
        "image": img_name,
        "model": "juggernaut-xl-lightning",
        "variant": "e2e-sdxl-daemon-app-path",
        "width": W,
        "height": H,
        "steps": STEPS,
        "seed": SEED,
        "guidance": 1.0,
        "sampler": meta.get("sampler", "euler_trailing"),
        "quantization_bits": 4,
        "fast_vae": True,
        "wired_limit_gb": generator._wired_limit_bytes() // (1 << 30),
        "prompt": PROMPT,
        "system_load": _sysload(),
        "wall_clock": round(wall, 3),
        "generation_time": meta.get("generation_time"),
        "data_generated": {"artifacts": data_artifacts, "present": present},
        "created_at_utc": stamp,
        "pipeline": "generator.generate() -> sdxl subprocess daemon incl. save",
    })
    MANIFEST.write_text(json.dumps(runs, indent=2, ensure_ascii=False))

    print(f"\n== SDXL Lightning | {W}x{H} | {STEPS} steps | seed {SEED} ==", flush=True)
    print(f"  wall {wall:6.2f}s | generation_time={meta.get('generation_time')}s", flush=True)
    print(f"  data/generated: {data_artifacts} -> {present}", flush=True)
    print(f"  image saved : {img_path}", flush=True)
    print(f"  manifest run(s) -> {MANIFEST}", flush=True)