"""Per-job subprocess engine for Qwen-Image 2.1 (mflux/MLX).

Spawns one fresh, isolated interpreter per generation so the long-lived API worker's
process-global state can never corrupt denoising (fresh processes are provably clean).
Speaks JSON-lines on stdout: progress/phase updates and one final result dict.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLX_KREA_WIRED_LIMIT_GB", "9")
import mlx.core as mx
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

MODEL = "mlx-community/Qwen-Image-2.1-MLX-4bit"
QWEN_SCHEDULERS = {"linear": "linear", "euler": "flow_match_euler_discrete"}


def _wired_budget():
    try:
        d = mx.device_info()
        cap = d.get("max_recommended_working_set_size") or 0
        mem = d.get("memory_size", 0)
        if mem > 0:
            budget = min(int(mem * 0.68), 9 * (1 << 30))
            if cap > 0:
                budget = min(budget, cap)
            return budget
    except Exception:
        pass
    return 9 * (1 << 30)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _dump_env_if_requested():
    path = os.environ.get("QWEN_ENGINE_DUMP_ENV")
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(json.dumps({k: str(v) for k, v in sorted(os.environ.items())}, indent=1))
    except Exception:
        pass


def _fingerprint(name, arr):
    try:
        import numpy as np
        mx.eval(arr)
        dt = None
        if getattr(arr, "dtype", None) is not None:
            dt = str(arr.dtype)
        if dt and dt.endswith("16"):
            arr = arr.astype(mx.float32)
        a = np.asarray(arr).reshape(-1)
        emit({"diag": name, "shape": list(getattr(arr, "shape", [])),
              "mean": float(a.mean()), "std": float(a.std()),
              "l1": float(np.abs(a).mean()),
              "min": float(a.min()), "max": float(a.max()),
              "first12": [float(x) for x in a[:12]]})
    except Exception as e:
        emit({"diag": name, "error": str(e)})


class _StepCB:
    def __init__(self, phase_cb=None):
        self._phase_cb = phase_cb

    def call_before_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("compiling", "Compiling Metal graph & encoding prompt...")

    def call_in_loop(self, t, **kwargs):
        emit({"progress": {"step": t + 1}})
        if self._phase_cb:
            self._phase_cb("generating", f"Denoising step {t + 1}...")

    def call_after_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("saving", "Decoding VAE latents & saving image...")


def main() -> int:
    _dump_env_if_requested()
    try:
        raw = sys.argv[1]
        if raw.lstrip().startswith("{"):
            req = json.loads(raw)
        else:
            req = json.loads(Path(raw).read_text())
    except Exception as e:
        emit({"error": f"bad request: {e}"})
        return 2

    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
    set_limit(_wired_budget())

    prompt = req["prompt"]
    negative = req.get("negative_prompt") or ""
    guidance = req.get("guidance")
    if guidance is None:
        guidance = 1.0
    eff_guidance = guidance
    if negative and eff_guidance <= 1.0:
        eff_guidance = 3.0
    scheduler = QWEN_SCHEDULERS.get((req.get("sampler") or "linear").lower(), "linear")
    seed = req.get("seed")
    if seed is None:
        seed = int(time.time())
    refs = [Path(p) for p in (req.get("ref_paths") or []) if p]

    try:
        t0 = time.time()
        pipe = QwenImage21(quantize=4, model_path=req.get("model_path") or MODEL)
        load_time = time.time() - t0

        if os.environ.get("QWEN_ENGINE_DIAG"):
            from mflux.models.common.config.config import Config as _Config
            from mflux.models.common.config import ModelConfig as _MC
            from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator as _LC
            from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder as _PE
            te = pipe.text_encoder
            _tok = pipe.tokenizers["qwen21"]
            _ids = mx.array(_tok.tokenize(prompt).input_ids).reshape(1, -1)
            _tok_emb = te.embed_tokens(_ids.astype(mx.int32))
            mx.eval(_tok_emb)
            _fingerprint("tok_emb_for_tokens", _tok_emb)
            _lat = _LC.create_noise(seed, int(req["height"]), int(req["width"]))
            _lat = _lat.astype(_MC.precision)
            mx.eval(_lat)
            _fingerprint("noise", _lat)
            _emb, _msk = _PE.encode_prompt(
                prompt=prompt,
                prompt_cache=pipe.prompt_cache,
                tokenizer=pipe.tokenizers["qwen21"],
                text_encoder=pipe.text_encoder,
            )
            mx.eval(_emb)
            _fingerprint("text_embeds", _emb)
            mx.eval(_msk)
            _fingerprint("text_mask", _msk.astype(mx.float32))
            _cfg = _Config(
                width=int(req["width"]), height=int(req["height"]), guidance=eff_guidance,
                scheduler=scheduler, image_path=None, image_strength=None,
                model_config=_MC.qwen_image_21(), num_inference_steps=int(req["steps"]),
            )
            _t0 = next(iter(_cfg.time_steps))
            _o = pipe.transformer(
                t=_t0, config=_cfg,
                hidden_states=_cfg.scheduler.scale_model_input(_lat, _t0),
                encoder_hidden_states=_emb, encoder_hidden_states_mask=_msk,
            )
            mx.eval(_o)
            _fingerprint("transformer_first_step_out", _o)

        def phase(ph, detail=""):
            emit({"phase": ph, "detail": detail})

        if req.get("phase_cb"):
            phase("loading_model", "Loading Qwen-Image-2.1 4-bit pipeline into unified memory...")

        fast_vae = bool(req.get("fast_vae", True))
        if not fast_vae and hasattr(pipe, "vae"):
            pipe.vae.decoder = mx.compile(pipe.vae.decoder)

        step_cb = _StepCB(phase_cb=(lambda ph, d: phase(ph, d)) if req.get("phase_cb") else None)
        pipe.callbacks.in_loop = [step_cb]
        pipe.callbacks.before_loop = [step_cb]
        pipe.callbacks.after_loop = [step_cb]

        t1 = time.time()
        out = pipe.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=int(req["steps"]),
            height=int(req["height"]),
            width=int(req["width"]),
            guidance=eff_guidance,
            image_path=refs[0] if refs else None,
            image_strength=req.get("image_strength"),
            negative_prompt=negative or None,
            scheduler=scheduler,
        )
        infer_time = round(time.time() - t1, 2)
        out.image.save(req["dest"])
        emit({
            "generation_time": infer_time,
            "load_time": round(load_time, 2),
            "steps": int(req["steps"]),
            "sampler": scheduler,
        })
        return 0
    except Exception as e:
        try:
            import traceback
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        emit({"error": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())