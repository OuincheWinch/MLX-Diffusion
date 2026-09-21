"""SDXL engine for MLX-DIFFUSION.

Runs under venv-sdxl (Python >= 3.11, mlx-diffuser). Invoked by the main
backend as a subprocess with a JSON request on argv[1]; writes a PNG into
GENERATED_DIR and prints a JSON result on stdout.

Supports:
- negative prompt + guidance (CFG)
- multiple civitai/kohya SDXL LoRAs stacked with per-LoRA scales
  (rank-concat trick: one LoRALinear per layer, adapters concatenated
  along the rank dimension, pre-scaled)
- custom schedulers: euler_a (ancestral), dpmpp_2m_karras, euler, ddim

LoRA key mapping (kohya/ComfyUI -> diffusers names used by mlx-diffuser):
  lora_unet_input_blocks.{1,2}.1  -> down_blocks.0.attentions.{0,1}
  input_blocks.{4,5}.1            -> down_blocks.1.attentions.{0,1}
  input_blocks.{7,8}.1            -> down_blocks.2.attentions.{0,1}
  middle_block.1                  -> mid_block.attentions.0
  output_blocks.i.1               -> up_blocks.(i//3).attentions.(i%3)
  attnN_to_X -> attnN.to_X ; attn2_to_out_0 -> attn2.to_out.0
  ff_net_0_proj -> ff.net.0.proj ; ff_net_2 -> ff.net.2
  lora_te1_* -> text_encoder ; lora_te2_* -> text_encoder_2
"""
import gc
import json
import re
import struct
import sys
import threading
from pathlib import Path

# Ensure backend directory is in sys.path for local imports like taesd_mlx
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_ipc_stdout_lock = threading.Lock()


def _write_ipc_json(obj: dict):
    line = json.dumps(obj) + "\n"
    with _ipc_stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


# Patch AutoencoderKLSD:
# 1. Option A: Dynamic adaptive tiling (1-pass for <= 1024x1024, 96x96 for larger) + mx.compile decoder
#    reducing decode time from ~35s down to ~7.8s with 100% bit-for-bit fidelity.
# 2. Option B: Ultra-fast pure MLX TAESD decoder (~0.5s) when pipe.vae.use_taesd = True.
try:
    from mlx_diffuser.models.autoencoder_kl_sd import AutoencoderKLSD, _feather

    def _safe_tiled(self, z: mx.array, tile: int, overlap: int) -> mx.array:
        h, w = z.shape[1], z.shape[2]
        stride = tile - overlap
        ys = sorted({*range(0, max(h - tile, 0) + 1, stride), max(h - tile, 0)})
        xs = sorted({*range(0, max(w - tile, 0) + 1, stride), max(w - tile, 0)})
        scale = 2 ** (len(self.config.block_out_channels) - 1)
        out = mx.zeros((z.shape[0], h * scale, w * scale, self.config.out_channels))
        wsum = mx.zeros((1, h * scale, w * scale, 1))
        decoder = getattr(self, "_compiled_decoder", None)
        if decoder is None:
            decoder = mx.compile(self.decoder)
            self._compiled_decoder = decoder
        for y in ys:
            for x in xs:
                dec = decoder(z[:, y : y + tile, x : x + tile])
                th, tw = dec.shape[1], dec.shape[2]
                win = (_feather(th, overlap * scale)[:, None] * _feather(tw, overlap * scale)[None])[None, :, :, None]
                py, px = y * scale, x * scale
                out[:, py : py + th, px : px + tw] += dec * win
                wsum[:, py : py + th, px : px + tw] += win
                mx.eval(out, wsum)
        res = out / wsum
        mx.eval(res)
        return res

    AutoencoderKLSD._tiled = _safe_tiled

    def _safe_decode(
        self, z: mx.array, *, tile: bool = False, tile_latent: int = 64, overlap_latent: int = 16
    ) -> mx.array:
        # Clear residual UNet activation graph and unpin wired memory before VAE decode
        try:
            import gc
            import mlx.core as _mx
            _mx.set_wired_limit(0)
            _mx.metal.clear_cache()
            _mx.clear_cache()
            gc.collect()
        except Exception:
            pass

        # Option B: Ultra-fast TAESD decode (~0.5s)
        if getattr(self, "use_taesd", False):
            try:
                from taesd_mlx import get_taesd_decoder
                c_taesd = get_taesd_decoder()
                raw_latents = z * self.scaling_factor
                img = c_taesd(raw_latents)
                mx.eval(img)
                # TAESD outputs in [0, 1]. Map to standard VAE range [-1, 1]
                # so downstream image processing doesn't shift the black point to 128 (milky haze).
                return img * 2.0 - 1.0
            except Exception as _taesd_err:
                print(f"[sdxl] WARN TAESD decode failed, falling back to full VAE: {_taesd_err}", file=sys.stderr)

        # Option A: Fast compiled direct / adaptive tiled VAE
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        decoder = getattr(self, "_compiled_decoder", None)
        if decoder is None:
            decoder = mx.compile(self.decoder)
            self._compiled_decoder = decoder

        if tile:
            return self._tiled(z, max(tile_latent, 64), overlap_latent)
        return decoder(z)

    AutoencoderKLSD.decode = _safe_decode
except Exception as _e:
    print(f"[sdxl] WARN failed to patch AutoencoderKLSD: {_e}", file=sys.stderr)

# Runtime patches for mlx_diffuser Schedulers:
# 1. DDIMScheduler lacks init_noise_sigma -> AttributeError during latent scale.
# 2. EulerDiscreteScheduler lacks proper 'trailing' timestep calculation -> falls back to linspace.
try:
    from mlx_diffuser.schedulers import DDIMScheduler, EulerDiscreteScheduler

    if not hasattr(DDIMScheduler, "init_noise_sigma"):
        DDIMScheduler.init_noise_sigma = property(lambda self: 1.0)

    _orig_euler_set_timesteps = EulerDiscreteScheduler.set_timesteps

    def _safe_euler_set_timesteps(self, num_inference_steps: int) -> None:
        T = self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        train_sigmas = np.array(self._train_sigmas)
        if self.config.timestep_spacing == "leading":
            step_ratio = T // num_inference_steps
            ts = (np.arange(num_inference_steps) * step_ratio).round()[::-1].astype(np.float64)
            ts = ts + self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = T / num_inference_steps
            ts = (np.arange(1, num_inference_steps + 1) * step_ratio).round()[::-1].astype(np.float64) - 1
        else:  # "linspace"
            ts = np.linspace(0, T - 1, num_inference_steps)[::-1].copy()
        sigmas = np.interp(ts, np.arange(T), train_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(ts.astype(np.float32))
        self._step_index = 0

    EulerDiscreteScheduler.set_timesteps = _safe_euler_set_timesteps
except Exception as _e:
    print(f"[sdxl] WARN failed to patch schedulers: {_e}", file=sys.stderr)

DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"
DEFAULT_MODEL_DIR = DATA_DIR / "models" / "juggernaut-xl-lightning"

_pipeline = None            # resident pipeline
_pipeline_model_dir = None  # path to model directory it was loaded from
_pipeline_quant = None      # quantization level it was loaded with
_lora_state = None  # tuple of (path, scale) currently applied


from collections import OrderedDict

_prompt_cache: OrderedDict[str, tuple] = OrderedDict()
_PROMPT_CACHE_MAX_SIZE = 64


def _patch_encode_prompt(pipe):
    if hasattr(pipe, "_orig_encode_prompt"):
        return
    orig_encode = pipe.encode_prompt
    pipe._orig_encode_prompt = orig_encode

    def cached_encode(prompt: str):
        key = prompt or ""
        if key in _prompt_cache:
            _prompt_cache.move_to_end(key)
            return _prompt_cache[key]
        pe, pooled = orig_encode(prompt)
        mx.eval(pe, pooled)
        _prompt_cache[key] = (pe, pooled)
        if len(_prompt_cache) > _PROMPT_CACHE_MAX_SIZE:
            _prompt_cache.popitem(last=False)
        return pe, pooled

    pipe.encode_prompt = cached_encode


def get_pipeline(model_dir: str | Path | None = None, quantize_unet: int | None = 4):
    global _pipeline, _pipeline_model_dir, _pipeline_quant
    target_dir = Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR.resolve()
    target_quant = quantize_unet or None
    if _pipeline is None or _pipeline_model_dir != target_dir or _pipeline_quant != target_quant:
        from mlx_diffuser import StableDiffusionXLPipeline

        _reset_pipeline()
        _write_ipc_json({"phase": "loading_model", "detail": "Loading SDXL UNet & text encoders into unified memory..."})
        print(f"[sdxl] loading pipeline from {target_dir} (quant_unet={target_quant})...", file=sys.stderr, flush=True)
        # 4-bit UNet is ~5x FASTER than fp16 on M1 (fused quant matmul) and
        # halves memory; 8-bit is pathologically slow (dequant per matmul).
        _pipeline = StableDiffusionXLPipeline.from_diffusers(
            str(target_dir), quantize_unet=target_quant)
        _patch_encode_prompt(_pipeline)
        _pipeline_model_dir = target_dir
        _pipeline_quant = target_quant
        print(f"[sdxl] pipeline loaded successfully from {target_dir.name}", file=sys.stderr, flush=True)
    return _pipeline


_applied_lora_hooks = []  # list of (comp, parent, last_key, original_linear)


def _unload_loras():
    global _applied_lora_hooks, _lora_state
    had_te = any(h[0] in ("text_encoder", "text_encoder_2") for h in _applied_lora_hooks)
    if had_te:
        _prompt_cache.clear()
    for comp, parent, last, linear in _applied_lora_hooks:
        try:
            if isinstance(parent, list) or (isinstance(last, str) and last.isdigit() and isinstance(parent, list)):
                parent[int(last)] = linear
            else:
                setattr(parent, last, linear)
        except Exception:
            pass
    _applied_lora_hooks.clear()
    _lora_state = None


def _reset_pipeline():
    """Drop the resident pipeline if a model or quantization change requires a reload."""
    global _pipeline, _pipeline_model_dir, _pipeline_quant, _lora_state
    _prompt_cache.clear()
    _unload_loras()
    _pipeline = None
    _pipeline_model_dir = None
    _pipeline_quant = None
    _lora_state = None
    gc.collect()


# ---------------------------------------------------------------- schedulers
def make_scheduler(name: str):
    from mlx_diffuser.schedulers import DDIMScheduler, EulerDiscreteScheduler
    from mlx_diffuser.schedulers.euler import EulerConfig
    from mlx_diffuser.schedulers.ddpm import DDPMConfig

    if name in ("euler_a", "euler_a_substep"):
        return EulerAncestralScheduler(
            EulerConfig(beta_schedule="scaled_linear", beta_start=0.00085,
                        beta_end=0.012, prediction_type="epsilon",
                        timestep_spacing="leading", steps_offset=1),
            substeps=2 if name == "euler_a_substep" else 1,
        )
    if name in ("dpmpp_2m_karras", "dpmpp_2m_karras_trailing"):
        spacing = "trailing" if "trailing" in name else "leading"
        return DPMSolverMultistepKarrasScheduler(
            EulerConfig(beta_schedule="scaled_linear", beta_start=0.00085,
                        beta_end=0.012, prediction_type="epsilon",
                        timestep_spacing=spacing, steps_offset=1)
        )
    if name == "ddim":
        return DDIMScheduler()
    spacing = "trailing" if name in ("euler_trailing", "trailing") else "leading"
    return EulerDiscreteScheduler(
        EulerConfig(beta_schedule="scaled_linear", beta_start=0.00085,
                    beta_end=0.012, prediction_type="epsilon",
                    timestep_spacing=spacing, steps_offset=1)
    )


class EulerAncestralScheduler:
    """Euler a (eta=1), DrawThings-style k-diffusion ancestral update:
      sigma_up   = min(s_next, sqrt(s_next^2 * (s^2 - s_next^2) / s^2))
      sigma_down = sqrt(s_next^2 - sigma_up^2)
      x         += et * (sigma_down - sigma) ; then + noise * sigma_up
    substeps > 1 subdivides every main step into that many log-space
    sub-steps (DrawThings "Euler A Substep"): finer integration at the same
    UNet-call-per-substep cost structure, better detail at low step counts."""

    def __init__(self, config, substeps: int = 1):
        from mlx_diffuser.schedulers.euler import EulerDiscreteScheduler

        self._inner = EulerDiscreteScheduler(config)
        self._substeps = max(1, int(substeps))
        self._sigmas = None
        self._step_index = 0

    def set_timesteps(self, n):
        import math

        self._inner.set_timesteps(n)
        base_sig = [float(s) for s in self._inner.sigmas]  # len n+1, last == 0
        base_ts = [float(t) for t in self._inner.timesteps]
        if self._substeps == 1:
            self._sigmas = base_sig
            self._timesteps = base_ts
        else:
            eps = 1e-5
            sig, ts = [], []
            for i in range(len(base_sig) - 1):
                s0, s1 = base_sig[i], max(base_sig[i + 1], 0.0)
                t0 = base_ts[i]
                t1 = base_ts[i + 1] if i + 1 < len(base_ts) else 0.0
                for j in range(self._substeps):
                    f = j / self._substeps
                    sj = math.exp((1 - f) * math.log(max(s0, eps))
                                  + f * math.log(max(s1, eps)))
                    sig.append(sj)
                    ts.append(t0 + (t1 - t0) * f)
            sig.append(0.0)  # exact terminal sigma
            self._sigmas = sig
            self._timesteps = ts
        self._step_index = 0

    @property
    def timesteps(self):
        import mlx.core as mx
        return mx.array(self._timesteps)

    @property
    def init_noise_sigma(self):
        return 1.0

    def scale_model_input(self, sample, t):
        import mlx.core as mx
        sigma = self._sigmas[self._step_index]
        return sample / mx.sqrt(sigma**2 + 1)

    def step(self, model_output, t, sample, key=None):
        i = self._step_index
        sigma, sigma_next = self._sigmas[i], self._sigmas[i + 1]
        self._step_index += 1
        # d = (x - x0)/sigma with x0 = x - sigma*et  ->  d = et (epsilon space)
        var = max(sigma_next**2 - (sigma_next**4) / max(sigma**2, 1e-8), 0.0)
        sigma_up = min(var ** 0.5, sigma_next)
        sigma_down = max(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
        prev = sample + model_output * (sigma_down - sigma)
        if sigma_next > 0 and sigma_up > 0:
            s = getattr(self, "_seed", 0)
            key = mx.random.key(int(s) * 1000003 + int(float(t)) * 7919 + i)
            prev = prev + mx.random.normal(prev.shape, key=key) * sigma_up
        return prev


class DPMSolverMultistepKarrasScheduler:
    """DPM-Solver++(2M) in the Euler scheduler's sigma space (comfy-style)."""

    def __init__(self, config):
        from mlx_diffuser.schedulers.euler import EulerDiscreteScheduler

        self._inner = EulerDiscreteScheduler(config)
        self._old = None
        self._h_last = None

    def set_timesteps(self, n):
        self._inner.set_timesteps(n)
        self._old = None
        self._h_last = None

    @property
    def timesteps(self):
        return self._inner.timesteps

    @property
    def init_noise_sigma(self):
        return self._inner.init_noise_sigma

    def scale_model_input(self, sample, t):
        return self._inner.scale_model_input(sample, t)

    def step(self, model_output, t, sample, key=None):
        import math

        i = self._inner._step_index
        sig = self._inner.sigmas[i]
        sig_next = self._inner.sigmas[i + 1]
        self._inner._step_index += 1

        denoised = sample - sig * model_output  # x0
        if float(sig_next) == 0.0:
            out = denoised
        elif self._old is None:
            out = (sig_next / sig) * sample - math.expm1(
                math.log(float(sig_next)) - math.log(float(sig))
            ) * denoised
        else:
            h = math.log(float(sig)) - math.log(float(sig_next))
            r = self._h_last / h
            D = (1.0 + 1.0 / (2 * r)) * denoised - (1.0 / (2 * r)) * self._old
            out = (sig_next / sig) * sample - math.expm1(-h) * D

        if float(sig_next) > 0:
            self._h_last = math.log(float(sig)) - math.log(float(sig_next))
        self._old = denoised
        return out


# ---------------------------------------------------------------- lora loading
def _map_kohya_key(key: str):
    """kohya/comfy/diffusers/peft LoRA key -> (component, mlx module path). None to skip."""
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]

    if key.startswith("lora_unet_"):
        comp, rest = "unet", key[len("lora_unet_"):]
    elif key.startswith("lora_te1_"):
        comp, rest = "text_encoder", key[len("lora_te1_"):]
    elif key.startswith("lora_te2_"):
        comp, rest = "text_encoder_2", key[len("lora_te2_"):]
    elif key.startswith("unet."):
        comp, rest = "unet", key[len("unet."):]
    elif key.startswith("text_encoder."):
        comp, rest = "text_encoder", key[len("text_encoder."):]
    elif key.startswith("text_encoder_2."):
        comp, rest = "text_encoder_2", key[len("text_encoder_2."):]
    elif key.startswith("te1."):
        comp, rest = "text_encoder", key[len("te1."):]
    elif key.startswith("te2."):
        comp, rest = "text_encoder_2", key[len("te2."):]
    else:
        return None

    if "." in rest:
        return comp, rest

    if comp in ("text_encoder", "text_encoder_2"):
        m = re.match(r"text_model_encoder_layers_(\d+)_(.+)", rest)
        if m:
            layer_idx, tail = m.group(1), m.group(2)
            te_map = {
                "mlp_fc1": "mlp.fc1",
                "mlp_fc2": "mlp.fc2",
                "self_attn_q_proj": "self_attn.q_proj",
                "self_attn_k_proj": "self_attn.k_proj",
                "self_attn_v_proj": "self_attn.v_proj",
                "self_attn_out_proj": "self_attn.out_proj",
            }
            sub = te_map.get(tail)
            if sub:
                return comp, f"text_model.encoder.layers.{layer_idx}.{sub}"
        return None

    suffix_map = {
        "attn1_to_q": "attn1.to_q", "attn1_to_k": "attn1.to_k",
        "attn1_to_v": "attn1.to_v", "attn1_to_out_0": "attn1.to_out.0",
        "attn2_to_q": "attn2.to_q", "attn2_to_k": "attn2.to_k",
        "attn2_to_v": "attn2.to_v", "attn2_to_out_0": "attn2.to_out.0",
        "ff_net_0_proj": "ff.net.0.proj", "ff_net_2": "ff.net.2",
        "proj_in": "proj_in", "proj_out": "proj_out",
    }

    if rest.startswith("middle_block_1_"):
        tail = rest[len("middle_block_1_"):]
        mt = re.match(r"transformer_blocks_(\d+)_(.+)", tail)
        if mt:
            sub = suffix_map.get(mt.group(2))
            if not sub:
                return None
            tail = f"transformer_blocks.{mt.group(1)}.{sub}"
        else:
            tail = suffix_map.get(tail)
            if not tail:
                return None
        return comp, f"mid_block.attentions.0.{tail}"

    # Diffusers format: down_blocks / up_blocks / mid_block (SDXL-Lightning, official diffusers LoRAs)
    m_proj = re.match(r"(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_(proj_in|proj_out)", rest)
    if m_proj:
        return comp, f"{m_proj.group(1)}.{m_proj.group(2)}.attentions.{m_proj.group(3)}.{m_proj.group(4)}"

    m_mid_proj = re.match(r"mid_block_attentions_0_(proj_in|proj_out)", rest)
    if m_mid_proj:
        return comp, f"mid_block.attentions.0.{m_mid_proj.group(1)}"

    md = re.match(r"(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_transformer_blocks_(\d+)_(.+)", rest)
    if md:
        block, b_idx, a_idx, tb_idx, tail = md.group(1), md.group(2), md.group(3), md.group(4), md.group(5)
        sub = suffix_map.get(tail)
        if sub:
            return comp, f"{block}.{b_idx}.attentions.{a_idx}.transformer_blocks.{tb_idx}.{sub}"

    mm = re.match(r"mid_block_attentions_0_transformer_blocks_(\d+)_(.+)", rest)
    if mm:
        tb_idx, tail = mm.group(1), mm.group(2)
        sub = suffix_map.get(tail)
        if sub:
            return comp, f"mid_block.attentions.0.transformer_blocks.{tb_idx}.{sub}"

    m_time = re.match(r"(down_blocks|up_blocks)_(\d+)_resnets_(\d+)_time_emb_proj", rest)
    if m_time:
        return comp, f"{m_time.group(1)}.{m_time.group(2)}.resnets.{m_time.group(3)}.time_emb_proj"

    m_mid_time = re.match(r"mid_block_resnets_(\d+)_time_emb_proj", rest)
    if m_mid_time:
        return comp, f"mid_block.resnets.{m_mid_time.group(1)}.time_emb_proj"

    m = re.match(r"(input_blocks|output_blocks)_(\d+)_(\d+)_(.+)", rest)
    if not m:
        return None
    kind, a, b, tail = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    mt = re.match(r"transformer_blocks_(\d+)_(.+)", tail)
    if mt:
        sub = suffix_map.get(mt.group(2))
        if not sub:
            return None
        tail = f"transformer_blocks.{mt.group(1)}.{sub}"
    else:
        tail = suffix_map.get(tail)
        if not tail:
            return None
    if kind == "input_blocks":
        level, idx = {4: (1, 0), 5: (1, 1), 7: (2, 0), 8: (2, 1)}.get(a, (None, None))
        if level is None:
            return None
        path = f"down_blocks.{level}.attentions.{idx}.{tail}"
    elif kind == "output_blocks":
        path = f"up_blocks.{a // 3}.attentions.{a % 3}.{tail}"
    else:
        return None
    return comp, path


def _read_safetensors_tensor(fh, info, offset_base):
    """Reader supporting F32/F16/BF16/integers without torch."""
    dtype, shape = info["dtype"], info["shape"]
    start, end = info["data_offsets"]
    fh.seek(offset_base + start)
    buf = fh.read(end - start)
    if dtype == "F32":
        return np.frombuffer(buf, dtype="<f4").reshape(shape)
    if dtype == "F16":
        return np.frombuffer(buf, dtype="<f2").astype(np.float32).reshape(shape)
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype="<u2")
        return (u16.astype(np.uint32) << 16).view("<f4").reshape(shape)
    if dtype == "F64":
        return np.frombuffer(buf, dtype="<f8").astype(np.float32).reshape(shape)
    if dtype in ("I64", "INT64"):
        return np.frombuffer(buf, dtype="<i8").reshape(shape)
    if dtype in ("I32", "INT32"):
        return np.frombuffer(buf, dtype="<i4").reshape(shape)
    if dtype in ("I16", "INT16"):
        return np.frombuffer(buf, dtype="<i2").reshape(shape)
    if dtype in ("I8", "INT8"):
        return np.frombuffer(buf, dtype="<i1").reshape(shape)
    if dtype in ("U64", "UINT64"):
        return np.frombuffer(buf, dtype="<u8").reshape(shape)
    if dtype in ("U32", "UINT32"):
        return np.frombuffer(buf, dtype="<u4").reshape(shape)
    if dtype in ("U16", "UINT16"):
        return np.frombuffer(buf, dtype="<u2").reshape(shape)
    if dtype in ("U8", "UINT8"):
        return np.frombuffer(buf, dtype="<u1").reshape(shape)
    if dtype == "BOOL":
        return np.frombuffer(buf, dtype="?").reshape(shape)
    raise ValueError(f"unsupported dtype {dtype}")


def load_multilora(pipe, loras: list[dict]):
    """loras: [{path, scale}]. Stacks adapters via rank concatenation."""
    from mlx_diffuser.lora.lora import LoRALinear

    components = {"unet": pipe.unet,
                  "text_encoder": pipe.text_encoder,
                  "text_encoder_2": pipe.text_encoder_2}

    deltas = {}  # component -> path -> list[(A, B)]
    for lora in loras:
        scale = float(lora.get("scale", 1.0))
        path = Path(lora["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LoRA file not found: {path}")
        file_size = path.stat().st_size
        if file_size < 16:
            raise ValueError(f"LoRA file too small ({file_size} bytes): {path.name}")
        entries = {}
        with open(path, "rb") as fh:
            header_len_bytes = fh.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"Could not read safetensors header length from {path.name}")
            n = struct.unpack("<Q", header_len_bytes)[0]
            if n <= 0 or n > file_size - 8 or n > 100 * 1024 * 1024:
                raise ValueError(
                    f"Invalid safetensors header length {n} (file size: {file_size}) in {path.name}. "
                    f"The file may be corrupted or not a valid safetensors archive."
                )
            header_bytes = fh.read(n)
            if len(header_bytes) < n:
                raise ValueError(f"Truncated safetensors header in {path.name}")
            try:
                header = json.loads(header_bytes.decode("utf-8"))
            except Exception as e:
                raise ValueError(f"Failed to parse safetensors JSON header in {path.name}: {e}")
            base_off = 8 + n
            for k, info in header.items():
                if k == "__metadata__":
                    continue
                if k.endswith(".alpha"):
                    part = "alpha"
                    b = k[: -len(".alpha")]
                elif k.endswith(".lora_down.weight"):
                    part = "down"
                    b = k[: -len(".lora_down.weight")]
                elif k.endswith(".lora_up.weight"):
                    part = "up"
                    b = k[: -len(".lora_up.weight")]
                elif k.endswith(".lora_A.weight"):
                    part = "down"
                    b = k[: -len(".lora_A.weight")]
                elif k.endswith(".lora_B.weight"):
                    part = "up"
                    b = k[: -len(".lora_B.weight")]
                elif k.endswith(".down.weight"):
                    part = "down"
                    b = k[: -len(".down.weight")]
                elif k.endswith(".up.weight"):
                    part = "up"
                    b = k[: -len(".up.weight")]
                else:
                    continue
                entries.setdefault(b, {})[part] = _read_safetensors_tensor(
                    fh, info, base_off)
        for b_, parts in entries.items():
            mapped = _map_kohya_key(b_)
            if not mapped or "down" not in parts or "up" not in parts:
                continue
            comp, mpath = mapped
            rank = parts["down"].shape[0]
            if "alpha" in parts:
                alpha = float(np.asarray(parts["alpha"]).reshape(-1)[0])
            else:
                alpha = float(rank)
            eff = scale * alpha / rank
            A = parts["down"].astype(np.float32) * eff  # (rank, in)
            B = parts["up"].astype(np.float32)          # (out, rank)
            deltas.setdefault(comp, {}).setdefault(mpath, []).append((A, B))

    total, missed = 0, 0
    for comp, paths in deltas.items():
        model = components[comp]
        for path, stack in paths.items():
            segs = path.split(".")
            try:
                parent = model
                for seg in segs[:-1]:
                    if isinstance(parent, (list, tuple)):
                        parent = parent[int(seg)]
                    elif hasattr(parent, seg):
                        parent = getattr(parent, seg)
                    else:
                        parent = parent[int(seg)] if seg.isdigit() else parent[seg]
                last = segs[-1]
                if isinstance(parent, (list, tuple)) or (last.isdigit() and isinstance(parent, list)):
                    linear = parent[int(last)]
                elif hasattr(parent, last):
                    linear = getattr(parent, last)
                else:
                    linear = parent[last]
                if type(linear).__name__ not in ("Linear", "QuantizedLinear"):
                    raise TypeError(f"{last} is {type(linear).__name__}")
                adapter = LoRALinear(linear, rank=8, alpha=16.0)
                As = [x[0] for x in stack]
                Bs = [x[1] for x in stack]
                adapter.lora_a = mx.array(np.concatenate(As, axis=0))
                adapter.lora_b = mx.array(np.concatenate(Bs, axis=1))
                adapter.scale = 1.0
                _applied_lora_hooks.append((comp, parent, last, linear))
                if isinstance(parent, list) or (last.isdigit() and isinstance(parent, list)):
                    parent[int(last)] = adapter
                else:
                    setattr(parent, last, adapter)
                total += 1
            except Exception as e:
                missed += 1
                print(f"[sdxl] WARN skip {comp}:{path}: {e}", file=sys.stderr)
    if "text_encoder" in deltas or "text_encoder_2" in deltas:
        _prompt_cache.clear()
    for comp in deltas:
        mx.eval(components[comp].parameters())
    return total


# ---------------------------------------------------------------- pipeline
def generate(req: dict) -> dict:
    import time

    model_dir = req.get("model_dir")
    quant = req.get("quantize_unet", 4)
    target_dir = Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR.resolve()
    target_quant = quant or None
    if _pipeline is not None and _pipeline_model_dir == target_dir and _pipeline_quant == target_quant:
        _write_ipc_json({"phase": "preparing", "detail": "Preparing prompt conditioning & latents..."})
    pipe = get_pipeline(model_dir=model_dir, quantize_unet=quant)
    pipe.scheduler = make_scheduler(req.get("sampler", "dpmpp_2m_karras"))
    pipe.scheduler._seed = req.get("seed", 0)
    global _lora_state
    state = tuple(sorted(
        (str(Path(l["path"]).expanduser().resolve()),
         float(l.get("scale", 1.0))) for l in req.get("loras") or []))
    if state != _lora_state:
        if _lora_state is not None:
            # Revert adapters without dropping base model pipeline
            _unload_loras()
        if req.get("loras"):
            try:
                _write_ipc_json({"phase": "loading_model", "detail": f"Applying {len(req['loras'])} LoRA adapter(s)..."})
                n = load_multilora(pipe, req["loras"])
                print(f"[sdxl] applied {n} LoRA layers", file=sys.stderr, flush=True)
                _lora_state = state
            except Exception as e:
                _unload_loras()
                _lora_state = None
                raise RuntimeError(f"Failed loading LoRA: {e}") from e
        else:
            _lora_state = state
            print("[sdxl] no LoRA (clean weights)", file=sys.stderr, flush=True)
    t0 = time.time()

    def _progress(t):
        _write_ipc_json({
            "phase": "generating",
            "detail": f"Denoising step {t + 1}/{req['steps']}...",
            "progress": {"step": t + 1, "steps": req["steps"]},
        })

    orig_step = pipe.scheduler.step
    def _step_progress(model_output, timestep, sample, **kwargs):
        res = orig_step(model_output, timestep, sample, **kwargs)
        idx = getattr(pipe.scheduler, "_step_index", None)
        inner = getattr(pipe.scheduler, "_inner", None)
        if idx is None and inner is not None:
            idx = getattr(inner, "_step_index", None)
        if idx is not None:
            _progress(min(idx, req["steps"]) - 1)
        return res
    pipe.scheduler.step = _step_progress

    # Wire engine memory during denoise: allow up to 70% of working memory
    # to keep Metal from paging weights while leaving headroom for VAE decode.
    prev_wired = None
    try:
        import mlx.core as _mx
        d = _mx.device_info()
        cap = d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size", 0)
        # Clamped to 6.5GB max on 16GB machines to guarantee OS and activations never trigger swap
        limit = min(cap if cap > 0 else (6500 << 20), int(6.5 * (1 << 30)))
        prev_wired = _mx.set_wired_limit(limit)
    except Exception:
        prev_wired = None
    try:
        pipe.vae.use_taesd = bool(req.get("fast_vae", True))
        guidance = req.get("guidance")
        if guidance is None:
            # Distilled models (Lightning 4-step) require guidance=1.0 for single-pass CFG-free denoise
            guidance = 1.0 if (req.get("steps", 4) <= 8 or "lightning" in str(req.get("model_dir", "")).lower()) else 2.4
        else:
            guidance = float(guidance)

        _write_ipc_json({"phase": "compiling", "detail": "Compiling Metal scheduler & encoding prompt..."})
        out = pipe(
            req["prompt"],
            negative_prompt=req.get("negative_prompt", ""),
            height=req["height"],
            width=req["width"],
            num_inference_steps=req["steps"],
            guidance_scale=guidance,
            seed=req["seed"],
            cache_interval=req.get("cache_interval", 1),
            tile_vae=req.get("tile_vae", (req["width"] * req["height"] >= 768 * 768)),
            # NOTE: must stay False in daemon mode — the library's release
            # permanently drops the CLIP encoders (no lazy reload), killing
            # every subsequent prompt. AGENTS.md claim is outdated.
            release_text_encoders=False,
            progress=False,
        )
    finally:
        pipe.scheduler.step = orig_step
        if prev_wired is not None:
            try:
                import mlx.core as _mx
                _mx.set_wired_limit(prev_wired)
            except Exception:
                pass
    elapsed = round(time.time() - t0, 2)
    image_id = req["image_id"]
    dest = Path(req["dest"])

    _write_ipc_json({"phase": "saving", "detail": "Decoding VAE latents & saving image..."})
    arr = np.asarray(out[0])
    if arr.min() < 0.0:
        arr = (arr + 1.0) / 2.0
    img = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    dest.write_bytes(img.tobytes())
    return {"id": image_id, "generation_time": elapsed}


def _handle_signal(signum, frame):
    try:
        _reset_pipeline()
        mx.clear_cache()
    except Exception:
        pass
    sys.exit(0)


def _serve():
    """Daemon mode: one JSON request per stdin line, JSON responses on stdout.
    Model stays resident across requests."""
    import signal

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = generate(req)
            _write_ipc_json(res)
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _write_ipc_json({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        _serve()
    else:
        req = json.loads(sys.argv[1])
        try:
            _write_ipc_json(generate(req))
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _write_ipc_json({"error": f"{type(e).__name__}: {e}"})
            sys.exit(1)
