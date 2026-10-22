"""Hooks that add reference latents to ComfyUI's native MOSS-Image model.

- model_base.MOSSImage passes the conditioning's "reference_latents" (from this pack's node, Set Reference Latent or
  Set Reference Multi Latent) to the DiT as ref_latents, with the node's kv_cache setting, and counts their tokens
  in the memory estimate.
- The DiT runs the edit forward (model.py) when ref_latents are present; without them it is the stock forward.
- comfy.lora_convert also accepts the upstream adapter_model.safetensors as it is published.
Each hook stands down when ComfyUI handles the same thing itself.
"""
import inspect
import logging
import math

import comfy.conds
import comfy.ldm.moss_image.model
import comfy.lora_convert
import comfy.model_base

from . import lora as edit_lora
from .model import KV_CACHE, forward_edit

MOSSImageModel = comfy.model_base.MOSSImage
MOSSImageDiT = comfy.ldm.moss_image.model.MOSSImage


def _patch_model_base():
    original_init = MOSSImageModel.__init__
    original_extra_conds = MOSSImageModel.extra_conds
    original_extra_conds_shapes = MOSSImageModel.extra_conds_shapes

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if "ref_latents" not in self.memory_usage_factor_conds:
            self.memory_usage_factor_conds = tuple(self.memory_usage_factor_conds) + ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = original_extra_conds(self, **kwargs)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None and "ref_latents" not in out:
            out["ref_latents"] = comfy.conds.CONDList([self.process_latent_in(latent) for latent in ref_latents])
        kv_cache = kwargs.get(KV_CACHE, None)
        if kv_cache is not None:
            out[KV_CACHE] = comfy.conds.CONDConstant(kv_cache)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = original_extra_conds_shapes(self, **kwargs)
        ref_latents = kwargs.get("reference_latents", None)
        if ref_latents is not None and "ref_latents" not in out:
            # one token per latent pixel (the latents are already 2x2 packed)
            out["ref_latents"] = [1, self.latent_format.latent_channels, sum(math.prod(latent.shape[2:]) for latent in ref_latents)]
        return out

    MOSSImageModel.__init__ = __init__
    MOSSImageModel.extra_conds = extra_conds
    MOSSImageModel.extra_conds_shapes = extra_conds_shapes


def _patch_dit():
    original_forward = MOSSImageDiT._forward

    def _forward(self, x, timestep, context, transformer_options={}, ref_latents=None, moss_edit_kv_cache="auto", **kwargs):
        if not ref_latents:
            return original_forward(self, x, timestep, context, transformer_options, **kwargs)
        return forward_edit(self, x, timestep, context, transformer_options, ref_latents, moss_edit_kv_cache)

    MOSSImageDiT._forward = _forward


def _patch_lora_convert():
    original_convert = comfy.lora_convert.convert_lora

    def convert_lora(sd):
        if edit_lora.is_upstream(sd):
            sd = edit_lora.to_comfy(sd)
        return original_convert(sd)

    comfy.lora_convert.convert_lora = convert_lora


def install():
    if getattr(MOSSImageDiT, "_moss_edit_native", False):
        return
    if "ref_latents" in inspect.signature(MOSSImageDiT._forward).parameters:
        logging.warning("MOSS-Image Edit Native: this ComfyUI already passes reference latents to MOSS-Image, leaving its model code alone.")
    else:
        _patch_model_base()
        _patch_dit()
    _patch_lora_convert()
    MOSSImageDiT._moss_edit_native = True
