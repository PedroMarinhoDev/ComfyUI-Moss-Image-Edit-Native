"""Key conversion for the Moss-Image-Edit LoRA (format "moss-edit-lora" v1).

Upstream:  layers.{i}.proj.{q,k,v,o,ffn_up_gate,ffn_down}.lora_A / .lora_B, scale alpha / rank from adapter_config.json
ComfyUI:   diffusion_model.layers.{i}.proj.{...}.lora_A.weight / .lora_B.weight / .alpha

ComfyUI maps diffusion_model.layers.{i}.proj.{q,k,v} onto the fused qkv weight of the int8 files by itself.
"""
import json

import torch

FORMAT = "moss-edit-lora"
PREFIX = "diffusion_model."


def is_upstream(sd):
    return "layers.0.proj.ffn_up_gate.lora_A" in sd and "layers.0.proj.q.lora_A" in sd and "video_embedder.weight" not in sd


def adapter_config(metadata):
    config = json.loads((metadata or {}).get("adapter_config", "null") or "null")
    if config is not None and (config.get("format") != FORMAT or config.get("format_version") != 1):
        raise ValueError("Not a moss-edit-lora v1 adapter: format {!r} version {!r}".format(config.get("format"), config.get("format_version")))
    return config


def to_comfy(sd, config=None, dtype=None):
    """Upstream state dict -> ComfyUI LoRA keys. Without the adapter config, alpha is left out: ComfyUI then scales by
    1.0, which is the published adapter's alpha / rank (64 / 64)."""
    modules = (config or {}).get("modules", {})
    out = {}
    for key, tensor in sd.items():
        name, _, part = key.rpartition(".")
        if part not in ("lora_A", "lora_B"):
            raise ValueError("Unexpected key in a moss-edit-lora adapter: {}".format(key))
        out["{}{}.{}.weight".format(PREFIX, name, part)] = tensor if dtype is None else tensor.to(dtype)
        if part == "lora_A" and name in modules:
            out["{}{}.alpha".format(PREFIX, name)] = torch.tensor(float(modules[name]["alpha"]))
    return out
