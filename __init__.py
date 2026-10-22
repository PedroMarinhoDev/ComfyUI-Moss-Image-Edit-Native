"""Moss-Image-Edit (huggingface.co/chinoll/Moss-Image-Edit) on ComfyUI's native MOSS-Image model: stock loaders,
text encoder, VAE, LoRA loader and samplers, with the reference branch added to the DiT."""
from .moss_edit.patches import install
from .nodes import comfy_entrypoint

install()

__all__ = ["comfy_entrypoint"]
