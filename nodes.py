import logging
import math

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import node_helpers
from comfy_api.latest import ComfyExtension, io

from .moss_edit.model import KV_CACHE, KV_CACHE_MODES

MAX_REFERENCES = 4  # the LoRA was trained with one to four references
GRID = 16


def resize_reference(image, max_edge):
    """One IMAGE item (H, W, C) -> [1, h, w, 3]: upstream's Lanczos resize of 8-bit pixels, each side rounded to 16
    and the long edge capped at max_edge, without cropping."""
    pixels = image[:, :, :3].float().cpu().clamp(0, 1)
    picture = Image.fromarray((pixels * 255).round().to(torch.uint8).numpy())
    width, height = picture.size
    limit = max_edge // GRID * GRID
    scale = min(1.0, limit / max(width, height))

    def aligned(value):
        return min(limit, max(GRID, int(math.floor(value * scale / GRID + 0.5)) * GRID))

    picture = picture.resize((aligned(width), aligned(height)), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0)[None]


class MOSSImageEditReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageEditReferences",
            display_name="MOSS-Image Edit References",
            category="model/conditioning/moss_image",
            description="Adds one to four reference images for the Moss-Image-Edit LoRA to the positive and negative conditioning, "
                        "resized as upstream does. Refer to them in the prompt as image 1, image 2 and so on.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", tooltip="The MOSS-Image VAE (flux2-vae_small or flux2-vae: the same encoder)."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, MAX_REFERENCES + 1)],
                        min=1,
                    ),
                    tooltip="References in order. Each batch item counts as a reference of its own.",
                ),
                io.Int.Input("reference_max_edge", default=512, min=GRID, max=4096, step=GRID,
                             tooltip="Longest side of each reference in pixels; smaller images are not upscaled. Sides round to multiples of 16, "
                                     "without cropping. The LoRA was trained and published with 512."),
                io.Combo.Input("kv_cache", options=list(KV_CACHE_MODES), default="auto",
                               tooltip="The references don't depend on the step or the prompt. on: run them once per sampling run and keep "
                                       "their keys/values, about twice as fast (0.5 MB per reference token in bf16: 0.5 GB for a 512px "
                                       "reference, 2 GB for 1024px). off: run them again at every step, no extra memory. auto: on when "
                                       "the keys/values take at most half the free memory. Same image either way. Use off with LoRA "
                                       "schedules (hooks) that change the weights during sampling."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="An empty latent the size of the first resized reference. Any size works."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, images: io.Autogrow.Type, reference_max_edge=512, kv_cache="auto") -> io.NodeOutput:
        items = []
        for name in sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
            if images[name] is not None:
                items.extend(images[name])
        if not items:
            raise ValueError("Connect at least one reference image.")
        if len(items) > MAX_REFERENCES:
            logging.warning("MOSS-Image Edit: %d references, the LoRA was trained with at most %d.", len(items), MAX_REFERENCES)
        latents = [vae.encode(resize_reference(image, reference_max_edge)) for image in items]
        positive, negative = [node_helpers.conditioning_set_values(node_helpers.conditioning_set_values(c, {"reference_latents": latents}, append=True),
                                                                   {KV_CACHE: kv_cache}) for c in (positive, negative)]
        return io.NodeOutput(positive, negative, {"samples": torch.zeros_like(latents[0])})


class MOSSImageEditExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MOSSImageEditReferences,
        ]


async def comfy_entrypoint() -> MOSSImageEditExtension:
    return MOSSImageEditExtension()
