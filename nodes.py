import logging
import math

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import comfy.text_encoders.qwen_vl
import node_helpers
from comfy_api.latest import ComfyExtension, io

from .moss_edit.model import ISOLATED, KV_CACHE, KV_CACHE_MODES, TEXT_START

MAX_REFERENCES = 4  # the LoRAs were trained with one to four references
# Moss-Image-Edit's layout; the method names are ComfyUI's (Comfy-Org/ComfyUI#17296), so the conditioning means the
# same to this pack's model hooks and to ComfyUI's own MOSS edit support, which takes ai-toolkit's layout without one
REFERENCE_METHOD = TEXT_START
GRID = 16
BUCKET_PIXELS = 1024 * 1024  # reference_max_edge 0: about the size the model and the LoRAs were trained and tested at
# edit LoRA formats: the reference layout, and whether the text encoder also sees the references (ai-toolkit's
# "Picture N:" vision prompt)
EDIT_FORMATS = {
    "Moss-Image-Edit (chinoll)": (TEXT_START, False),
    "ai-toolkit (MOSS-Image-Edit-HQ)": (ISOLATED, True),
}
VISION_MAX_PIXELS = 384 * 384  # ai-toolkit's vlm_max_pixels: the vision tower sees each reference at most this big
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def bucket(width, height, max_pixels=BUCKET_PIXELS):
    """The multiple-of-16 size closest to the image's aspect ratio with at most max_pixels: about 1 MP, the size the
    model and the LoRAs edit best at, whatever the image's own size."""
    aspect = width / height
    best = None
    h = math.sqrt(max_pixels / aspect)
    for bh in {max(GRID, int(h // GRID) * GRID), max(GRID, int(h // GRID) * GRID + GRID)}:
        for bw in {max(GRID, int(bh * aspect // GRID) * GRID), max(GRID, int(bh * aspect // GRID) * GRID + GRID)}:
            if bw * bh > max_pixels:
                continue
            # closest aspect first, then the larger area
            key = (abs(math.log(bw / bh / aspect)), -bw * bh)
            if best is None or key < best[0]:
                best = (key, (bw, bh))
    return best[1]


def resize_reference(image, max_edge):
    """One IMAGE item (H, W, C) -> [1, h, w, 3], Lanczos on 8-bit pixels as upstream, no crop. max_edge 0: the 1 MP
    bucket of the image's aspect ratio (up or down). Otherwise each side rounded to 16 and the long edge capped at
    max_edge, never upscaled."""
    pixels = image[:, :, :3].float().cpu().clamp(0, 1)
    picture = Image.fromarray((pixels * 255).round().to(torch.uint8).numpy())
    width, height = picture.size
    if max_edge == 0:
        size = bucket(width, height)
    else:
        limit = max_edge // GRID * GRID
        scale = min(1.0, limit / max(width, height))
        size = tuple(min(limit, max(GRID, int(math.floor(value * scale / GRID + 0.5)) * GRID)) for value in (width, height))
    if size != (width, height):
        picture = picture.resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0)[None]


def reference_items(images):
    """Autogrow sockets in order, every batch item a reference of its own."""
    items = []
    for name in sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
        if images[name] is not None:
            items.extend(images[name])
    if not items:
        raise ValueError("Connect at least one reference image.")
    if len(items) > MAX_REFERENCES:
        logging.warning("MOSS-Image Edit: %d references, the LoRAs were trained with at most %d.", len(items), MAX_REFERENCES)
    return items


def vision_image(image):
    """One IMAGE item (H, W, C) -> [1, h, w, 3] for the vision tower: ai-toolkit's _prep_vlm_images, scaled down (never
    up) to VISION_MAX_PIXELS, bicubic with antialiasing, at least 32 px a side."""
    pixels = image[None, :, :, :3].movedim(-1, 1).float()
    h, w = pixels.shape[-2:]
    scale = min(1.0, math.sqrt(VISION_MAX_PIXELS / (h * w)))
    size = (max(round(h * scale), 32), max(round(w * scale), 32))
    if size != (h, w):
        pixels = torch.nn.functional.interpolate(pixels, size=size, mode="bicubic", antialias=True).clamp(0, 1)
    return pixels.movedim(1, -1)


def encode_prompt(clip, prompt, vision_images, image_strength=1.0):
    """The prompt, after "Picture N: <image>" for each vision image as ai-toolkit trains; image_strength biases the
    attention to the image tokens (log of it on their keys), as ComfyUI's Text Encode MOSS-Image Edit does."""
    if vision_images:
        prompt = "".join("Picture {}: {}".format(i + 1, VISION_BLOCK) for i in range(len(vision_images))) + prompt
    tokens = clip.tokenize(prompt, images=vision_images)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if vision_images and image_strength != 1.0:
        weight = math.log(image_strength)
        bias = []
        for t in next(iter(tokens.values()))[0]:
            if isinstance(t[0], dict):
                # Qwen3.5 merges 2x2 patches of 16 px into one token
                _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(t[0]["data"], patch_size=16)
                bias += [weight] * (int(grid[0, 1] * grid[0, 2]) // 4)
            else:
                bias.append(0.0)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"context_attention_bias": torch.tensor([bias])})
    return conditioning


def output_latent(image, latent, max_edge):
    """Empty latent: the first image's aspect with its long edge at max_edge (multiples of 16), or the first
    reference's size when max_edge is 0."""
    if max_edge == 0:
        return {"samples": torch.zeros_like(latent)}
    h, w = image.shape[:2]
    scale = max_edge / max(h, w)
    size = [max(GRID, int(math.floor(v * scale / GRID + 0.5)) * GRID) // GRID for v in (h, w)]
    return {"samples": latent.new_zeros((1, latent.shape[1], size[0], size[1]))}


class MOSSImageEditTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageEditTextEncode",
            display_name="MOSS-Image Edit Text Encode",
            category="model/conditioning/moss_image",
            description="Encodes an edit instruction and its negative with one to four reference images for a MOSS-Image edit LoRA, "
                        "in the layout the LoRA was trained with. Refer to the images in the prompt as image 1, image 2 and so on.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="The MOSS-Image VAE (flux2-vae_small or flux2-vae: the same encoder)."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, tooltip="The edit instruction."),
                io.String.Input("negative_prompt", multiline=True, default="", tooltip="Both LoRAs were trained and published with an empty negative."),
                io.Combo.Input("edit_format", options=list(EDIT_FORMATS),
                               tooltip="The layout the LoRA was trained with. Moss-Image-Edit (chinoll): references at the text's start. "
                                       "ai-toolkit (akhaliq's MOSS-Image-Edit-HQ and other ai-toolkit edit LoRAs): references at the "
                                       "image's position, and the text encoder also sees them, as \"Picture N:\" before the prompt."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=[f"image_{i}" for i in range(1, MAX_REFERENCES + 1)], min=1),
                    tooltip="References in order; image 1 is the one being edited. Each batch item counts as a reference of its own.",
                ),
                io.Int.Input("reference_max_edge", default=0, min=0, max=4096, step=GRID,
                             tooltip="0 (recommended): each reference is resized, up or down, to the 1 MP size closest to its own aspect "
                                     "ratio (sides multiples of 16, no crop), about the size the LoRAs edit best at; with output_max_edge 0 "
                                     "the output takes image 1's. Otherwise the longest side in pixels, never upscaled."),
                io.Int.Input("output_max_edge", default=0, min=0, max=4096, step=GRID,
                             tooltip="0 (recommended): the output latent is the size of reference 1. Otherwise its longest side, in "
                                     "image 1's aspect. Both LoRAs edit best when the output and the references are the same size; "
                                     "with smaller references the image zooms out or the edit fails."),
                io.Combo.Input("kv_cache", options=list(KV_CACHE_MODES), default="auto",
                               tooltip="Keep the references' keys/values for the whole sampling run (about twice as fast, 0.5 MB per "
                                       "reference token) or recompute them every step. auto: keep them when they fit. Same image either way."),
                io.Float.Input("image_strength", default=1.0, min=0.0, max=100.0, step=0.05, advanced=True,
                               tooltip="ai-toolkit format: scales how much the model attends to the images in the prompt. 0 leaves them out, "
                                       "1 is as trained."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="An empty latent the size of reference 1 (or output_max_edge). Sample at this size: a "
                                                                "different output size than the references zooms the image out or weakens the edit."),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, negative_prompt, edit_format, images: io.Autogrow.Type, reference_max_edge=0,
                output_max_edge=0, kv_cache="auto", image_strength=1.0) -> io.NodeOutput:
        method, vision = EDIT_FORMATS[edit_format]
        items = reference_items(images)
        vision_images = [vision_image(image) for image in items] if vision and image_strength > 0 else []
        latents = [vae.encode(resize_reference(image, reference_max_edge)) for image in items]
        values = {"reference_latents": latents, "reference_latents_method": method, KV_CACHE: kv_cache}
        positive, negative = [node_helpers.conditioning_set_values(encode_prompt(clip, text, vision_images, image_strength), values)
                              for text in (prompt, negative_prompt)]
        return io.NodeOutput(positive, negative, output_latent(items[0], latents[0], output_max_edge))


class MOSSImageEditReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageEditReferences",
            display_name="MOSS-Image Edit References",
            category="model/conditioning/moss_image",
            description="Adds one to four reference images for the Moss-Image-Edit LoRA (chinoll) to the positive and negative "
                        "conditioning, resized as upstream does. Refer to them in the prompt as image 1, image 2 and so on. For "
                        "ai-toolkit LoRAs such as MOSS-Image-Edit-HQ use MOSS-Image Edit Text Encode: their text encoder also sees "
                        "the images.",
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
                io.Int.Input("reference_max_edge", default=0, min=0, max=4096, step=GRID,
                             tooltip="0 (recommended): each reference is resized, up or down, to the 1 MP size closest to its own aspect "
                                     "ratio (sides multiples of 16, no crop). Otherwise the longest side in pixels, never upscaled."),
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
                io.Latent.Output(display_name="latent", tooltip="An empty latent the size of the first resized reference. Sample at this size: a "
                                                                "different output size than the references zooms the image out or weakens the edit."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, images: io.Autogrow.Type, reference_max_edge=0, kv_cache="auto") -> io.NodeOutput:
        items = reference_items(images)
        latents = [vae.encode(resize_reference(image, reference_max_edge)) for image in items]
        positive, negative = [node_helpers.conditioning_set_values(node_helpers.conditioning_set_values(c, {"reference_latents": latents}, append=True),
                                                                   {KV_CACHE: kv_cache, "reference_latents_method": REFERENCE_METHOD}) for c in (positive, negative)]
        return io.NodeOutput(positive, negative, {"samples": torch.zeros_like(latents[0])})


class MOSSImageEditExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MOSSImageEditTextEncode,
            MOSSImageEditReferences,
        ]


async def comfy_entrypoint() -> MOSSImageEditExtension:
    return MOSSImageEditExtension()
