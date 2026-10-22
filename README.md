# ComfyUI-Moss-Image-Edit-Native

Runs the [Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit) image-editing LoRA on ComfyUI's **native**
MOSS-Image model, edits with one to four reference images.

[Chenkin-x/ComfyUI-Moss-Image-Edit](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit) wraps the diffusers/transformers
pipeline: its own loaders for the full HF snapshot, optimum-quanto int8, a hand-written sampler and CPU/GPU shuffling.
This pack uses ComfyUI's own pieces instead:

| | upstream nodes | this pack |
|:--|:--|:--|
| DiT | diffusers `MOSSImageTransformer`, quanto int8 | ComfyUI `UNETLoader`: `moss_image_sft_bf16` / `_int8_convrot` (fused qkv, int8 kernels) |
| LoRA | its own `LoRALinear` injection | stock `LoraLoaderModelOnly`, any strength, stacks with other LoRAs |
| text encoder | transformers Qwen3.5-9B + quanto | `CLIPLoader` type `moss_image` (bf16 or `w4a8_gptq`) |
| VAE | diffusers `AutoencoderKLFlux2` | `VAELoader` (`flux2-vae_small` or `flux2-vae`) |
| sampling | Euler loop, its own scheduler | any KSampler / sampler / scheduler, SLG and other model patches |
| memory | whole models moved CPU↔GPU per stage | ComfyUI model management: smart/dynamic VRAM, prefetching, partial offload, memory estimates that count the reference tokens |

Dependencies: none beyond ComfyUI (with MOSS-Image support).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PedroMarinhoDev/ComfyUI-Moss-Image-Edit-Native
```

Models (ComfyUI's MOSS-Image files from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), plus the LoRA):

| folder | file |
|:--|:--|
| `diffusion_models` | `moss_image_sft_bf16.safetensors` or `moss_image_sft_int8_convrot.safetensors` (Base works too; the LoRA was trained on SFT) |
| `text_encoders` | `qwen3.5_9b_bf16.safetensors` or `qwen3.5_9b_w4a8_gptq.safetensors` |
| `vae` | `flux2-vae_small.safetensors` (MOSS's decoder) or `flux2-vae.safetensors`; the encoders are identical |
| `loras` | `moss_image_edit_lora_bf16.safetensors` from [PedroMarinhoDev/Moss-Image-Edit-ComfyUI](https://huggingface.co/PedroMarinhoDev/Moss-Image-Edit-ComfyUI) |

```bash
hf download PedroMarinhoDev/Moss-Image-Edit-ComfyUI moss_image_edit_lora_bf16.safetensors --local-dir ComfyUI/models/loras
```

### The LoRA

That file is the original [chinoll/Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit) adapter with its
keys renamed; the tensors are unchanged. The published `adapter_model.safetensors` uses keys like
`layers.0.proj.q.lora_A`. To convert it yourself to ComfyUI's
`diffusion_model.layers.0.proj.q.lora_A.weight` / `lora_B.weight` / `alpha`:

```bash
hf download chinoll/Moss-Image-Edit adapter_config.json adapter_model.safetensors --local-dir /tmp/moss-edit
python tools/convert_lora.py /tmp/moss-edit/adapter_model.safetensors ComfyUI/models/loras/moss_image_edit_lora_bf16.safetensors
```

It checks the sha256 against `adapter_config.json`, the targets, shapes and rank/alpha (252 linears, rank 64, alpha 64).
ComfyUI maps the q/k/v parts onto the fused `qkv` weight of the int8/repacked models by itself. With this pack installed
the unconverted file also loads (alpha/rank taken as 1, which is what the published adapter uses).

## Use

Open a workflow from **Templates → ComfyUI-Moss-Image-Edit-Native**, or drop one from
[`example_workflows/`](example_workflows) on the canvas:

- [`moss_image_edit.json`](example_workflows/moss_image_edit.json): edit one image.
- [`moss_image_edit_multi_reference.json`](example_workflows/moss_image_edit_multi_reference.json): two references,
  add up to four on the References node.

They load `moss_edit_bicycle.png` / `moss_edit_basket.png`: copy [`example_inputs/`](example_inputs) to `ComfyUI/input`
or pick your own images. The graph:

```
UNETLoader → LoraLoaderModelOnly (edit LoRA) ───────────────────────────────┐
CLIPLoader (moss_image) → CLIPTextEncode (instruction) ─┐                    │
                       → CLIPTextEncode ("")          ──┤                    │
LoadImage(s) ─────────────────────────────→ MOSS-Image Edit References ──→ KSampler → VAEDecode
VAELoader ─────────────────────────────────┘    (positive, negative, latent)
```

- **MOSS-Image Edit References**: positive, negative, VAE and up to four images (each batch item counts as one).
  Resizes them as upstream does (Lanczos, long edge ≤ `reference_max_edge`, default 512, sides rounded to 16, no crop)
  and appends their latents to both conditionings. Its `latent` output is an empty latent the size of the first
  reference; any `EmptyFlux2LatentImage` size works too. Name the images in the prompt: *"Dress the person in image 1
  in the outfit from image 2."*
- Upstream defaults: 30 steps, cfg 4, shift 5 (the model's default), empty negative. `euler` + `simple` reproduces
  the official schedule.
- ComfyUI's **Set Reference Latent** / **Set Reference Multi Latent** also work for the references; then the sizing is
  yours, and remember to add them to the negative too, as upstream does.
- `kv_cache` on the References node: `auto` (default) / `on` / `off`. `on` runs the references once per sampling run
  and keeps their keys/values, about twice as fast; `off` uses no extra memory. Same image either way, see below.
  References added with the stock nodes use `auto`.

## How it works

The LoRA's edit contract (`ref-prefix-only-t0__target-text-original-moss-v1`, from upstream's `moss_runtime/model.py`):

- the reference latents go through the same 42 blocks at timestep 0, RoPE position `(r = image index + 1, f = 0, h, w centered)`,
  and attend only to each other;
- the target and text tokens keep the stock joint attention, timestep and RoPE, and also attend to the references'
  keys/values in every block.

`moss_edit/patches.py` adds this to the native model, and stands down if ComfyUI handles it itself:
`model_base.MOSSImage` turns the conditioning's `reference_latents` into the `ref_latents` model input and counts their
tokens in the memory estimate, and the DiT takes the edit path (`moss_edit/model.py`) only when `ref_latents` are present;
text-to-image is untouched. The edit path reuses the native block pieces (kitchen fused RMSNorm-AdaLN and RMSNorm-RoPE,
`linear_input_act` situ, fused qkv, ComfyUI attention incl. the int8 attention, dynamic-VRAM prefetching and the memory
compiler region), and passes block replacement patches (skip layer guidance) through.

Since the reference branch never sees the text or the target, its keys/values are the same for every step and for cond
and uncond: they are computed once per sampling run (one extra reference-only pass) and reused, like upstream's KV cache.
That costs 42 × 2 × tokens × 3072 values, about 0.5 MB per reference token in bf16 (a 512² reference is 1024 tokens,
≈ 0.5 GB). The cache lives as long as the run's sigmas tensor and is freed when sampling ends. `auto` caches when it
takes at most half the free memory (counting dynamic-VRAM weights that can be evicted, as ComfyUI's own batching
does) and otherwise runs the reference branch next to the target at every model call, logged once per run.
Batched cond/uncond rows with the same references share one copy. Set `kv_cache` to `off` for LoRA schedules (hooks) that
change the weights during sampling.

## Results (RTX 4090, dynamic VRAM, `moss_image_sft_bf16` + `qwen3.5_9b_bf16` + `flux2-vae`, 30 steps, cfg 4)

![input, LoRA strength 0, LoRA strength 1](docs/bicycle_input_nolora_lora.jpg)

*"Paint the bicycle bright red, keeping the ivy, the brick wall and the lighting the same."* Input; the same graph
with the LoRA at 0 (the base model redraws the scene); with the LoRA.

![two references](docs/two_references_basket.jpg)

*"Attach the wicker basket from image 2 to the front of the bicycle in image 1, mounted on the handlebars, keeping the
ivy, the brick wall and the lighting the same."* Image 1, image 2, result (seed 1; other seeds sometimes leave a
stray copy of the strap on the wall). Image 1 stays the base.

| reference / output | cache | speed | |
|:--|:--|:--|:--|
| 512² (1024 tokens) | on, 504 MB | 5.0 it/s | |
| 512² | off | 2.7 it/s | bit-identical image |
| 1024² (4096 tokens) | on, 2016 MB | 1.04 s/it | |
| 1024² | off | 1.86 s/it | bit-identical image |
| text-to-image 512², no references | – | 4.7 it/s | stock forward |

With the cache an edit costs about what a text-to-image step does. Memory compiler: 1–2 graph breaks, 0 rogues
(text-to-image: 1).

## Tests (CPU only)

- `tests/test_equivalence.py`: the native edit forward against upstream's `MossEditModel` on the official
  `MOSSImageTransformer`, same random weights, float32: unfused and fused qkv, one/two references of different sizes,
  cached and uncached, cond+uncond sharing one prefill, cache invalidation between runs, mixed-reference batches,
  block replacement. Agreement is ~1e-6 relative.
- `tests/test_lora_keys.py`: every LoRA tensor (converted and raw) maps onto the native model, unfused and fused.

```bash
CUDA_VISIBLE_DEVICES="" ComfyUI/.venv/bin/python tests/test_equivalence.py --comfy ComfyUI \
    --pipeline MOSS-Image-SFT/pipeline_moss_image.py --upstream ComfyUI-Moss-Image-Edit
CUDA_VISIBLE_DEVICES="" ComfyUI/.venv/bin/python tests/test_lora_keys.py --comfy ComfyUI \
    --model ComfyUI/models/diffusion_models/moss_image_sft_bf16.safetensors \
    --converted ComfyUI/models/loras/moss_image_edit_lora_bf16.safetensors --raw /tmp/moss-edit/adapter_model.safetensors
```

## Credits and license

The LoRA was trained by [chinoll](https://huggingface.co/chinoll) / [Chenkin-x](https://github.com/Chenkin-x)
([Moss-Image-Edit](https://github.com/Chenkin-x/Moss-Image-Edit)) on [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
by OpenMOSS. The example images were generated with FLUX.2 Klein and MOSS-Image.

Licensed GPL-3.0, like ComfyUI and the upstream nodes.
