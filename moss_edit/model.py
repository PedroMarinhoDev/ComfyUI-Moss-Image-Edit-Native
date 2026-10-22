"""Reference editing on ComfyUI's native MOSS-Image DiT, for the Moss-Image-Edit LoRA (huggingface.co/chinoll/Moss-Image-Edit).

The LoRA's edit contract, "ref-prefix-only-t0__target-text-original-moss-v1":
- the reference latents go through the same blocks at timestep 0, with RoPE position (r=index+1, f=0, h, w centered),
  and attend only to each other;
- the target and text tokens keep the stock joint attention, timestep and RoPE, and also attend to the reference keys/values.

The reference branch never sees the text or the target, so its per-layer keys/values are the same at every step and for
cond and uncond. They are computed once per sampling run and reused (upstream's prepare_cache), unless they don't fit.
"""
import logging
import math
import weakref

import comfy_aimdo.model_vbar
import torch

import comfy.ldm.flux.math
import comfy.ldm.moss_image.model as moss
import comfy.memory_management
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

KV_CACHE = "moss_edit_kv_cache"  # conditioning key and model input: "auto", "on" or "off"
KV_CACHE_MODES = ("auto", "on", "off")
KV_CACHE_FREE_FRACTION = 0.5  # "auto" caches when the keys/values take at most this share of the free memory


def norm_rope(block, q, k, pe):
    # moss.attention up to the attention call: per-head RMSNorm and split-half RoPE
    B, N, _ = q.shape
    q = q.view(B, N, block.num_heads, -1)
    k = k.view(B, N, block.num_heads, -1)
    if comfy.model_management.in_training:
        q, k = moss.rope_split_half(block.q_norm(q), pe), moss.rope_split_half(block.k_norm(k), pe)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(block.q_norm, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(block.k_norm, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, block.q_norm.eps)
        comfy.ops.uncast_bias_weight(block.q_norm, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(block.k_norm, k_scale, None, k_stream)
    return q.flatten(2), k.flatten(2)


def qkv(block, x, mod):
    y = moss.modulated_rms_norm(x, block.norm1, mod[1], mod[0])
    if block.fused_qkv:
        return block.proj["qkv"](y).chunk(3, dim=-1)
    return block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)


def attend(block, q, k, v, transformer_options):
    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, block.num_heads, transformer_options=transformer_options, preferred_attention=block.comfy_attention)


def finish(block, x, out, mod):
    # moss.Block.forward after the attention
    x = moss.gated_residual(x, block.proj["o"](out), mod[2])
    y = moss.modulated_rms_norm(x, block.norm2, mod[4], mod[3])
    return moss.gated_residual(x, comfy.ops.linear_input_act(block.proj["ffn_down"], block.proj["ffn_up_gate"](y), "situ"), mod[5])


def reference_kv(block, r, mod, pe):
    q, k, v = qkv(block, r, mod)
    q, k = norm_rope(block, q, k, pe)
    # a fused qkv chunk is a view; a contiguous v doesn't keep the whole qkv output alive in the cache
    return q, k, v.contiguous()


def reference_block(block, r, mod, pe, transformer_options, last=False):
    """One block of the reference branch: the updated stream (None after the last block) and this layer's (k, v)."""
    q, k, v = reference_kv(block, r, mod, pe)
    if last:  # nothing reads the stream after the last block
        return None, (k, v)
    out = attend(block, q, k, v, transformer_options)
    return finish(block, r, out, mod), (k, v)


def body_block(block, x, mod, pe, ref_kv, transformer_options):
    """moss.Block.forward where the target and text also attend to the reference keys/values."""
    q, k, v = qkv(block, x, mod)
    q, k = norm_rope(block, q, k, pe)
    k = torch.cat((k, ref_kv[0]), dim=1)
    v = torch.cat((v, ref_kv[1]), dim=1)
    return finish(block, x, attend(block, q, k, v, transformer_options), mod)


def reference_rope(model, shapes, device, transformer_options={}):
    # moss.MOSSImage.rope for the references: r = index + 1, f = 0, rows and columns centered
    rope_options = transformer_options.get("rope_options", {})
    scales = (1.0, 1.0, rope_options.get("scale_y", 1.0), rope_options.get("scale_x", 1.0))
    ids = []
    for index, (h, w) in enumerate(shapes):
        part = torch.zeros(h * w, 4, device=device, dtype=torch.float32)
        part[:, 0] = index + 1
        part[:, 2] = ((torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * float(scales[2] != 0)).repeat_interleave(w)
        part[:, 3] = ((torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * float(scales[3] != 0)).repeat(h)
        ids.append(part)
    ids = torch.cat(ids).unsqueeze(0)
    pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000 * (s or 1.0) ** (-d / (d - 2))) for i, (d, s) in enumerate(zip(model.rope_axes_dim, scales))], dim=-3)
    return pe.unsqueeze(1).transpose(1, 2).contiguous()


def reference_inputs(model, refs, dtype, device, transformer_options):
    """Embedded reference tokens [U, N, dim] for refs (one [U, C, h, w] tensor per image), their RoPE and the t=0 modulation."""
    tokens = []
    for ref in refs:
        t = ref.to(device=device, dtype=dtype).flatten(2).transpose(1, 2)
        extra = model.video_embedder.weight.shape[1] - t.shape[-1]
        if extra > 0:  # color/depth control adapters widen video_embedder; zeros are no control
            t = torch.nn.functional.pad(t, (0, extra))
        tokens.append(t)
    r = model.video_embedder(torch.cat(tokens, dim=1))
    pe = reference_rope(model, [ref.shape[-2:] for ref in refs], device, transformer_options)
    vec = model.time_embedding(timestep_embedding(torch.zeros(r.shape[0], device=device), 256).to(dtype))
    mod = model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    return r, pe, mod


def prefill(model, refs, dtype, device, transformer_options):
    """Run the reference branch once through every block: the per-layer (k, v), batch 1 (upstream's prepare_cache)."""
    r, pe, mod = reference_inputs(model, refs, dtype, device, transformer_options)
    layers = list(model.layers)
    kv = []
    queue = comfy.model_prefetch.make_prefetch_queue(layers, device, transformer_options)
    for i, block in enumerate(layers):
        comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype)
        r, layer_kv = reference_block(block, r, mod, pe, transformer_options, last=i == len(layers) - 1)
        kv.append(layer_kv)
    comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
    return kv


def reference_sets(ref_latents, batch):
    """The distinct reference sets among the batch rows and the set of each row. Cond and uncond usually share one."""
    sets, rows = [], []
    for b in range(batch):
        row = [ref[b:b + 1] for ref in ref_latents]
        for i, s in enumerate(sets):
            if all(a.shape == c.shape and torch.equal(a, c) for a, c in zip(row, s)):
                rows.append(i)
                break
        else:
            rows.append(len(sets))
            sets.append(row)
    return sets, rows


class _Run:
    """Reference keys/values of one sampling run, dropped when the run's sample_sigmas tensor is freed."""

    def __init__(self, sample_sigmas, on_free):
        self.sigmas = weakref.ref(sample_sigmas, on_free)
        self.uuids = set()
        self.min_sigma = math.inf
        self.entries = []  # (reference set, per-layer kv)
        self.uncached = False  # "auto" found too little memory: this run recomputes the references

    def find(self, refs):
        for s, kv in self.entries:
            if len(s) == len(refs) and all(a.shape == c.shape and a.device == c.device and torch.equal(a, c) for a, c in zip(s, refs)):
                return kv
        return None


def current_run(model, device, transformer_options):
    """The cache of the sampling run this call belongs to, or None when the call isn't part of one."""
    sample_sigmas = transformer_options.get("sample_sigmas", None)
    if sample_sigmas is None:
        return None
    runs = getattr(model, "_moss_edit_runs", None)
    if runs is None:
        runs = {}
        model._moss_edit_runs = runs
    run = runs.get(device)
    uuids = set(transformer_options.get("uuids", ()))
    sigmas = transformer_options.get("sigmas", None)
    sigma = float(sigmas.max()) if sigmas is not None else None
    same = run is not None and run.sigmas() is sample_sigmas
    if same and not uuids <= run.uuids:
        # conds are new per run (sampler_helpers.convert_cond). New ones join this run at the step it has reached
        # (cond and uncond at the first step, timestep ranges later); at a higher sigma they are the next run
        # reusing a sigmas tensor that stayed alive, so the weights may have changed since.
        same = sigma is None or sigma <= run.min_sigma
    if not same:
        def on_free(ref, runs=runs, device=device):
            if device in runs and runs[device].sigmas is ref:
                del runs[device]
        run = _Run(sample_sigmas, on_free)
        runs[device] = run
    run.uuids |= uuids
    if sigma is not None:
        run.min_sigma = min(run.min_sigma, sigma)
    return run


def free_memory(device):
    # as ModelPatcher.get_free_memory: with dynamic VRAM, the staged weights count as free since they can be evicted
    free = comfy.model_management.get_free_memory(device)
    if comfy.memory_management.aimdo_enabled and getattr(device, "type", None) == "cuda":
        free += comfy_aimdo.model_vbar.vbars_analyze(device.index)
    return free


def cached_reference_kv(model, sets, dtype, device, transformer_options, mode="auto"):
    """Per-set per-layer (k, v) from this run's cache, prefilling the missing sets. None means run the branch jointly instead."""
    if mode == "off" or comfy.model_management.in_training:
        return None
    run = current_run(model, device, transformer_options)
    if run is None or run.uncached:
        return None
    out = []
    for refs in sets:
        kv = run.find(refs)
        if kv is None:
            tokens = sum(ref.shape[-2] * ref.shape[-1] for ref in refs)
            dim = model.shared_block_modulation.weight.shape[0] // 6
            size = len(model.layers) * 2 * tokens * dim * comfy.model_management.dtype_size(dtype)
            if mode == "auto" and size > free_memory(device) * KV_CACHE_FREE_FRACTION:
                logging.info("MOSS-Image Edit: %d reference tokens need %.0f MB of keys/values, more than the free memory allows; "
                             "recomputing the references at every step instead.", tokens, size / (1024 * 1024))
                run.uncached = True
                return None
            kv = prefill(model, refs, dtype, device, transformer_options)
            run.entries.append(([ref.clone() for ref in refs], kv))
            logging.info("MOSS-Image Edit: cached the keys/values of %d reference tokens (%.0f MB) for this run.", tokens, size / (1024 * 1024))
        out.append(kv)
    return out


def layer_kv(kvs, rows, i, batch):
    if len(kvs) == 1:
        k, v = kvs[0][i]
        return k.expand(batch, -1, -1), v.expand(batch, -1, -1)
    return torch.cat([kvs[r][i][0] for r in rows]), torch.cat([kvs[r][i][1] for r in rows])


def forward_edit(self, x, timestep, context, transformer_options, ref_latents, kv_cache="auto"):
    """moss.MOSSImage._forward with reference latents (a list of [B, C, h, w])."""
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    pe = self.rope(H, W, context.shape[1], device, transformer_options)

    vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

    txt = self.text_embedder(context, pe[:, H * W:], transformer_options)
    sets, rows = reference_sets(ref_latents, B)
    kvs = cached_reference_kv(self, sets, dtype, device, transformer_options, kv_cache)
    if kvs is None:
        # no cache: the reference sets run as a batch of their own, block by block next to the target
        r, ref_pe, ref_mod = reference_inputs(self, [torch.cat([s[j] for s in sets]) for j in range(len(ref_latents))], dtype, device, transformer_options)
        ref_index = torch.tensor(rows, device=device)
    # allocated outside the memory compiler region, the blocks update it in place
    x = torch.cat((self.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
    del txt

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(self.layers)
    transformer_options["block_type"] = "single"
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, transformer_options)
    comfy.model_prefetch.malloc_graph_begin(device)
    for i, block in enumerate(self.layers):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
        transformer_options["block_index"] = i
        if kvs is not None:
            ref_kv = layer_kv(kvs, rows, i, B)
        else:
            r, (k, v) = reference_block(block, r, ref_mod, ref_pe, transformer_options, last=i == len(self.layers) - 1)
            ref_kv = (k.expand(B, -1, -1), v.expand(B, -1, -1)) if len(sets) == 1 else (k[ref_index], v[ref_index])
            del k, v
        if ("single_block", i) in blocks_replace:
            def block_wrap(args, block=block, ref_kv=ref_kv):
                return {"img": body_block(block, args["img"], args["mod"], args["pe"], ref_kv, args["transformer_options"])}
            x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
        else:
            x = body_block(block, x, mod, pe, ref_kv, transformer_options)
        del ref_kv
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
    comfy.model_prefetch.malloc_graph_end()

    out = self.final_linear_video(moss.modulated_rms_norm(x[:, :H * W], self.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)
