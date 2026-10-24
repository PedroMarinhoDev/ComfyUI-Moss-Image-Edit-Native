"""CPU check of the native edit forward against the code each edit LoRA was trained with: upstream's MossEditModel on
the official MOSSImageTransformer (Moss-Image-Edit layout) and ai-toolkit's MossImageTransformer with isolate_refs
(ai-toolkit layout, MOSS-Image-Edit-HQ).

All get the same random weights, at a small width with the real head size (128), in float32:

    CUDA_VISIBLE_DEVICES="" ~/AI/ComfyUI/.venv/bin/python tests/test_equivalence.py \
        --comfy ~/AI/ComfyUI --pipeline <MOSS-Image-SFT>/pipeline_moss_image.py --upstream <ComfyUI-Moss-Image-Edit clone> \
        --aitk <ai-toolkit>/extensions_built_in/diffusion_models/moss_image/src/transformer.py
"""
import types
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]


def load_module(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True, help="official pipeline_moss_image.py")
    parser.add_argument("--upstream", type=Path, required=True, help="clone of Chenkin-x/ComfyUI-Moss-Image-Edit")
    parser.add_argument("--aitk", type=Path, help="ai-toolkit's moss_image/src/transformer.py, for the ai-toolkit layout")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    sys.path.insert(0, str(args.comfy))
    sys.path.insert(0, str(PACK))
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    import comfy.ldm.moss_image.model as moss
    import comfy.ops
    from moss_edit import patches
    from moss_edit.model import ISOLATED, TEXT_START
    patches.install()

    official = load_module("pipeline_moss_image", args.pipeline)
    runtime = args.upstream / "moss_runtime"
    load_module("moss_runtime", runtime / "__init__.py", runtime)
    upstream = load_module("moss_runtime.model", runtime / "model.py")

    cfg = dict(dim=256, ffn_dim=512, num_heads=2, num_layers=3, text_dim=64, latent_channels=128)
    ref_model = official.MOSSImageTransformer(**cfg, freq_dim=256).eval()
    with torch.no_grad():  # move the norms off 1.0 so their weights are exercised
        for name, p in ref_model.named_parameters():
            if "norm" in name:
                p.add_(torch.randn_like(p) * 0.1)
    editor = upstream.MossEditModel(ref_model).eval()
    sd = {k: v for k, v in ref_model.state_dict().items() if k != "rope.bands"}

    def native(fused):
        model = moss.MOSSImage(**cfg, fused_qkv=fused, operations=comfy.ops.disable_weight_init).eval()
        state = dict(sd)
        if fused:
            for i in range(cfg["num_layers"]):
                p = "layers.{}.proj.".format(i)
                state[p + "qkv.weight"] = torch.cat([state.pop(p + n + ".weight") for n in "qkv"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all("comfy_attention" in k for k in missing), (missing, unexpected)
        return model

    def run_options(sample_sigmas, uuids, sigma):
        return {"sample_sigmas": sample_sigmas, "uuids": uuids, "sigmas": torch.tensor([sigma])}

    H, W = 6, 5
    refs = [torch.randn(1, 128, 4, 7), torch.randn(1, 128, 5, 3)]
    pos, neg = torch.randn(1, 9, 64), torch.randn(1, 4, 64)
    x = torch.randn(1, 128, H, W)
    t = 0.73

    failures = []

    def check(label, got, want, tol=2e-4, expect_match=True):
        err = (got - want).abs().max().item() / want.abs().max().item()
        ok = err < tol if expect_match else err >= tol
        print("{:<72} rel err {:.2e}{}".format(label, err, "" if ok else "  <-- unexpected"))
        if not ok:
            failures.append(label)

    with torch.inference_mode(False), torch.no_grad():
        want_plain = ref_model(x, torch.tensor([t * 1000]), pos, return_dict=False)[0]
        want_pos = editor(x, torch.tensor([t]), pos, reference_latents=refs)
        want_neg = editor(x, torch.tensor([t]), neg, reference_latents=refs)
        want_one = editor(x, torch.tensor([t]), pos, reference_latents=refs[:1])
        cache = editor.prepare_cache(refs)
        check("upstream: cached == uncached", editor(x, torch.tensor([t]), pos, reference_latents=refs, cache=cache), want_pos, 1e-5)

    with torch.no_grad():
        for fused in (False, True):
            model = native(fused)
            tag = "fused qkv" if fused else "q/k/v"
            ts = torch.tensor([t])
            check("[{}] no references == official transformer".format(tag), model(x, ts, pos), want_plain)
            check("[{}] joint (no cache), 2 refs".format(tag), model(x, ts, pos, ref_latents_method=TEXT_START, ref_latents=refs, moss_edit_kv_cache="off"), want_pos)
            check("[{}] 1 ref".format(tag), model(x, ts, pos, ref_latents_method=TEXT_START, ref_latents=refs[:1]), want_one)

            # one sampling run: cond then uncond at the first sigma, then a later step; one prefill shared by both
            sigmas = torch.tensor([0.9, t, 0.0])
            calls = []
            original = sys.modules["moss_edit.model"].prefill
            sys.modules["moss_edit.model"].prefill = lambda *a, **k: calls.append(1) or original(*a, **k)
            try:
                check("[{}] cached run, cond @0.9".format(tag), model(x, torch.tensor([0.9]), pos, transformer_options=run_options(sigmas, ["c"], 0.9), ref_latents_method=TEXT_START, ref_latents=refs, moss_edit_kv_cache="on"),
                      editor(x, torch.tensor([0.9]), pos, reference_latents=refs))
                check("[{}] cached run, uncond @0.9".format(tag), model(x, torch.tensor([0.9]), neg, transformer_options=run_options(sigmas, ["u"], 0.9), ref_latents_method=TEXT_START, ref_latents=refs, moss_edit_kv_cache="on"),
                      editor(x, torch.tensor([0.9]), neg, reference_latents=refs))
                check("[{}] cached run, cond @t".format(tag), model(x, ts, pos, transformer_options=run_options(sigmas, ["c"], t), ref_latents_method=TEXT_START, ref_latents=refs, moss_edit_kv_cache="on"), want_pos)
                assert len(calls) == 1, "expected one prefill per run, got {}".format(len(calls))
                # the next run reuses the same sigmas tensor with new conds: it must not reuse the keys/values
                model(x, torch.tensor([0.9]), pos, transformer_options=run_options(sigmas, ["c2"], 0.9), ref_latents_method=TEXT_START, ref_latents=refs, moss_edit_kv_cache="on")
                assert len(calls) == 2, "a new run with the same sigmas tensor reused the cache"
                del sigmas
                assert not getattr(model, "_moss_edit_runs", {}), "the cache outlived its sigmas tensor"
            finally:
                sys.modules["moss_edit.model"].prefill = original

            # cond and uncond batched together: same references, different text lengths can't batch, so pad by reusing pos
            xb = torch.cat([x, x])
            refs_b = [torch.cat([r, r]) for r in refs]
            got = model(xb, torch.tensor([t, t]), torch.cat([pos, pos]), transformer_options=run_options(torch.tensor([t, 0.0]), ["a", "b"], t), ref_latents_method=TEXT_START, ref_latents=refs_b, moss_edit_kv_cache="on")
            check("[{}] batch of 2 sharing references".format(tag), got, torch.cat([want_pos, want_pos]))
            # rows with different references in one batch
            other = [torch.randn_like(r) for r in refs]
            with torch.inference_mode(False):
                want_other = editor(x, torch.tensor([t]), pos, reference_latents=other)
            mixed = [torch.cat([a, b]) for a, b in zip(refs, other)]
            for mode in ("on", "off"):
                got = model(xb, torch.tensor([t, t]), torch.cat([pos, pos]), transformer_options=run_options(torch.tensor([t, 0.0]), ["m" + mode, "n" + mode], t), ref_latents_method=TEXT_START, ref_latents=mixed, moss_edit_kv_cache=mode)
                check("[{}] batch of 2, different references, cache {}".format(tag, mode), got, torch.cat([want_pos, want_other]))

            # a skip-layer-guidance style block replacement that only calls the original block
            replace = {("single_block", 1): lambda args, extra: extra["original_block"](args)}
            got = model(x, ts, pos, transformer_options={"patches_replace": {"dit": replace}}, ref_latents_method=TEXT_START, ref_latents=refs)
            check("[{}] block replacement passes references through".format(tag), got, want_pos)

    if args.aitk:
        # ai-toolkit's transformer only needs its model mixin from the rest of ai-toolkit
        for name in ("toolkit", "toolkit.models", "toolkit.models.v2", "toolkit.models.v2._mixin"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["toolkit.models.v2._mixin"].OstrisModelMixin = type("OstrisModelMixin", (), {})
        aitk = load_module("aitk_moss_transformer", args.aitk)
        trainer = aitk.MossImageTransformer(aitk.MossImageParams(**cfg)).eval()
        state = dict(sd)
        for i in range(cfg["num_layers"]):
            p = "layers.{}.proj.".format(i)
            state[p + "qkv.weight"] = torch.cat([state.pop(p + n + ".weight") for n in "qkv"])
        trainer.load_state_dict(state, strict=True)
        with torch.no_grad():
            per_sample = [[r[0] for r in refs]]
            want_aitk = trainer(x, torch.tensor([t * 1000]), pos, ref_latents=per_sample, isolate_refs=True)
            want_aitk_neg = trainer(x, torch.tensor([t * 1000]), neg, ref_latents=per_sample, isolate_refs=True)
            want_aitk_one = trainer(x, torch.tensor([t * 1000]), pos, ref_latents=[[refs[0][0]]], isolate_refs=True)
            cache = {}
            trainer(x, torch.tensor([0.9 * 1000]), pos, ref_latents=per_sample, isolate_refs=True, ref_kv_cache=cache)
            check("ai-toolkit: cached == uncached", trainer(x, torch.tensor([t * 1000]), pos, ref_latents=per_sample, isolate_refs=True, ref_kv_cache=cache), want_aitk, 1e-5)
            check("ai-toolkit layout differs from Moss-Image-Edit's", want_aitk, want_pos, expect_match=False)
            for fused in (False, True):
                model = native(fused)
                tag = "fused qkv" if fused else "q/k/v"
                ts = torch.tensor([t])
                check("[{}] ai-toolkit layout, 2 refs, no cache".format(tag), model(x, ts, pos, ref_latents_method=ISOLATED, ref_latents=refs, moss_edit_kv_cache="off"), want_aitk)
                check("[{}] ai-toolkit layout is the default without a method".format(tag), model(x, ts, pos, ref_latents=refs, moss_edit_kv_cache="off"), want_aitk)
                check("[{}] ai-toolkit layout, 1 ref".format(tag), model(x, ts, pos, ref_latents_method=ISOLATED, ref_latents=refs[:1]), want_aitk_one)
                sigmas = torch.tensor([0.9, t, 0.0])
                check("[{}] ai-toolkit layout, cached run, cond @0.9".format(tag),
                      model(x, torch.tensor([0.9]), pos, transformer_options=run_options(sigmas, ["c"], 0.9), ref_latents_method=ISOLATED, ref_latents=refs, moss_edit_kv_cache="on"),
                      trainer(x, torch.tensor([900.0]), pos, ref_latents=per_sample, isolate_refs=True))
                check("[{}] ai-toolkit layout, cached run, uncond @0.9 (other text length)".format(tag),
                      model(x, torch.tensor([0.9]), neg, transformer_options=run_options(sigmas, ["u"], 0.9), ref_latents_method=ISOLATED, ref_latents=refs, moss_edit_kv_cache="on"),
                      trainer(x, torch.tensor([900.0]), neg, ref_latents=per_sample, isolate_refs=True))
                check("[{}] ai-toolkit layout, cached run, cond @t".format(tag),
                      model(x, ts, pos, transformer_options=run_options(sigmas, ["c"], t), ref_latents_method=ISOLATED, ref_latents=refs, moss_edit_kv_cache="on"), want_aitk)
                del sigmas
                # a zero attention bias on the text keys is no bias
                zero = torch.zeros(1, pos.shape[1])
                check("[{}] ai-toolkit layout, zero text attention bias".format(tag), model(x, ts, pos, ref_latents_method=ISOLATED, ref_latents=refs, context_attention_bias=zero), want_aitk)
                # the bias reaches the attention (its exact effect is checked against ComfyUI's own code in test_conditioning_path.py)
                biased =model(x, ts, pos, ref_latents_method=ISOLATED, ref_latents=refs, context_attention_bias=torch.full((1, pos.shape[1]), -30.0))
                check("[{}] ai-toolkit layout, -30 text attention bias changes the output".format(tag), biased, want_aitk, expect_match=False)

    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
