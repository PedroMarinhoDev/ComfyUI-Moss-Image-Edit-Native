"""CPU check of the whole path: this pack's conditioning -> the model's extra_conds as the sampler calls them -> the
DiT. Moss-Image-Edit conditioning (the References node) against upstream's MossEditModel; ai-toolkit conditioning
(layout, and image-token attention bias as the Text Encode node sets for image_strength) saved on one ComfyUI and
compared on another. Run it on a ComfyUI with native MOSS edit support (Comfy-Org/ComfyUI#17296 and #17320, ComfyUI
master) with --save, then on one without (a 0.42 release, or master before them), where this pack's hooks do the work, with --compare:

    CUDA_VISIBLE_DEVICES="" python tests/test_conditioning_path.py --comfy <ComfyUI master> ... --save native.pt
    CUDA_VISIBLE_DEVICES="" python tests/test_conditioning_path.py --comfy ~/AI/ComfyUI \
        --pipeline <MOSS-Image-SFT>/pipeline_moss_image.py --upstream <ComfyUI-Moss-Image-Edit at f70d118> --compare native.pt
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

PACK = Path(__file__).resolve().parents[1]


def load_module(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StubVAE:
    """Hands out fixed latents in order, whatever the image size."""

    def __init__(self, latents):
        self.latents = list(latents)

    def encode(self, image):
        return self.latents.pop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True, help="official pipeline_moss_image.py")
    parser.add_argument("--upstream", type=Path, required=True, help="clone of Chenkin-x/ComfyUI-Moss-Image-Edit")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save", type=Path, help="save the ai-toolkit outputs here")
    parser.add_argument("--compare", type=Path, help="compare the ai-toolkit outputs with ones saved by --save")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    sys.path.insert(0, str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    import comfy.model_base
    import comfy.sampler_helpers
    import comfy.supported_models
    import comfy.ldm.moss_image.model as moss

    load_module("moss_edit_native", PACK / "__init__.py", PACK)
    nodes = sys.modules["moss_edit_native.nodes"]
    native = getattr(moss.MOSSImage, "preprocess_refs", None) is not None
    print("ComfyUI has native MOSS edit support:", native, "| pack model hooks active:",
          "_patch_model_base" in comfy.model_base.MOSSImage.extra_conds.__qualname__)

    official = load_module("pipeline_moss_image", args.pipeline)
    runtime = args.upstream / "moss_runtime"
    load_module("moss_runtime", runtime / "__init__.py", runtime)
    upstream = load_module("moss_runtime.model", runtime / "model.py")

    cfg = dict(dim=256, ffn_dim=512, num_heads=2, num_layers=3, text_dim=64, latent_channels=128)
    ref_model = official.MOSSImageTransformer(**cfg, freq_dim=256).eval()
    with torch.no_grad():
        for name, p in ref_model.named_parameters():
            if "norm" in name:
                p.add_(torch.randn_like(p) * 0.1)
    editor = upstream.MossEditModel(ref_model).eval()
    sd = {k: v for k, v in ref_model.state_dict().items() if k != "rope.bands"}

    x = torch.randn(1, 128, 6, 5)
    t = 0.73
    refs = [torch.randn(1, 128, 4, 7), torch.randn(1, 128, 5, 3)]
    pos, neg = torch.randn(1, 9, 64), torch.randn(1, 4, 64)
    with torch.no_grad():
        want = {"positive": editor(x, torch.tensor([t]), pos, reference_latents=refs),
                "negative": editor(x, torch.tensor([t]), neg, reference_latents=refs)}

    failures = []
    saved = torch.load(args.compare) if args.compare else None
    outputs = {}
    bias = torch.zeros(1, pos.shape[1])
    bias[0, 2:6] = 0.7  # as image_strength 2 would put on four image tokens

    def check(label, got, ref, tol=2e-4, expect_match=True):
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        ok = err < tol if expect_match else err >= tol
        print("{:<72} rel err {:.2e}{}".format(label, err, "" if ok else "  <-- unexpected"))
        if not ok:
            failures.append(label)

    for fused in (False, True):
        model_config = comfy.supported_models.MOSSImage({"image_model": "moss_image", "fused_qkv": fused, **cfg})
        model_config.set_inference_dtype(torch.float32, None)
        model = comfy.model_base.MOSSImage(model_config, device="cpu")
        state = dict(sd)
        if fused:
            for i in range(cfg["num_layers"]):
                p = "layers.{}.proj.".format(i)
                state[p + "qkv.weight"] = torch.cat([state.pop(p + n + ".weight") for n in "qkv"])
        model.diffusion_model.load_state_dict(state, strict=False)
        tag = "fused" if fused else "q/k/v"

        images = {"image_1": torch.rand(1, 64, 112, 3), "image_2": torch.rand(1, 80, 48, 3)}
        result = nodes.MOSSImageEditReferences.execute([[pos, {}]], [[neg, {}]], StubVAE(refs), images, 1024, "auto")
        conditionings = dict(zip(("positive", "negative"), result.args[:2]))

        def run(conditioning, drop=()):
            cond = comfy.sampler_helpers.convert_cond(conditioning)[0]
            for key in drop:
                cond.pop(key, None)
            with torch.inference_mode():  # as ComfyUI samples, the native code precomputes the references only then
                conds = model.extra_conds(**cond, device=torch.device("cpu"), noise=x)
                inputs = {k: v.process_cond(batch_size=1, device=torch.device("cpu")).cond for k, v in conds.items()}
                context = inputs.pop("c_crossattn")
                return model.diffusion_model(x, torch.tensor([t]), context, transformer_options={}, **inputs), sorted(inputs)

        for name, conditioning in conditionings.items():
            got, keys = run(conditioning)
            check(f"[{tag}] {name}: References node -> extra_conds -> DiT  {keys}", got, want[name])
        if native:
            # what the native code does with references that carry no method (stock Set Reference Latent)
            got, keys = run(conditionings["positive"], drop=("reference_latents_method",))
            check(f"[{tag}] positive without the method (ai-toolkit layout, should differ)", got, want["positive"], expect_match=False)

        # ai-toolkit conditioning, as the Text Encode node builds it
        for name, values in (("ai-toolkit", {}), ("ai-toolkit + image-token bias", {"context_attention_bias": bias})):
            conditioning = [[pos, {"reference_latents": refs, "reference_latents_method": "index_timestep_zero_isolated", **values}]]
            got, keys = run(conditioning)
            outputs[f"{tag} {name}"] = got
            if saved is not None:
                check(f"[{tag}] {name}: same as the saved run  {keys}", got, saved[f"{tag} {name}"])

    if args.save:
        torch.save(outputs, args.save)
        print("saved", sorted(outputs), "to", args.save)
    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
