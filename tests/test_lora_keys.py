"""Check that ComfyUI maps every tensor of the converted and of the raw Moss-Image-Edit LoRA onto the native MOSS-Image
model, unfused (bf16 files) and fused qkv (int8 files). The models are built on the meta device: no weights are loaded.

    CUDA_VISIBLE_DEVICES="" ~/AI/ComfyUI/.venv/bin/python tests/test_lora_keys.py --comfy ~/AI/ComfyUI \
        --model models/diffusion_models/moss_image_sft_bf16.safetensors --converted ... --raw adapter_model.safetensors
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

PACK = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="a native MOSS-Image diffusion model (only its header is read)")
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.comfy))
    sys.path.insert(0, str(PACK))
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = True
    import comfy.lora
    import comfy.lora_convert
    import comfy.model_detection
    import comfy.utils
    from moss_edit import patches
    patches.install()

    with safe_open(str(args.model), framework="pt", device="cpu") as f:
        shapes = {k: torch.empty(f.get_slice(k).get_shape(), device="meta") for k in f.keys()}
    unet_config = comfy.model_detection.detect_unet_config(shapes, "")
    print("detected", unet_config)

    def lora_file(path):
        sd = comfy.utils.load_torch_file(str(path))
        return comfy.lora_convert.convert_lora(sd)

    failures = []
    for fused in (False, True):
        config = comfy.model_detection.model_config_from_unet_config(dict(unet_config, fused_qkv=fused), shapes)
        with torch.device("meta"):
            model = config.get_model({}, device=torch.device("meta"))
        key_map = comfy.lora.model_lora_keys_unet(model, {})
        for label, path in (("converted", args.converted), ("raw", args.raw)):
            sd = lora_file(path)
            patches_ = comfy.lora.load_lora(sd, key_map, log_missing=True)
            targets = {k if isinstance(k, str) else k[0] for k in patches_}
            sliced = sum(1 for k in patches_ if not isinstance(k, str))
            expected = sum(1 for k in sd if k.endswith("lora_A.weight"))
            ok = len(patches_) == expected
            print("{:<6} {:<9} {} patches for {} LoRA modules on {} weights ({} as q/k/v slices of a fused weight)".format(
                "fused" if fused else "q/k/v", label, len(patches_), expected, len(targets), sliced))
            if not ok:
                failures.append("{} {}".format(label, fused))
            alphas = {float(v.weights[2]) if v.weights[2] is not None else None for v in patches_.values()}
            print("       alpha values", alphas)

    print("FAILED: " + ", ".join(failures) if failures else "all LoRA keys mapped")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
