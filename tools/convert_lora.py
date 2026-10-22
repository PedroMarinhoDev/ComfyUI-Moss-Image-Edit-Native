"""Convert the Moss-Image-Edit adapter (adapter_model.safetensors from huggingface.co/chinoll/Moss-Image-Edit) to ComfyUI
LoRA keys, for LoraLoaderModelOnly with the native MOSS-Image model.

    python tools/convert_lora.py adapter_model.safetensors models/loras/moss_image_edit_lora_bf16.safetensors

Checks the adapter against the config embedded in it (format, sha256 when adapter_config.json sits next to it, rank,
alpha, shapes) and writes diffusion_model.layers.N.proj.X.lora_A.weight / lora_B.weight / alpha. Only needs torch and
safetensors. (The pack also loads the unconverted file, with alpha / rank taken as 1.)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moss_edit"))
import lora  # noqa: E402

DTYPES = {"keep": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("adapter", type=Path, help="adapter_model.safetensors")
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=DTYPES, default="keep", help="dtype of lora_A/lora_B (default: as published, bf16)")
    args = parser.parse_args()

    with safe_open(str(args.adapter), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        sd = {key: f.get_tensor(key) for key in f.keys()}
    config = lora.adapter_config(metadata)
    if config is None:
        raise SystemExit("No adapter_config in the file's metadata; is this the published adapter_model.safetensors?")

    sidecar = args.adapter.with_name("adapter_config.json")
    if sidecar.is_file():
        expected = json.loads(sidecar.read_text()).get("weights_sha256")
        with args.adapter.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if expected != actual:
            raise SystemExit("sha256 {} does not match adapter_config.json ({}); incomplete download?".format(actual, expected))
        print("sha256 matches adapter_config.json")

    modules = config["modules"]
    if sorted(modules) != config["targets"] or {k.rpartition(".")[0] for k in sd} != set(modules):
        raise SystemExit("The tensors don't match the adapter's targets.")
    for name, m in modules.items():
        a, b = sd[name + ".lora_A"], sd[name + ".lora_B"]
        if a.shape != (m["rank"], m["in_features"]) or b.shape != (m["out_features"], m["rank"]):
            raise SystemExit("Shape mismatch for {}: A {} B {}".format(name, tuple(a.shape), tuple(b.shape)))
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise SystemExit("Non-finite values in {}".format(name))

    out = lora.to_comfy(sd, config, DTYPES[args.dtype])
    meta = config.get("metadata", {})
    scales = sorted({m["alpha"] / m["rank"] for m in modules.values()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.output), metadata={
        "adapter_config": metadata["adapter_config"],
        "moss_edit_contract": meta.get("edit_contract", ""),
        "converted_by": "ComfyUI-Moss-Image-Edit-Native tools/convert_lora.py",
    })
    print("{} modules, rank {}, alpha/rank {} -> {} ({} tensors)".format(
        len(modules), sorted({m["rank"] for m in modules.values()}), scales, args.output, len(out)))


if __name__ == "__main__":
    main()
