"""The README's images: banner, workflow cards and the workflow diagram, in light and dark, rendered from HTML with
headless Chrome at 2x; and the gallery/findings cells (one JPEG each, labels stay in the README as text).

  python dev/readme_assets.py <lab>     # <lab>: the test folder with input/src_*.png and output/sweeps/*

Writes assets/banner-{light,dark}.png, assets/workflow-{light,dark}.png, assets/card-<workflow>-{light,dark}.png,
docs/samples/<task>/*.jpg and docs/findings/{strength,size,base}/*.jpg.
"""
import html
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS, DOCS = ROOT / "assets", ROOT / "docs"
CHROME = "google-chrome"
ACCENT = "#A371F7"
THEMES = {
    "light": {"bg": "#FFFFFF", "card": "#F6F8FA", "ink": "#1F2328", "muted": "#59636E", "line": "#D1D9E0", "accent": "#8250DF",
              "node": "#F6F8FA", "node_title": "#EAEEF2", "widget": "#FFFFFF", "link_alpha": "0.9"},
    "dark": {"bg": "#0D1117", "card": "#151B23", "ink": "#F0F6FC", "muted": "#9198A1", "line": "#3D444D", "accent": ACCENT,
             "node": "#262A30", "node_title": "#1C1F24", "widget": "#16191D", "link_alpha": "0.85"},
}
FONT = "'IBM Plex Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace"
SOCKETS = {"MODEL": "#B39DDB", "CLIP": "#FFD500", "VAE": "#FF6E6E", "IMAGE": "#64B5F6", "CONDITIONING": "#FFA931", "LATENT": "#FF9CF9"}

# gallery tasks: name -> (sources, prompt), as tested (the lab's sweep.py)
TASKS = {
    "background": (["cafe_woman"], "Replace the background with a snowy mountain landscape, keeping the woman exactly the same."),
    "color": (["red_car"], "Change the color of the car to bright blue."),
    "material": (["marble_bust"], "Make the statue look like it is made of polished gold."),
    "pose": (["dog_park"], "Make the dog lie down on the grass."),
    "expression": (["cafe_woman"], "Make the woman smile warmly, showing her teeth."),
    "hair": (["cafe_woman"], "Give the woman short straight blonde hair."),
    "accessory": (["cafe_woman"], "Add black sunglasses to the woman."),
    "style": (["mountain_lake"], "Turn this photo into a watercolor painting."),
    "add": (["living_room"], "Add a glass vase with red roses on the coffee table."),
    "remove": (["beach_person"], "Remove the man from the beach."),
    "replace": (["dog_park"], "Replace the dog with a black cat."),
    "text": (["bakery_sign"], "Change the text on the signs from \"BAKERY\" to \"COFFEE\"."),
    "tone": (["city_street"], "Make it a night scene with the street lamps lit."),
    "multi": (["cafe_woman", "leather_jacket"], "Dress the woman in image 1 in the black leather jacket from image 2."),
}
# workflow cards: name -> (title, inputs, prompt, result), paths relative to the lab
CARDS = {
    "moss_image_edit": ("chinoll · one image", ["input/moss_edit_bicycle.png"],
                        "Paint the bicycle bright red, keeping the ivy, the brick wall and the lighting the same.",
                        "output/readme/moss_image_edit.png"),
    "moss_image_edit_multi_reference": ("chinoll · two images", ["input/src_cafe_woman.png", "input/src_leather_jacket.png"],
                                        TASKS["multi"][1], "output/sweeps/gallery_chinoll/multi.png"),
    "moss_image_edit_hq": ("HQ · one image", ["input/src_red_car.png"], TASKS["color"][1], "output/sweeps/gallery_hq/color.png"),
    "moss_image_edit_hq_multi_reference": ("HQ · two images", ["input/src_cafe_woman.png", "input/src_leather_jacket.png"],
                                           TASKS["multi"][1], "output/sweeps/gallery_hq/multi.png"),
}


def shoot(page, width, height, out):
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "page.html"
        source.write_text(page, encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--allow-file-access-from-files", "--virtual-time-budget=3000",
                        f"--window-size={width},{height}", f"--screenshot={out}", source.as_uri()],
                       check=True, capture_output=True)


def uri(path):
    return Path(path).resolve().as_uri()


def banner(theme, lab):
    t = THEMES[theme]
    # label, task, crop focus (the portrait is framed on her face, not the middle of the picture)
    pairs = [("sunglasses", "accessory", "50% 14%"), ("blue car", "color", "50% 60%"), ("watercolor", "style", "50% 50%")]
    columns = "".join(f"""<div class="col"><img style="object-position:{focus}" src="{uri(lab / 'input' / f'src_{TASKS[task][0][0]}.png')}">
<div class="arrow">↓</div><img style="object-position:{focus}" src="{uri(lab / 'output/sweeps/gallery_chinoll' / f'{task}.png')}"><span>{label}</span></div>"""
                      for label, task, focus in pairs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; width: 1280px; height: 320px; background: {t['bg']}; font-family: {FONT}; color: {t['ink']}; overflow: hidden; position: relative; }}
.text {{ position: absolute; left: 64px; top: 58px; width: 560px; }}
h1 {{ margin: 0; font-size: 44px; font-weight: 700; letter-spacing: -1px; line-height: 1.1; }}
h1 span {{ color: {t['accent']}; }}
p {{ margin: 16px 0 0; font-size: 19px; line-height: 1.45; color: {t['muted']}; }}
.built {{ position: absolute; left: 64px; bottom: 30px; font-size: 15px; color: {t['muted']}; letter-spacing: 0.4px; }}
.bar {{ position: absolute; left: 0; top: 0; width: 8px; height: 320px; background: {ACCENT}; }}
.pairs {{ position: absolute; right: 48px; top: 22px; display: flex; gap: 18px; }}
.col {{ display: flex; flex-direction: column; align-items: center; width: 172px; }}
.col img {{ width: 172px; height: 118px; object-fit: cover; border-radius: 8px; border: 1px solid {t['line']}; }}
.arrow {{ color: {t['accent']}; font-size: 18px; line-height: 22px; height: 22px; }}
.col span {{ margin-top: 8px; font-size: 13px; color: {t['muted']}; font-family: {MONO}; }}
</style></head><body><div class="bar"></div>
<div class="text"><h1>ComfyUI-<span>Moss-Image-Edit</span>-Native</h1>
<p>Edit images with a text instruction on ComfyUI's native MOSS-Image, with the chinoll and HQ edit LoRAs.</p></div>
<div class="built">One to four images · stock loaders and samplers · no extra dependencies</div>
<div class="pairs">{columns}</div></body></html>"""


def card(theme, lab, title, inputs, prompt, result):
    t = THEMES[theme]
    boxes = "".join(f'<div class="box{" half" if len(inputs) > 1 else ""}"><img src="{uri(lab / p)}">'
                    f'<i>{"image " + str(i + 1) if len(inputs) > 1 else "source"}</i></div>' for i, p in enumerate(inputs))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; width: 640px; height: 258px; background: {t['bg']}; font-family: {FONT}; color: {t['ink']}; overflow: hidden; }}
.card {{ position: absolute; inset: 8px; background: {t['card']}; border: 1px solid {t['line']}; border-radius: 12px; padding: 14px 16px; box-sizing: border-box; }}
h2 {{ margin: 0 0 12px; font-size: 16px; font-weight: 600; }}
.row {{ display: flex; align-items: center; gap: 10px; }}
.row > * {{ flex: none; }}
.inputs {{ display: flex; gap: 6px; width: 180px; }}
.box {{ position: relative; width: 180px; height: 180px; border-radius: 8px; background: {t['bg']}; border: 1px solid {t['line']}; overflow: hidden; }}
.box.half {{ width: 87px; }}
.box img {{ width: 100%; height: 100%; object-fit: contain; }}
.box i {{ position: absolute; left: 6px; bottom: 6px; font-style: normal; font-size: 11px; padding: 2px 6px; border-radius: 5px;
          background: rgba(0,0,0,0.55); color: #fff; }}
.prompt {{ width: 150px; font-size: 12px; line-height: 1.45; color: {t['ink']}; font-family: {MONO}; background: {t['bg']};
           border: 1px solid {t['line']}; border-radius: 8px; padding: 10px; box-sizing: border-box; position: relative; }}
.prompt b {{ display: block; font-family: {FONT}; font-size: 11px; font-weight: 600; color: {t['accent']}; margin-bottom: 4px;
             text-transform: uppercase; letter-spacing: 0.6px; }}
.arrow {{ color: {t['accent']}; font-size: 20px; width: 20px; text-align: center; }}
.result {{ border-color: {t['accent']}; }}
</style></head><body><div class="card"><h2>{html.escape(title)}</h2><div class="row">
<div class="inputs">{boxes}</div><div class="arrow">→</div>
<div class="prompt"><b>Prompt</b>{html.escape(prompt)}</div><div class="arrow">→</div>
<div class="box result"><img src="{uri(lab / result)}"><i>result</i></div></div></div></body></html>"""


NODES = [
    # id, title, x, y, width, inputs, outputs, widgets, ours
    ("unet", "Load Diffusion Model", 24, 24, 250, [], [("MODEL", "MODEL")], [("unet_name", "moss_image_sft_bf16")], False),
    ("lora", "LoraLoaderModelOnly", 314, 24, 260, [("model", "MODEL")], [("MODEL", "MODEL")],
     [("lora_name", "chinoll  or  HQ"), ("strength_model", "1.00")], False),
    ("slg", "Skip Layer Guidance DiT Simple", 614, 24, 270, [("model", "MODEL")], [("MODEL", "MODEL")],
     [("single_layers", "22"), ("start · end", "0.00 · 1.00")], False),
    ("clip", "Load CLIP", 24, 200, 250, [], [("CLIP", "CLIP")], [("clip_name", "qwen3.5_9b_bf16"), ("type", "moss_image")], False),
    ("vae", "Load VAE", 24, 590, 250, [], [("VAE", "VAE")], [("vae_name", "flux2-vae")], False),
    ("img1", "Load Image", 24, 362, 250, [], [("IMAGE", "IMAGE")], [("image", "image 1 (edited)")], False),
    ("img2", "Load Image", 24, 476, 250, [], [("IMAGE", "IMAGE")], [("image", "image 2 (optional)")], False),
    ("enc", "MOSS-Image Edit Text Encode", 384, 220, 330, [("clip", "CLIP"), ("vae", "VAE"), ("image_1", "IMAGE"), ("image_2", "IMAGE")],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
     [("prompt", "Dress the woman in image 1 in…"), ("negative_prompt", ""), ("edit_format", "chinoll  or  ai-toolkit (HQ)"),
      ("reference_max_edge", "0  (1 MP)"), ("output_max_edge", "0"), ("kv_cache", "auto")], True),
    ("ks", "KSampler", 944, 200, 270, [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
     [("LATENT", "LATENT")], [("seed", "42"), ("steps", "30"), ("cfg", "5 chinoll · 3 HQ"), ("sampler · scheduler", "euler · simple")], False),
    ("dec", "VAE Decode", 1264, 200, 170, [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")], [], False),
    ("save", "Save Image", 1474, 200, 160, [("images", "IMAGE")], [], [], False),
]
LINKS = [("unet", 0, "lora", 0), ("lora", 0, "slg", 0), ("slg", 0, "ks", 0), ("clip", 0, "enc", 0), ("vae", 0, "enc", 1),
         ("img1", 0, "enc", 2), ("img2", 0, "enc", 3), ("enc", 0, "ks", 1), ("enc", 1, "ks", 2), ("enc", 2, "ks", 3),
         ("ks", 0, "dec", 0), ("vae", 0, "dec", 1), ("dec", 0, "save", 0)]


def diagram(theme):
    t = THEMES[theme]
    nodes = []
    for nid, title, x, y, w, ins, outs, widgets, ours in NODES:
        slots = max(len(ins), len(outs))
        rows = "".join(f'<div class="slot"><span class="in">{f"<em data-k={nid}-i{i} style=background:{SOCKETS[ins[i][1]]}></em>{ins[i][0]}" if i < len(ins) else ""}</span>'
                       f'<span class="out">{f"{outs[i][0]}<em data-k={nid}-o{i} style=background:{SOCKETS[outs[i][1]]}></em>" if i < len(outs) else ""}</span></div>'
                       for i in range(slots))
        wid = "".join(f'<div class="w"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>' for k, v in widgets)
        nodes.append(f'<div class="node{" ours" if ours else ""}" style="left:{x}px;top:{y}px;width:{w}px"><div class="t">{html.escape(title)}</div>'
                     f'{rows}{wid}</div>')
    links = ",".join(f'["{a}-o{ao}","{b}-i{bi}"]' for a, ao, b, bi in LINKS)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin: 0; width: 1660px; height: 730px; background: {t['bg']}; font-family: {FONT}; color: {t['ink']}; overflow: hidden; position: relative; }}
svg {{ position: absolute; inset: 0; }}
.node {{ position: absolute; background: {t['node']}; border: 1px solid {t['line']}; border-radius: 10px; font-size: 12.5px; padding-bottom: 6px;
         box-shadow: 0 2px 6px rgba(0,0,0,0.12); }}
.node.ours {{ border: 2px solid {t['accent']}; }}
.t {{ background: {t['node_title']}; padding: 7px 12px; border-radius: 9px 9px 0 0; font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
.ours .t {{ color: {t['accent']}; }}
.slot {{ display: flex; justify-content: space-between; height: 20px; align-items: center; color: {t['muted']}; }}
.slot span {{ display: flex; align-items: center; gap: 7px; }}
.slot .in {{ margin-left: -5px; }} .slot .out {{ margin-right: -5px; }}
em {{ display: inline-block; width: 10px; height: 10px; border-radius: 5px; }}
.w {{ display: flex; justify-content: space-between; gap: 10px; margin: 4px 10px 0; padding: 4px 10px; background: {t['widget']};
      border-radius: 6px; border: 1px solid {t['line']}; }}
.w span {{ color: {t['muted']}; white-space: nowrap; }} .w b {{ font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.legend {{ position: absolute; left: 944px; top: 70px; width: 420px; line-height: 1.5; font-size: 13px; color: {t['muted']}; }}
.legend b {{ color: {t['accent']}; }}
</style></head><body><svg id="links" width="1660" height="730"></svg>{''.join(nodes)}
<div class="legend"><b>MOSS-Image Edit Text Encode</b> is this pack's node; everything else is stock ComfyUI.</div>
<script>
const BELOW = 712, svg = document.getElementById("links"), pos = k => {{ const r = document.querySelector(`[data-k="${{k}}"]`).getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2, r]; }};
for (const [a, b] of [{links}]) {{
  const [x1, y1, ra] = pos(a), [x2, y2] = pos(b), c = Math.max(50, Math.abs(x2 - x1) / 2);
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  // links that would cross other nodes go around below them
  const d = x2 - x1 > 700 ? `M${{x1}},${{y1}} C${{x1 + 90}},${{y1}} ${{x1 + 90}},${{BELOW}} ${{x1 + 180}},${{BELOW}} L${{x2 - 140}},${{BELOW}} C${{x2 - 60}},${{BELOW}} ${{x2 - 60}},${{y2}} ${{x2}},${{y2}}`
                          : `M${{x1}},${{y1}} C${{x1 + c}},${{y1}} ${{x2 - c}},${{y2}} ${{x2}},${{y2}}`;
  p.setAttribute("d", d);
  p.setAttribute("stroke", document.querySelector(`[data-k="${{a}}"]`).style.background); p.setAttribute("stroke-width", "2.4");
  p.setAttribute("fill", "none"); p.setAttribute("opacity", "{t['link_alpha']}"); svg.appendChild(p);
}}
</script></body></html>"""


def thumb(src, dst, size=640, quality=86):
    if not Path(src).exists():
        print("missing, skipped:", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    im.save(dst, quality=quality, optimize=True, progressive=True)


def cells(lab):
    sweeps = lab / "output" / "sweeps"
    for task, (sources, _) in TASKS.items():
        out = DOCS / "samples" / task
        thumb(lab / "input" / f"src_{sources[0]}.png", out / "source.jpg")
        if len(sources) > 1:
            thumb(lab / "input" / f"src_{sources[1]}.png", out / "source2.jpg")
        thumb(sweeps / "gallery_chinoll" / f"{task}.png", out / "chinoll.jpg")
        thumb(sweeps / "gallery_hq" / f"{task}.png", out / "hq.jpg")
    for task in ("multi", "remove"):
        for lora, prefix in (("chinoll", "c"), ("hq", "h")):
            for strength, config in (("025", "s025"), ("050", "s050"), ("075", "s075"), ("100", "base")):
                thumb(sweeps / f"{prefix}_{config}" / f"{task}.png", DOCS / "findings" / "strength" / f"{task}_{lora}_{strength}.jpg", 400)
    for task in ("text", "color"):
        for variant, config in (("ref1024_out1024", "c_base"), ("ref512_out1024", "c_ref512"), ("ref512_out512", "c_res512")):
            thumb(sweeps / config / f"{task}.png", DOCS / "findings" / "size" / f"{task}_{variant}.jpg", 400)
    for task in ("hair", "replace", "text", "multi"):
        for lora, prefix in (("chinoll", "cbest"), ("hq", "hbest")):
            for model in ("sft", "base"):
                thumb(sweeps / f"{prefix}_{model}_s42" / f"{task}.png", DOCS / "findings" / "base" / f"{task}_{lora}_{model}.jpg", 400)


def main():
    lab = Path(sys.argv[1]).expanduser().resolve()
    ASSETS.mkdir(exist_ok=True)
    cells(lab)
    for theme in THEMES:
        shoot(banner(theme, lab), 1280, 320, ASSETS / f"banner-{theme}.png")
        shoot(diagram(theme), 1660, 730, ASSETS / f"workflow-{theme}.png")
        for name, (title, inputs, prompt, result) in CARDS.items():
            if not (lab / result).exists():
                print("missing, skipped card:", name)
                continue
            shoot(card(theme, lab, title, inputs, prompt, result), 640, 258, ASSETS / f"card-{name}-{theme}.png")
    print("wrote", len(list(ASSETS.glob("*.png"))), "assets,", len(list(DOCS.rglob("*.jpg"))), "docs images")


if __name__ == "__main__":
    main()
