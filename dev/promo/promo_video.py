"""The promo video for X and Reddit: 1920x1080, 60 fps, 62 s, quiet sound effects only (no music).

    python dev/promo/promo_video.py --stills 2,7,20,30,38,47,53,59     # check frames (out/still_*.png)
    python dev/promo/promo_video.py                                     # out/moss-image-edit-promo.mp4 (+ -silent.mp4)

The picture is promo.html + promo.js rendered frame by frame by headless Chrome; every frame is a pure function of
its time, so the frames are split between several Chrome processes (--workers) and written as JPEGs, then encoded
with NVENC (h264_nvenc on the GPU given by --gpu, CUDA's PCI bus order; libx264 if NVENC is missing). The pictures
are the README gallery's sources and seed-42 results from the test lab (MOSS_EDIT_LAB, see PICTURES) and the
README's findings and node diagram. Same design and renderer as ComfyUI-Hy-Nomas's promo. Needs ComfyUI's Python
(aiohttp, numpy, scipy, Pillow), google-chrome and ffmpeg.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD, OUT = HERE / "build", HERE / "out"
LAB = Path(os.environ.get("MOSS_EDIT_LAB", Path.home() / "AI/moss-edit-lab"))
SRC, CHINOLL, HQ = LAB / "input", LAB / "output/sweeps/gallery_chinoll", LAB / "output/sweeps/gallery_hq"
FINDINGS = ROOT / "docs/findings"
FONTS = Path.home() / "Projects" / "music-visualizer" / "public" / "fonts"
PLEX = Path("/usr/share/fonts/truetype/ibm-plex")
FPS, W, H = 60, 1920, 1080
SR = 48000
PORT, CDP0 = 8196, 9440
NAME = "moss-image-edit-promo"

# key -> (picture, longest edge in the build)
PICTURES = {
    "woman_src": (SRC / "src_cafe_woman.png", 1400), "glasses_out": (CHINOLL / "accessory.png", 1400),
    "bike_src": (SRC / "moss_edit_bicycle.png", 1400), "bike_out": (LAB / "output/readme/moss_image_edit.png", 1400),
    "jacket_src": (SRC / "src_leather_jacket.png", 1000), "multi_c": (CHINOLL / "multi.png", 1000), "multi_h": (HQ / "multi.png", 1000),
    "bg_c": (CHINOLL / "background.png", 1200), "bg_h": (HQ / "background.png", 1200),
    "str_025": (FINDINGS / "strength/multi_chinoll_025.jpg", 600), "str_050": (FINDINGS / "strength/multi_chinoll_050.jpg", 600),
    "str_075": (FINDINGS / "strength/multi_chinoll_075.jpg", 600), "str_100": (FINDINGS / "strength/multi_chinoll_100.jpg", 600),
    "size_small": (FINDINGS / "size/text_ref512_out1024.jpg", 800), "size_match": (FINDINGS / "size/text_ref1024_out1024.jpg", 800),
    "base_base": (FINDINGS / "base/multi_hq_base.jpg", 800), "base_sft": (FINDINGS / "base/multi_hq_sft.jpg", 800),
    "diagram": (ROOT / "assets/workflow-dark.png", 3456),
}
# the grid of edit types: task, name, LoRA shown, prompt, crop focus (seed-42 gallery results)
TILES = [
    ("background", "Background", "chinoll", "Replace the background with a snowy mountain landscape, keeping the woman exactly the same.", "50% 30%"),
    ("tone", "Time of day", "chinoll", "Make it a night scene with the street lamps lit.", "50% 55%"),
    ("style", "Style", "chinoll", "Turn this photo into a watercolor painting.", "50% 50%"),
    ("color", "Color", "chinoll", "Change the color of the car to bright blue.", "50% 60%"),
    ("material", "Material", "chinoll", "Make the statue look like it is made of polished gold.", "50% 45%"),
    ("add", "Add", "chinoll", "Add a glass vase with red roses on the coffee table.", "60% 60%"),
    ("remove", "Remove", "HQ", "Remove the man from the beach.", "50% 60%"),
    ("replace", "Replace", "HQ", "Replace the dog with a black cat.", "50% 60%"),
    ("expression", "Expression", "chinoll", "Make the woman smile warmly, showing her teeth.", "50% 26%"),
    ("hair", "Hair", "chinoll", "Give the woman short straight blonde hair.", "50% 26%"),
]
TILE_SOURCES = {"background": "cafe_woman", "tone": "city_street", "style": "mountain_lake", "color": "red_car",
                "material": "marble_bust", "add": "living_room", "remove": "beach_person", "replace": "dog_park",
                "expression": "cafe_woman", "hair": "cafe_woman"}


# --------------------------------------------------------------------------------------------- #
# Page assets
# --------------------------------------------------------------------------------------------- #
def picture(src: Path, key: str, edge: int, sizes: dict) -> None:
    if not src.exists():
        sys.exit(f"missing {src}")
    im = Image.open(src).convert("RGB")
    k = min(1.0, edge / max(im.size))
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    im.save(BUILD / "img" / f"{key}.jpg", quality=93)
    sizes[key] = {"w": im.width, "h": im.height}


def build() -> None:
    (BUILD / "img").mkdir(parents=True, exist_ok=True)
    for name in ("promo.html", "promo.js"):
        shutil.copyfile(HERE / name, BUILD / name)
    (BUILD / "fonts").mkdir(exist_ok=True)
    shutil.copyfile(FONTS / "anton-latin-400-normal.woff2", BUILD / "fonts" / "anton.woff2")
    shutil.copyfile(FONTS / "oswald-latin-wght-normal.woff2", BUILD / "fonts" / "oswald.woff2")
    for f in ("IBMPlexSans-Regular", "IBMPlexSans-Medium", "IBMPlexSans-SemiBold", "IBMPlexSans-Bold",
              "IBMPlexMono-Regular", "IBMPlexMono-Medium"):
        shutil.copyfile(PLEX / f"{f}.ttf", BUILD / "fonts" / f"{f}.ttf")
    sizes: dict = {}
    for key, (src, edge) in PICTURES.items():
        picture(src, key, edge, sizes)
    tiles = []
    for task, name, lora, text, pos in TILES:
        picture(SRC / f"src_{TILE_SOURCES[task]}.png", f"{task}_src", 760, sizes)
        picture((HQ if lora == "HQ" else CHINOLL) / f"{task}.png", f"{task}_out", 760, sizes)
        tiles.append({"task": task, "name": name, "lora": lora, "text": text, "pos": pos})
    mosaic = []
    for gallery, tag in ((CHINOLL, "c"), (HQ, "h")):
        for p in sorted(gallery.glob("*.png")):
            # leave out what didn't work at seed 42 (pose and text for both, remove and replace for chinoll)
            if p.stem in ("pose", "text") or (tag == "c" and p.stem in ("remove", "replace")):
                continue
            picture(p, f"mos_{tag}_{p.stem}", 560, sizes)
            mosaic.append(f"mos_{tag}_{p.stem}")
    (BUILD / "data.json").write_text(json.dumps({"img": sizes, "tiles": tiles, "mosaic": mosaic}, indent=1))
    print(f"{len(sizes)} pictures, {len(tiles)} tiles, {len(mosaic)} in the mosaic")


def serve() -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(BUILD))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


# --------------------------------------------------------------------------------------------- #
# Rendering through the DevTools protocol
# --------------------------------------------------------------------------------------------- #
class Page:
    def __init__(self, ws):
        self.ws, self.seq = ws, 0

    async def call(self, method, **params):
        self.seq += 1
        await self.ws.send_json({"id": self.seq, "method": method, "params": params})
        while True:
            msg = json.loads((await self.ws.receive()).data)
            if msg.get("id") == self.seq:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]
            if msg.get("method") == "Runtime.exceptionThrown":
                print("page error:", msg["params"]["exceptionDetails"].get("exception", {}).get("description"), file=sys.stderr)
            if msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "warning"):
                print("console:", [a.get("value", a.get("description")) for a in msg["params"]["args"]], file=sys.stderr)

    async def eval(self, expr, timeout=120000):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True, timeout=timeout)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r))
        return r["result"].get("value")

    async def shot(self, fmt="jpeg") -> bytes:
        params = {"format": fmt, "clip": {"x": 0, "y": 0, "width": W, "height": H, "scale": 0.5}}
        if fmt == "jpeg":
            params["quality"] = 94
        r = await self.call("Page.captureScreenshot", **params)
        return base64.b64decode(r["data"])


async def with_page(fn, port: int):
    import aiohttp

    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--hide-scrollbars", f"--remote-debugging-port={port}",
                               f"--user-data-dir={profile}", f"--window-size={W},{H}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http_:
            for _ in range(200):
                try:
                    pages = await (await http_.get(f"http://127.0.0.1:{port}/json")).json()
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.1)
            target = next(p for p in pages if p["type"] == "page")
            async with http_.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                page = Page(ws)
                await page.call("Runtime.enable")
                await page.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
                await page.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/promo.html")
                for _ in range(600):
                    if await page.eval("window.promoReady === true"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("the page did not get ready")
                return await fn(page)
    finally:
        chrome.terminate()
        shutil.rmtree(profile, ignore_errors=True)


async def render_stills(times: list[float]) -> None:
    async def go(page):
        for t in times:
            await page.eval(f"renderFrame({t})")
            (OUT / f"still_{t:05.2f}.png").write_bytes(await page.shot("png"))
            print(f"still {t}")
    await with_page(go, CDP0)


async def render_frames(workers: int) -> dict:
    frames_dir = OUT / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    meta_box = {}
    t0 = time.time()
    done = [0]

    async def worker(k):
        async def go(page):
            meta = await page.eval("window.PROMO")
            meta_box["meta"] = meta
            n = int(round(meta["duration"] * FPS))
            for f in range(k, n, workers):
                await page.eval(f"renderFrame({f / FPS})")
                (frames_dir / f"{f:05d}.jpg").write_bytes(await page.shot())
                done[0] += 1
                if k == 0 and (f // workers) % 30 == 0:
                    el = time.time() - t0
                    print(f"{done[0]}/{n} frames, {el:.0f}s elapsed, ~{el / max(1, done[0]) * (n - done[0]):.0f}s left",
                          flush=True)
        await with_page(go, CDP0 + k)

    await asyncio.gather(*(worker(k) for k in range(workers)))
    print(f"rendered {done[0]} frames in {time.time() - t0:.0f}s with {workers} browsers")
    return meta_box["meta"]


def encode(frames_dir: Path, audio: Path, out: Path, gpu: int) -> None:
    has_nvenc = "h264_nvenc" in subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    common = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.jpg"), "-i", str(audio),
              "-map", "0:v", "-map", "1:a"]
    if has_nvenc:
        video = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "11M",
                 "-maxrate", "15M", "-bufsize", "30M", "-profile:v", "high", "-bf", "3", "-spatial-aq", "1",
                 "-temporal-aq", "1"]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    else:
        video = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high"]
        env = None
    t0 = time.time()
    subprocess.run(common + video + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                                     "-movflags", "+faststart", str(out)], check=True, env=env)
    print(f"encoded with {'h264_nvenc' if has_nvenc else 'libx264'} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------------------------- #
# Sound effects (no music)
# --------------------------------------------------------------------------------------------- #
def highpass(x, cutoff, order=2):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(order, cutoff, "high", fs=SR, output="sos"), x)


def lowpass(x, cutoff, order=2):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(order, cutoff, "low", fs=SR, output="sos"), x)


def add(buf, start, sig, gain=1.0, pan=0.0):
    i = int(round(start * SR))
    if i >= buf.shape[1] or i + len(sig) <= 0:
        return
    if i < 0:
        sig, i = sig[-i:], 0
    j = min(buf.shape[1], i + len(sig))
    s = sig[: j - i] * gain
    buf[0, i:j] += s * np.sqrt(0.5 * (1 - pan))
    buf[1, i:j] += s * np.sqrt(0.5 * (1 + pan))


def sound_effects(events: list[dict], duration: float) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros((2, n))
    rng = np.random.default_rng(9)
    for e in events:
        t, kind = e["t"], e["kind"]
        if kind == "key":
            # a soft key tap, panned a little at random
            m = int(0.035 * SR)
            tt = np.arange(m) / SR
            sig = highpass(rng.standard_normal(m), 1800) * np.exp(-tt / 0.004) + np.sin(2 * np.pi * 620 * tt) * np.exp(-tt / 0.006) * 0.5
            add(out, t, sig, 0.10, rng.uniform(-0.3, 0.3))
        elif kind in ("blip", "blip2"):
            m = int(0.09 * SR)
            tt = np.arange(m) / SR
            f = 1760 if kind == "blip" else 2349
            add(out, t, np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.025), 0.12)
        elif kind == "success":
            for k, f in enumerate((1318.5, 1760.0)):
                m = int(0.7 * SR)
                tt = np.arange(m) / SR
                sig = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 0.22)
                add(out, t + k * 0.09, sig, 0.13)
        elif kind == "whoosh":
            m = int(0.55 * SR)
            tt = np.arange(m) / SR
            bell = np.sin(np.pi * tt / (m / SR)) ** 2
            add(out, t - 0.2, lowpass(rng.standard_normal(m), 2500) * bell, 0.10)
        elif kind == "swipe":
            # the before/after wipe: a short brushed sweep, left to right
            m = int(0.5 * SR)
            tt = np.arange(m) / SR
            bell = np.sin(np.pi * tt / (m / SR)) ** 2
            noise = lowpass(highpass(rng.standard_normal(m), 2500), 7000)
            sweep = np.linspace(-0.6, 0.6, m)
            for ch, sgn in ((0, -1), (1, 1)):
                out[ch, int(t * SR):int(t * SR) + m] += (noise * bell * 0.05 * np.sqrt(0.5 * (1 + sgn * sweep)))[: max(0, min(m, n - int(t * SR)))]
    return out


def write_audio(meta: dict, path: Path, peak_db: float) -> None:
    import wave

    mixdown = sound_effects(meta["sfx"], meta["duration"])
    # the effects sit far in the background: peak at peak_db dBFS, no loudness push
    mixdown *= 10 ** (peak_db / 20) / max(1e-6, np.abs(mixdown).max())
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mixdown.T * 32767).astype("<i2").tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stills", help="comma-separated times to render as PNG instead of the video")
    p.add_argument("--workers", type=int, default=8, help="Chrome processes rendering frames in parallel")
    p.add_argument("--gpu", type=int, default=0, help="NVENC device (nvidia-smi index)")
    p.add_argument("--sfx-db", type=float, default=-30.0, help="peak level of the sound effects, dBFS (barely audible)")
    p.add_argument("--remix", action="store_true", help="only rebuild the sound and put it on the rendered video")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.remix:
        write_audio(json.loads((OUT / "meta.json").read_text()), OUT / "audio.wav", a.sfx_db)
        tmp = OUT / f"{NAME}.tmp.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT / f"{NAME}-silent.mp4"), "-i", str(OUT / "audio.wav"),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(OUT / f"{NAME}.mp4")
        print(f"remixed {OUT / NAME}.mp4 with the effects peaking at {a.sfx_db} dBFS")
        return
    build()
    server = serve()
    try:
        if a.stills:
            asyncio.run(render_stills([float(x) for x in a.stills.split(",")]))
            return
        meta = asyncio.run(render_frames(a.workers))
        (OUT / "meta.json").write_text(json.dumps(meta))
        write_audio(meta, OUT / "audio.wav", a.sfx_db)
        encode(OUT / "frames", OUT / "audio.wav", OUT / f"{NAME}.mp4", a.gpu)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT / f"{NAME}.mp4"), "-c", "copy", "-an",
                        "-movflags", "+faststart", str(OUT / f"{NAME}-silent.mp4")], check=True)
        print(f"wrote {OUT / NAME}.mp4 and -silent.mp4")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
