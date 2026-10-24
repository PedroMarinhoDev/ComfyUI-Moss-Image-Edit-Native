// The promo video, as a page rendered one frame at a time: renderFrame(t) puts every element where it is at t
// seconds. Nothing depends on earlier frames, so any frame can be rendered on its own (the renderer splits the
// frames between several browsers). Pictures are build/img/<key>.jpg, listed with their sizes in data.json.

const $ = (id) => document.getElementById(id);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (t, a, b) => clamp((t - a) / (b - a));
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const easeOut = (x) => 1 - Math.pow(1 - x, 3);
const lerp = (a, b, k) => a + (b - a) * k;
const XF = 0.4;
const vis = (t, a, b, first = false, last = false) =>
  Math.min(first ? 1 : seg(t, a - XF / 2, a + XF / 2), last ? 1 : 1 - seg(t, b - XF / 2, b + XF / 2));
// moves land on whole device pixels (the page renders at 2x), so text is never drawn between pixels differently
// from frame to frame
const px = (v) => Math.round(v * 2) / 2;
const fadeUp = (el, k, dy = 24) => { el.style.opacity = k; el.style.transform = `translateY(${px((1 - k) * dy)}px)`; };
const raf = () => new Promise((r) => requestAnimationFrame(() => r()));

const data = await (await fetch("data.json")).json();
const IMG = data.img;
const src = (key) => `img/${key}.jpg`;
const aspect = (key) => IMG[key].w / IMG[key].h;
const decoding = [];
function img(key, cls = "") {
  const el = document.createElement("img");
  el.src = src(key);
  if (cls) el.className = cls;
  decoding.push(el.decode().catch(() => {}));
  return el;
}
// a frame holding a before and an after picture; wipe(k) reveals the after one from the left
function pairFrame(parent, before, after, box, labels = ["BEFORE", "AFTER"]) {
  const f = document.createElement("div");
  f.className = "frame";
  Object.assign(f.style, { left: `${box.x}px`, top: `${box.y}px`, width: `${box.w}px`, height: `${box.h}px` });
  if (box.pos) f.style.setProperty("--pos", box.pos);
  const a = img(before), b = img(after, "after");
  for (const im of [a, b]) if (box.pos) im.style.objectPosition = box.pos;
  const line = document.createElement("div");
  line.className = "wipe";
  f.append(a, b, line);
  let lb = null, la = null;
  if (labels) {
    lb = document.createElement("div"); lb.className = "badge tl"; lb.innerHTML = labels[0];
    la = document.createElement("div"); la.className = "badge tr"; la.innerHTML = `<b>${labels[1]}</b>`;
    f.append(lb, la);
  }
  parent.append(f);
  const wipe = (k) => {
    b.style.clipPath = `inset(0 ${px((1 - k) * box.w)}px 0 0)`;
    line.style.left = `${px(k * box.w)}px`;
    line.style.opacity = k > 0 && k < 1 ? 1 : 0;
    if (lb) { lb.style.opacity = 1 - seg(k, 0.0, 0.25); la.style.opacity = seg(k, 0.75, 1); }
  };
  wipe(0);
  return { el: f, wipe };
}
function single(parent, key, box, label) {
  const f = document.createElement("div");
  f.className = "frame";
  Object.assign(f.style, { left: `${box.x}px`, top: `${box.y}px`, width: `${box.w}px`, height: `${box.h}px` });
  f.append(img(key));
  if (label) { const l = document.createElement("div"); l.className = "badge tl"; l.innerHTML = label; f.append(l); }
  parent.append(f);
  return f;
}
// fit a picture of the given aspect inside a w x h box, centred
function fit(key, x, y, w, h) {
  const r = aspect(key);
  const fw = Math.min(w, h * r), fh = fw / r;
  return { x: Math.round(x + (w - fw) / 2), y: Math.round(y + (h - fh) / 2), w: Math.round(fw), h: Math.round(fh) };
}
const typed = (text, k) => {
  const n = Math.floor(text.length * k + 1e-6);
  return `${text.slice(0, n)}<span class="caret"></span>`;
};

// ---- timeline -------------------------------------------------------------------------------------------
const S = { title: [0, 4], type: [4, 15], grid: [15, 26], multi: [26, 33.5], lora: [33.5, 41], tested: [41, 50],
            native: [50, 56], outro: [56, 62] };
const DURATION = 62;
const sfx = [];
for (const k of Object.keys(S).slice(1)) sfx.push({ t: S[k][0] - 0.15, kind: "whoosh" });
const typingSfx = (t0, t1, text) => {
  const n = text.length;
  for (let i = 0; i < n; i += 2) if (text[i] !== " ") sfx.push({ t: t0 + (t1 - t0) * (i / n), kind: "key" });
};

// ---- mosaics --------------------------------------------------------------------------------------------
const MOS = data.mosaic;
function buildMosaic(el, offset) {
  const rows = [];
  for (let r = 0; r < 4; r++) {
    const row = document.createElement("div");
    row.className = "mrow";
    for (let c = 0; c < 7; c++) row.append(img(MOS[(offset + r * 7 + c * 5) % MOS.length]));
    el.append(row);
    rows.push({ row, dir: r % 2 ? 1 : -1 });
  }
  return rows;
}
const mosaics = [buildMosaic($("mosaic1"), 0), buildMosaic($("mosaic2"), 11)];
function updateMosaic(rows, t) {
  for (const { row, dir } of rows) row.style.transform = `translateX(${px(-420 + dir * (t * 26 - 60))}px)`;
}

// ---- 01 type the edit ------------------------------------------------------------------------------------
const EDITS = [
  { before: "woman_src", after: "glasses_out", lora: "chinoll", text: "Add black sunglasses to the woman.", pos: "50% 30%" },
  { before: "bike_src", after: "bike_out", lora: "chinoll",
    text: "Paint the bicycle bright red, keeping the ivy, the brick wall and the lighting the same." },
];
const EDIT_LEN = (S.type[1] - S.type[0]) / EDITS.length;
// each picture and the prompt card are centred together; the card slides when the next picture has another shape
const PROMPT_W = 564, PROMPT_GAP = 56;
const editBoxes = EDITS.map((e) => {
  const b = fit(e.before, 0, 236, 1110, 760);
  b.x = Math.round((1920 - (b.w + PROMPT_GAP + PROMPT_W)) / 2);
  return b;
});
const editFrames = EDITS.map((e, i) => pairFrame($("e-frames"), e.before, e.after, editBoxes[i]));
const promptX = (t) => {
  const u = t - S.type[0];
  let x = editBoxes[0].x + editBoxes[0].w + PROMPT_GAP;
  for (let j = 1; j < EDITS.length; j++) {
    const next = editBoxes[j].x + editBoxes[j].w + PROMPT_GAP;
    x = lerp(x, next, ease(seg(u, j * EDIT_LEN - 0.35, j * EDIT_LEN + 0.35)));
  }
  return px(x);
};
const T_TYPE = [0.4, 2.0], T_WIPE = [2.4, 3.5];
EDITS.forEach((e, i) => {
  const u0 = S.type[0] + i * EDIT_LEN;
  typingSfx(u0 + T_TYPE[0], u0 + T_TYPE[1], e.text);
  sfx.push({ t: u0 + T_WIPE[0], kind: "swipe" }, { t: u0 + T_WIPE[1] - 0.1, kind: "blip" });
});
function sceneType(t) {
  const [a] = S.type;
  fadeUp($("s1"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const i = clamp(Math.floor((t - a) / EDIT_LEN), 0, EDITS.length - 1);
  EDITS.forEach((e, j) => {
    const u = t - a - j * EDIT_LEN;
    const inK = j === 0 ? easeOut(seg(u, 0.1, 0.6)) : seg(u, -XF / 2, XF / 2);
    const outK = j === EDITS.length - 1 ? 1 : 1 - seg(u, EDIT_LEN - XF / 2, EDIT_LEN + XF / 2);
    const v = Math.min(inK, outK);
    const f = editFrames[j];
    f.el.style.display = v > 0 ? "" : "none";
    f.el.style.opacity = v;
    f.el.style.transform = `scale(${lerp(0.96, 1, easeOut(seg(u, -0.2, 0.5)))})`;
    f.wipe(ease(seg(u, T_WIPE[0], T_WIPE[1])));
  });
  const u = t - a - i * EDIT_LEN;
  const e = EDITS[i];
  fadeUp($("e-prompt"), easeOut(seg(t, a + 0.2, a + 0.7)), 20);
  const blink = Math.floor((t - a) * 2.4) % 2 === 0 || (u > T_TYPE[0] && u < T_TYPE[1]);
  $("e-text").innerHTML = typed(e.text, seg(u, T_TYPE[0], T_TYPE[1]));
  $("e-text").querySelector(".caret").style.opacity = blink ? 1 : 0;
  fadeUp($("e-chip1"), easeOut(seg(t, a + 0.6, a + 1.1)), 16);
  fadeUp($("e-chip2"), easeOut(seg(t, a + 0.8, a + 1.3)), 16);
  fadeUp($("e-lora"), easeOut(seg(t, a + 1.0, a + 1.5)), 16);
  for (const id of ["e-prompt", "e-chip1", "e-chip2", "e-lora"]) $(id).style.left = `${promptX(t)}px`;
  $("e-lora").innerHTML = `LoRA <i>${e.lora}</i> · seed 42 · first try`;
}

// ---- 02 grid of edit types ---------------------------------------------------------------------------------
const TILES = data.tiles;          // [{task, name, lora, text, pos}]
const GRID = { x: 96, y: 236, cols: 5, w: 332, h: 252, gx: 17, gy: 84 };
const tileEls = TILES.map((tile, i) => {
  const d = document.createElement("div");
  d.className = "tile";
  const col = i % GRID.cols, row = Math.floor(i / GRID.cols);
  Object.assign(d.style, { left: `${GRID.x + col * (GRID.w + GRID.gx)}px`, top: `${GRID.y + row * (GRID.h + GRID.gy)}px`,
                           width: `${GRID.w}px` });
  const holder = document.createElement("div");
  const pf = pairFrame(holder, `${tile.task}_src`, `${tile.task}_out`, { x: 0, y: 0, w: GRID.w, h: GRID.h, pos: tile.pos }, null);
  pf.el.style.position = "relative";
  d.append(pf.el);
  const name = document.createElement("div");
  name.className = "name";
  name.innerHTML = `<span>${tile.name}</span><em>${tile.lora}</em>`;
  d.append(name);
  $("g-tiles").append(d);
  return { d, wipe: pf.wipe };
});
const G_START = 1.3, G_STEP = 0.82, G_WIPE = 0.6;
TILES.forEach((_, i) => sfx.push({ t: S.grid[0] + G_START + i * G_STEP, kind: "swipe" }));
function sceneGrid(t) {
  const [a] = S.grid;
  fadeUp($("s2"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const cur = Math.floor((t - a - G_START) / G_STEP);
  tileEls.forEach(({ d, wipe }, i) => {
    fadeUp(d, easeOut(seg(t, a + 0.25 + i * 0.07, a + 0.75 + i * 0.07)), 26);
    const w0 = a + G_START + i * G_STEP;
    wipe(ease(seg(t, w0, w0 + G_WIPE)));
    d.classList.toggle("on", i === cur);
  });
  const k = clamp(cur, 0, TILES.length - 1);
  const tile = TILES[k];
  $("g-ticker").innerHTML = `<small>${tile.name.toUpperCase()}</small>"${tile.text}"`;
  $("g-ticker").style.opacity = seg(t, a + G_START, a + G_START + 0.3);
}

// ---- 03 two images ---------------------------------------------------------------------------------------------
const MULTI_TEXT = "Dress the woman in image 1 in the black leather jacket from image 2.";
const MH = 520, MY = 250;
const mBox = (key, x) => ({ x, y: MY, w: Math.round(MH * aspect(key)), h: MH });
const m1 = single($("m-frames"), "woman_src", mBox("woman_src", 96), "IMAGE 1");
const m2 = single($("m-frames"), "jacket_src", { x: 520, y: MY, w: MH * 0.78 | 0, h: MH }, "IMAGE 2");
const mOut = [single($("m-frames"), "multi_c", mBox("multi_c", 1040), "<b>chinoll</b>"),
              single($("m-frames"), "multi_h", mBox("multi_h", 1040 + Math.round(MH * aspect("multi_c")) + 20), "<b>HQ</b>")];
const mPlus = document.createElement("div"); mPlus.className = "plus"; mPlus.textContent = "+";
Object.assign(mPlus.style, { left: "466px", top: `${MY + MH / 2 - 50}px` });
const mArrow = document.createElement("div"); mArrow.className = "arrow";
Object.assign(mArrow.style, { left: "944px", top: `${MY + MH / 2}px`, width: "66px" });
$("m-frames").append(mPlus, mArrow);
typingSfx(S.multi[0] + 1.0, S.multi[0] + 2.6, MULTI_TEXT);
sfx.push({ t: S.multi[0] + 3.0, kind: "blip" }, { t: S.multi[0] + 3.35, kind: "blip2" });
function sceneMulti(t) {
  const [a] = S.multi;
  fadeUp($("s3"), easeOut(seg(t, a + 0.05, a + 0.7)));
  fadeUp(m1, easeOut(seg(t, a + 0.15, a + 0.65)), 30);
  fadeUp(mPlus, easeOut(seg(t, a + 0.35, a + 0.8)), 10);
  fadeUp(m2, easeOut(seg(t, a + 0.45, a + 0.95)), 30);
  $("m-ticker").innerHTML = `<small>PROMPT</small>${typed(MULTI_TEXT, seg(t, a + 1.0, a + 2.6))}`;
  $("m-ticker").style.opacity = seg(t, a + 0.8, a + 1.1);
  $("m-ticker").querySelector(".caret").style.opacity = Math.floor((t - a) * 2.4) % 2 === 0 || t < a + 2.6 ? 1 : 0;
  $("m-ticker").style.overflow = "visible";
  $("m-ticker").style.whiteSpace = "nowrap";
  mArrow.style.transform = `scaleX(${easeOut(seg(t, a + 2.7, a + 3.1))})`;
  mArrow.style.opacity = seg(t, a + 2.7, a + 2.8);
  mOut.forEach((el, i) => {
    const k = easeOut(seg(t, a + 3.0 + i * 0.35, a + 3.6 + i * 0.35));
    el.style.opacity = k;
    el.style.transform = `scale(${lerp(0.92, 1, k)})`;
  });
}

// ---- 04 two LoRAs ---------------------------------------------------------------------------------------------
const LORA_TEXT = "Replace the background with a snowy mountain landscape, keeping the woman exactly the same.";
const LH = 620, LY = 236;
const lw = Math.round(LH * aspect("woman_src"));
const lSrc = single($("l-frames"), "woman_src", { x: 96, y: LY, w: lw, h: LH }, "SOURCE");
const lPairs = [
  pairFrame($("l-frames"), "woman_src", "bg_c", { x: 96 + lw + 24, y: LY, w: lw, h: LH }, ["BEFORE", "chinoll"]),
  pairFrame($("l-frames"), "woman_src", "bg_h", { x: 96 + 2 * (lw + 24), y: LY, w: lw, h: LH }, ["BEFORE", "HQ"]),
];
sfx.push({ t: S.lora[0] + 1.2, kind: "swipe" }, { t: S.lora[0] + 2.4, kind: "swipe" });
function sceneLora(t) {
  const [a] = S.lora;
  fadeUp($("s4"), easeOut(seg(t, a + 0.05, a + 0.7)));
  fadeUp(lSrc, easeOut(seg(t, a + 0.15, a + 0.65)), 30);
  lPairs.forEach((p, i) => {
    fadeUp(p.el, easeOut(seg(t, a + 0.3 + i * 0.15, a + 0.8 + i * 0.15)), 30);
    p.wipe(ease(seg(t, a + 1.2 + i * 1.2, a + 2.0 + i * 1.2)));
  });
  fadeUp($("l-c"), easeOut(seg(t, a + 0.9, a + 1.4)), 20);
  fadeUp($("l-h"), easeOut(seg(t, a + 2.1, a + 2.6)), 20);
  $("l-c").classList.toggle("on", t >= a + 1.2 && t < a + 2.4);
  $("l-h").classList.toggle("on", t >= a + 2.4);
  $("l-ticker").innerHTML = `<small>PROMPT</small>"${LORA_TEXT}"`;
  $("l-ticker").style.opacity = seg(t, a + 0.7, a + 1.1);
  $("l-ticker").style.whiteSpace = "normal";
  $("l-ticker").style.lineHeight = "1.4";
}

// ---- 05 tested defaults -----------------------------------------------------------------------------------------
const FINDINGS = [
  { title: "LoRA strength <b>1.0</b>", cells: [["str_025", "0.25"], ["str_050", "0.50"], ["str_075", "0.75"], ["str_100", "1.0", true]],
    d: "At lower strengths the scene is <b>redrawn</b>, and two-image edits give back image 2." },
  { title: "Images at <b>output size</b>", stack: true, cells: [["size_small", "512 → 1024"], ["size_match", "1024 → 1024", true]],
    d: "Each image goes in as a <b>1 MP bucket</b> of its own aspect: no squeezing, no upscaling blur." },
  { title: "<b>SFT</b>, not Base", cells: [["base_base", "BASE"], ["base_sft", "SFT", true]],
    d: "Same seed, same settings: on Base the HQ LoRA <b>breaks the face</b>; on SFT it holds." },
];
const FW = 560, FG = 24;
const fEls = FINDINGS.map((f, i) => {
  const d = document.createElement("div");
  d.className = "fcard";
  Object.assign(d.style, { left: `${96 + i * (FW + FG)}px`, top: "232px", width: `${FW}px`, height: "714px" });
  d.innerHTML = `<div class="t">${f.title}</div><div class="row${f.stack ? " stack" : ""}">${f.cells.map(([k, l, good]) =>
    `<div class="cell${good ? " good-later" : ""}"><img src="${src(k)}"><span>${l}</span></div>`).join("")}</div><div class="d">${f.d}</div>`;
  for (const im of d.querySelectorAll("img")) decoding.push(im.decode().catch(() => {}));
  $("f-cards").append(d);
  return d;
});
const SETTINGS = ["<b>CFG</b> 5 chinoll · 3 HQ", "<b>Skip Layer Guidance</b> on layer 22", "<b>30</b> steps · euler · simple",
                  "<b>KV cache</b> auto"];
const sEls = SETTINGS.map((x) => { const c = document.createElement("div"); c.className = "chip"; c.innerHTML = x; $("f-chips").append(c); return c; });
FINDINGS.forEach((_, i) => sfx.push({ t: S.tested[0] + 1.6 + i * 0.9, kind: "success" }));
SETTINGS.forEach((_, i) => sfx.push({ t: S.tested[0] + 4.6 + i * 0.2, kind: "blip2" }));
function sceneTested(t) {
  const [a] = S.tested;
  fadeUp($("s5"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const c = ease(seg(t, a + 0.3, a + 1.5));
  $("f-count").innerHTML = `${Math.round(300 * c)}+<small>TEST EDITS</small>`;
  $("f-count").style.opacity = seg(t, a + 0.2, a + 0.5);
  fEls.forEach((d, i) => {
    fadeUp(d, easeOut(seg(t, a + 0.3 + i * 0.25, a + 0.9 + i * 0.25)), 30);
    const on = t >= a + 1.6 + i * 0.9;
    for (const cell of d.querySelectorAll(".good-later")) cell.classList.toggle("good", on);
  });
  sEls.forEach((e, i) => fadeUp(e, easeOut(seg(t, a + 4.6 + i * 0.2, a + 5.1 + i * 0.2)), 14));
}

// ---- native nodes -------------------------------------------------------------------------------------------------
$("n-img").src = src("diagram");
decoding.push($("n-img").decode().catch(() => {}));
const NATIVE = ["<b>1–4</b> images", "references <b>cached</b> once per run", "ready-made <b>workflows</b>", "no extra dependencies"];
const nEls = NATIVE.map((x) => { const c = document.createElement("div"); c.className = "chip"; c.innerHTML = x; $("n-chips").append(c); return c; });
NATIVE.forEach((_, i) => sfx.push({ t: S.native[0] + 1.4 + i * 0.25, kind: "blip2" }));
function sceneNative(t) {
  const [a] = S.native;
  fadeUp($("s6"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const s = easeOut(seg(t, a + 0.1, a + 0.8));
  $("n-diagram").style.opacity = s;
  $("n-diagram").style.transform = `scale(${lerp(0.96, 1, s)})`;
  nEls.forEach((e, i) => fadeUp(e, easeOut(seg(t, a + 1.4 + i * 0.25, a + 1.9 + i * 0.25)), 14));
}

// ---- title and outro ---------------------------------------------------------------------------------------------
function sceneTitle(t) {
  updateMosaic(mosaics[0], t);
  fadeUp($("t-kicker"), easeOut(seg(t, 0.2, 0.8)), 16);
  const k = easeOut(seg(t, 0.4, 1.2));
  $("t-big").style.opacity = k;
  $("t-big").style.transform = `scale(${lerp(1.12, 1, k)})`;
  fadeUp($("t-tag"), easeOut(seg(t, 1.1, 1.7)), 16);
  fadeUp($("t-pill"), easeOut(seg(t, 1.7, 2.2)), 10);
}
function sceneOutro(t) {
  updateMosaic(mosaics[1], t - S.outro[0]);
}
sfx.push({ t: 0.4, kind: "whoosh" }, { t: 1.7, kind: "blip" });

// ---- frames -------------------------------------------------------------------------------------------------
const LAYERS = [["L-title", S.title, sceneTitle], ["L-type", S.type, sceneType], ["L-grid", S.grid, sceneGrid],
                ["L-multi", S.multi, sceneMulti], ["L-lora", S.lora, sceneLora], ["L-tested", S.tested, sceneTested],
                ["L-native", S.native, sceneNative], ["L-outro", S.outro, sceneOutro]];
window.renderFrame = async (t) => {
  LAYERS.forEach(([id, [a, b], fn], i) => {
    const v = vis(t, a, b, i === 0, i === LAYERS.length - 1);
    $(id).style.opacity = v;
    $(id).style.display = v > 0 ? "" : "none";
    if (v > 0) fn(t);
  });
  // fade in from and out to black
  document.body.style.filter = `brightness(${Math.min(seg(t, 0, 0.5), 1 - seg(t, DURATION - 0.9, DURATION))})`;
  await raf();
  await raf();
};
window.PROMO = { duration: DURATION, sfx: sfx.sort((a, b) => a.t - b.t), sections: S };
await document.fonts.ready;
await Promise.all(decoding);
await renderFrame(0);
window.promoReady = true;
