"""
player.py -- plays a finished run in the browser, frame by frame, with no flicker.

There is no model here and no Matplotlib. `player_html` takes a finished `Sim`
(built with keep_grids=True) and returns one self-contained HTML page: the whole
run, compressed, plus a small JavaScript player that draws it on a <canvas>.

Why not just redraw a Matplotlib figure every generation? Because each redraw is
a new PNG, and swapping one image for the next makes the page blink between
generations. Here the browser keeps one canvas and repaints it in place, so
playback is smooth, and it needs no Python at all once the page has loaded --
which also makes the page a file you can save and open offline.

Colours come from render.PALETTE, so a strategy has the same colour here as in
every Matplotlib figure.
"""
import base64
import html
import json
import zlib

import numpy as np

from render import PALETTE


def _payload(sim, names=None, stride=1):
    """The run as plain data: compressed lattices plus the per-generation numbers.

    `stride` keeps every stride-th generation (and always the last), for runs
    too big to ship whole; the numbers are kept at the same generations, and
    "gens" says which generation each frame is.
    """
    m = sim.model
    if not sim.grids:
        raise ValueError("no stored lattices -- build the Sim with keep_grids=True")
    if m.k > len(PALETTE):
        raise ValueError(f"the player has colours for {len(PALETTE)} strategies, "
                         f"not {m.k}")
    names = list(names) if names is not None else [f"S{i + 1}" for i in range(m.k)]
    if len(names) != m.k:
        raise ValueError(f"{len(names)} names for {m.k} strategies")
    n = len(sim.grids)
    keep = list(range(0, n, max(1, int(stride))))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    grids = np.stack([sim.grids[g] for g in keep]).astype(np.uint8)
    # "frozen" compares with the generation before, which a stride may have skipped
    frozen = [False] + [bool(np.array_equal(sim.grids[g], sim.grids[g - 1]))
                        for g in keep[1:]]
    r = sim.series("assortment")[keep]
    return {
        "rows": m.rows, "cols": m.cols, "k": m.k, "is1d": bool(m.is_1d),
        "last": len(keep) - 1, "gens": keep, "stride": max(1, int(stride)),
        "names": [str(x) for x in names],
        "colors": PALETTE[:m.k],
        # deflate (zlib) + base64: lattices compress very well, and the browser
        # inflates them with the built-in DecompressionStream
        "grids": base64.b64encode(zlib.compress(grids.tobytes(), 6)).decode("ascii"),
        "freq": np.round(sim.series("frequencies")[keep], 5).tolist(),
        "r": [None if np.isnan(x) else round(float(x), 5) for x in r],
        "payoff": [round(float(x), 5) for x in sim.series("mean_payoff")[keep]],
        "frozen": frozen,
    }


def player_html(sim, names=None, max_side=760, stride=1):
    """A standalone HTML page that plays `sim`.

    `max_side` is the largest the lattice is drawn, in CSS pixels; it shrinks to
    fit narrower windows. `stride` > 1 keeps only every stride-th generation.
    """
    data = json.dumps(_payload(sim, names, stride), separators=(",", ":"))
    # Strategy names can arrive in a shared link, and the data sits inside a
    # <script> element. Escaping every <, > and & keeps any name (say one
    # containing "</script>" or "<!--<script>") from ending or reshaping that
    # element; JSON.parse reads the \u escapes back as the same characters.
    data = (data.replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))
    return (_TEMPLATE.replace("__MAX_SIDE__", str(int(max_side)))
            .replace("__DATA__", data))


def fit_player(sim, names=None, max_bytes=7_000_000, max_side=760):
    """(page, iframe tag, stride) with the iframe tag under `max_bytes`.

    marimo replaces any cell output over 8 MB with a "your output is too large"
    notice, and a long run on a big lattice with a lot of mutation (which
    compresses badly) can exceed that. Rather than fail, keep every s-th
    generation, raising s until the page fits. Ordinary runs fit at s = 1.
    """
    stride = 1
    while True:
        page = player_html(sim, names, max_side, stride)
        tag = iframe_html(page)
        if len(tag) <= max_bytes or stride >= len(sim.grids):
            return page, tag, stride
        stride = max(stride + 1, int(np.ceil(stride * len(tag) / max_bytes)))


def iframe_html(page, height=900):
    """An <iframe> tag that shows `page`, for embedding in a notebook or app.

    Built here rather than with marimo's mo.iframe, which strips every newline
    from the tag it builds -- and with the newlines gone, the first `//` comment
    in the player's script comments out the rest of it, so the page loads blank.
    Encoding the newlines as &#10; keeps them through any such flattening. The
    player grows the frame to fit its content once it has laid itself out.
    """
    srcdoc = html.escape(page, quote=True).replace("\n", "&#10;")
    return (f'<iframe srcdoc="{srcdoc}" width="100%" height="{int(height)}px" '
            'frameborder="0" style="border:0;display:block"></iframe>')


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spatial run</title>
<style>
  :root { --ink:#222; --muted:#666; --line:#d9d9d9; --panel:#f6f6f6; }
  html, body { margin:0; background:#fff; color:var(--ink);
    font:13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:2px 0 10px; }
  button, select { font:inherit; padding:4px 10px; border:1px solid #bbb; border-radius:6px;
    background:#f7f7f7; color:var(--ink); cursor:pointer; }
  button:hover { background:#eee; }
  button#play { min-width:76px; font-weight:600; }
  input[type=range] { flex:1 1 160px; min-width:120px; }
  #gen { font-variant-numeric:tabular-nums; min-width:150px; color:var(--muted); }
  .main { display:flex; flex-wrap:wrap; gap:20px; align-items:flex-start; }
  canvas { display:block; }
  #lat { border:1px solid var(--line); background:#fff; }
  .side { display:flex; flex-direction:column; gap:10px; }
  .legend { display:grid; gap:2px 14px; }
  .item { display:flex; align-items:center; gap:6px; white-space:nowrap;
    font-variant-numeric:tabular-nums; }
  .sw { width:12px; height:12px; border-radius:2px; flex:0 0 auto; }
  .pct { margin-left:auto; color:var(--muted); }
  table.nums { border-collapse:collapse; }
  table.nums td { padding:1px 10px 1px 0; }
  table.nums td:first-child { color:var(--muted); }
  #msg { color:#a33; }
  #note { color:var(--muted); font-size:12px; }
</style></head>
<body>
<div class="bar">
  <button id="restart" title="Back to generation 0 (Home)">&#x23EE; Restart</button>
  <button id="back" title="One generation back (left arrow)">&#x25C0; Back</button>
  <button id="play" title="Play or pause (space)">&#x25B6; Play</button>
  <button id="step" title="One generation on (right arrow)">Step &#x25B6;</button>
  <button id="end" title="Last generation (End)">To end &#x23ED;</button>
  <select id="speed" title="Playback speed">
    <option value="2">2 / s</option><option value="5">5 / s</option>
    <option value="10" selected>10 / s</option><option value="20">20 / s</option>
    <option value="60">60 / s</option>
  </select>
  <input id="slider" type="range" min="0" value="0" step="1" aria-label="generation">
  <span id="gen"></span>
  <button id="save" title="Save the lattice as it stands now">Save picture (PNG)</button>
</div>
<div class="main">
  <canvas id="lat"></canvas>
  <div class="side" id="side">
    <div class="legend" id="legend"></div>
    <table class="nums">
      <tr><td>assortment r</td><td id="rval"></td></tr>
      <tr><td>mean payoff</td><td id="pval"></td></tr>
      <tr><td>frozen?</td><td id="fval"></td></tr>
    </table>
    <canvas id="freq"></canvas>
    <canvas id="assort"></canvas>
  </div>
</div>
<div id="note"></div>
<div id="msg"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("data").textContent);
  const MAX_SIDE = __MAX_SIDE__;
  const R = D.rows, C = D.cols, K = D.k, LAST = D.last, CELLS = R * C;
  const GENS = D.gens, GLAST = GENS[LAST];       // frame index -> generation
  const $ = (id) => document.getElementById(id);
  const rgb = D.colors.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));

  let grids = null, frame = 0, playing = false, lastTick = 0, side = 400;

  // ---- the lattice -------------------------------------------------------
  // One pixel per site on an offscreen canvas, scaled up without smoothing, so
  // a site is a crisp square at any size. For a 1D lattice the offscreen canvas
  // is the whole space-time diagram (one row per generation).
  const off = document.createElement("canvas");
  off.width = C; off.height = D.is1d ? LAST + 1 : R;
  const octx = off.getContext("2d");
  const lat = $("lat"), lctx = lat.getContext("2d");

  function paintRows(img, src, nrows) {
    const d = img.data;
    for (let i = 0; i < nrows * C; i++) {
      const c = rgb[src[i]];
      d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = 255;
    }
  }
  let frameImg = null;
  function prepareLattice() {
    if (D.is1d) {               // the whole space-time diagram, once
      const img = octx.createImageData(C, LAST + 1);
      paintRows(img, grids, LAST + 1);
      octx.putImageData(img, 0, 0);
    } else {
      frameImg = octx.createImageData(C, R);
    }
  }
  function latticeBox() {        // drawn size in CSS pixels, keeping the aspect
    if (D.is1d) return [side, side];
    return C >= R ? [side, Math.max(1, Math.round(side * R / C))]
                  : [Math.max(1, Math.round(side * C / R)), side];
  }
  function drawLattice() {
    const dpr = window.devicePixelRatio || 1;
    const [w, h] = latticeBox();
    if (lat.width !== Math.round(w * dpr) || lat.height !== Math.round(h * dpr)) {
      lat.width = Math.round(w * dpr); lat.height = Math.round(h * dpr);
      lat.style.width = w + "px"; lat.style.height = h + "px";
    }
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.imageSmoothingEnabled = false;
    if (D.is1d) {
      lctx.fillStyle = "#f2f2f2";
      lctx.fillRect(0, 0, lat.width, lat.height);
      const hh = lat.height * (frame + 1) / (LAST + 1);
      lctx.drawImage(off, 0, 0, C, frame + 1, 0, 0, lat.width, hh);
    } else {
      paintRows(frameImg, grids.subarray(frame * CELLS, (frame + 1) * CELLS), R);
      octx.putImageData(frameImg, 0, 0);
      lctx.drawImage(off, 0, 0, C, R, 0, 0, lat.width, lat.height);
      if (Math.max(R, C) <= 40) {   // cell borders help only on small lattices
        lctx.strokeStyle = "rgba(255,255,255,0.9)";
        lctx.lineWidth = Math.max(1, 0.6 * dpr);
        lctx.beginPath();
        for (let c = 1; c < C; c++) { const x = Math.round(c * lat.width / C) + 0.5;
          lctx.moveTo(x, 0); lctx.lineTo(x, lat.height); }
        for (let r = 1; r < R; r++) { const y = Math.round(r * lat.height / R) + 0.5;
          lctx.moveTo(0, y); lctx.lineTo(lat.width, y); }
        lctx.stroke();
      }
    }
  }

  // ---- the two time series ----------------------------------------------
  // Each plot is drawn once to two offscreen layers (axes, data). A frame then
  // costs two image copies: the axes, and the data clipped at the current
  // generation, plus the cursor line.
  function makePlot(canvas, ylo, yhi, yticks, ylabel, lines, zeroLine) {
    const P = { canvas, ctx: canvas.getContext("2d"), ylo, yhi, yticks, ylabel,
                lines, zeroLine, base: document.createElement("canvas"),
                data: document.createElement("canvas") };
    return P;
  }
  const M = { l: 46, r: 10, t: 8, b: 30 };
  function niceStep(span) {
    const raw = span / 5, p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }
  function layoutPlot(P, w, h) {
    const dpr = window.devicePixelRatio || 1;
    for (const cv of [P.canvas, P.base, P.data]) { cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr); }
    P.canvas.style.width = w + "px"; P.canvas.style.height = h + "px";
    P.w = w; P.h = h; P.dpr = dpr;
    // x is the GENERATION, so a thinned run still has a true time axis
    const X = (g) => M.l + (w - M.l - M.r) * (GLAST > 0 ? g / GLAST : 0);
    const Y = (v) => M.t + (h - M.t - M.b) * (1 - (v - P.ylo) / (P.yhi - P.ylo));
    P.X = X;
    // axes layer
    const b = P.base.getContext("2d");
    b.setTransform(dpr, 0, 0, dpr, 0, 0);
    b.font = "11px -apple-system, Segoe UI, Helvetica, Arial, sans-serif";
    b.fillStyle = "#666"; b.strokeStyle = "#999"; b.lineWidth = 1;
    b.beginPath(); b.moveTo(M.l, M.t); b.lineTo(M.l, h - M.b); b.lineTo(w - M.r, h - M.b);
    b.stroke();
    b.textAlign = "right"; b.textBaseline = "middle";
    for (const v of P.yticks) {
      const y = Y(v);
      b.beginPath(); b.moveTo(M.l - 4, y); b.lineTo(M.l, y); b.stroke();
      b.fillText(String(v), M.l - 6, y);
    }
    if (P.zeroLine) {
      b.save(); b.setLineDash([4, 3]); b.strokeStyle = "#aaa";
      b.beginPath(); b.moveTo(M.l, Y(0)); b.lineTo(w - M.r, Y(0)); b.stroke(); b.restore();
    }
    b.textAlign = "center"; b.textBaseline = "top";
    const st = Math.max(1, niceStep(Math.max(GLAST, 1)));   // whole generations
    for (let g = 0; g <= GLAST; g += st) {
      const x = X(g);
      b.beginPath(); b.moveTo(x, h - M.b); b.lineTo(x, h - M.b + 4); b.stroke();
      b.fillText(String(g), x, h - M.b + 6);
    }
    b.fillText("generation", (M.l + w - M.r) / 2, h - 13);
    b.save(); b.translate(11, (M.t + h - M.b) / 2); b.rotate(-Math.PI / 2);
    b.textBaseline = "middle"; b.fillText(P.ylabel, 0, 0); b.restore();
    // data layer
    const d = P.data.getContext("2d");
    d.setTransform(dpr, 0, 0, dpr, 0, 0);
    d.lineWidth = 1.6; d.lineJoin = "round";
    for (const [color, ys] of P.lines) {
      d.strokeStyle = color; d.beginPath();
      let pen = false;
      for (let g = 0; g <= LAST; g++) {
        const v = ys[g];
        if (v === null) { pen = false; continue; }     // r undefined: a gap
        const x = X(GENS[g]), y = Y(v);
        if (pen) d.lineTo(x, y); else { d.moveTo(x, y); pen = true; }
      }
      if (LAST === 0 && ys[0] !== null) { d.lineTo(X(0) + 2, Y(ys[0])); }
      d.stroke();
    }
  }
  function drawPlot(P) {
    const c = P.ctx, dpr = P.dpr;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, P.canvas.width, P.canvas.height);
    c.drawImage(P.base, 0, 0);
    const xg = P.X(GENS[frame]);
    const xc = Math.ceil((xg + 1.5) * dpr);              // the run so far, no spoilers
    if (xc > 0) c.drawImage(P.data, 0, 0, xc, P.data.height, 0, 0, xc, P.data.height);
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.strokeStyle = "rgba(0,0,0,0.35)"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(xg, M.t); c.lineTo(xg, P.h - M.b); c.stroke();
  }
  const freqLines = D.colors.map((col, i) => [col, D.freq.map((row) => row[i])]);
  const FP = makePlot($("freq"), 0, 1, [0, 0.5, 1], "frequency", freqLines, false);
  const AP = makePlot($("assort"), -1, 1, [-1, 0, 1], "assortment r",
                      [["#333333", D.r]], true);

  // ---- the numbers --------------------------------------------------------
  const legend = $("legend"), pcts = [];
  legend.style.gridTemplateColumns = "repeat(" + (K > 8 ? 3 : K > 4 ? 2 : 1) +
    ", minmax(0, max-content))";
  D.names.forEach((name, i) => {
    const it = document.createElement("div"); it.className = "item";
    const sw = document.createElement("span"); sw.className = "sw";
    sw.style.background = D.colors[i];
    const nm = document.createElement("span"); nm.textContent = name;
    const pc = document.createElement("span"); pc.className = "pct";
    it.append(sw, nm, pc); legend.append(it); pcts.push(pc);
  });
  function drawNumbers() {
    $("gen").textContent = "generation " + GENS[frame] + " of " + GLAST;
    $("slider").value = frame;
    D.freq[frame].forEach((f, i) => { pcts[i].textContent = (100 * f).toFixed(1) + "%"; });
    const r = D.r[frame];
    $("rval").innerHTML = r === null ? "undefined (one strategy left)"
      : "<b>" + (r >= 0 ? "+" : "") + r.toFixed(3) + "</b>";
    $("pval").textContent = D.payoff[frame].toFixed(3);
    $("fval").textContent = GENS[frame] === 0 ? "--"
      : D.frozen[frame] ? "yes -- nothing changed last generation" : "no";
  }

  // ---- layout, playback, controls ----------------------------------------
  function layout() {
    const W = document.documentElement.clientWidth;
    const sideMin = 320, gap = 20;
    let sideW;
    if (W >= Math.min(MAX_SIDE, 560) + gap + sideMin) {   // lattice and panel side by side
      side = Math.max(200, Math.min(MAX_SIDE, W - gap - sideMin));
      sideW = Math.min(520, W - side - gap - 2);
    } else {                                               // panel below the lattice
      side = Math.max(160, Math.min(MAX_SIDE, W - 2));
      sideW = Math.min(520, W);
    }
    // a frame laid out while hidden reports width 0; a zero-size canvas makes
    // drawImage throw, so keep a floor and relayout when the width arrives
    sideW = Math.max(200, sideW);
    $("side").style.width = sideW + "px";
    layoutPlot(FP, sideW, 200);
    layoutPlot(AP, sideW, 160);
    render();
    try {                       // grow the surrounding iframe to fit, if there is one
      const fe = window.frameElement;
      // the body's own height, not the document's scroll height, which never
      // drops below the frame's current height and so could only ever grow
      if (fe) fe.style.height = Math.ceil(document.body.getBoundingClientRect().bottom + 6) + "px";
    } catch (e) { /* cross-origin: the host sizes the frame */ }
  }
  function render() {
    if (!grids) return;
    drawLattice(); drawPlot(FP); drawPlot(AP); drawNumbers();
  }
  function go(g) { frame = Math.max(0, Math.min(LAST, g)); render(); }
  function setPlaying(p) {
    playing = p && LAST > 0;
    $("play").innerHTML = playing ? "&#x23F8; Pause" : "&#x25B6; Play";
    if (playing) { lastTick = 0; requestAnimationFrame(tick); }
  }
  function tick(t) {
    if (!playing) return;
    const interval = 1000 / Number($("speed").value);
    if (!lastTick) lastTick = t;
    if (t - lastTick >= interval) {
      const steps = Math.min(LAST - frame, Math.floor((t - lastTick) / interval));
      lastTick += steps * interval;
      go(frame + Math.max(steps, 1));
      if (frame >= LAST) { setPlaying(false); return; }   // stop at the end
    }
    requestAnimationFrame(tick);
  }
  $("play").onclick = () => {
    if (!playing && frame >= LAST) go(0);                 // play again from the start
    setPlaying(!playing);
  };
  $("restart").onclick = () => { setPlaying(false); go(0); };
  $("back").onclick = () => { setPlaying(false); go(frame - 1); };
  $("step").onclick = () => { setPlaying(false); go(frame + 1); };
  $("end").onclick = () => { setPlaying(false); go(LAST); };
  $("slider").max = LAST;
  $("slider").oninput = (e) => { setPlaying(false); go(Number(e.target.value)); };
  if (D.stride > 1)
    $("note").textContent = "This run is too long to show every generation, so the " +
      "player shows every " + D.stride + "th. The CSV download has all of them.";
  // leave a menu once a speed is picked, so the keys work again
  $("speed").onchange = () => $("speed").blur();
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    // a focused button already answers Space and Enter itself; acting here too
    // would press it twice (Play: pause, then play again)
    if (e.target.tagName === "BUTTON" && (e.key === " " || e.key === "Enter")) return;
    if (e.key === " ") { e.preventDefault(); $("play").click(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); $("step").click(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); $("back").click(); }
    else if (e.key === "Home") { e.preventDefault(); $("restart").click(); }
    else if (e.key === "End") { e.preventDefault(); $("end").click(); }
  });
  $("save").onclick = () => {
    // the lattice at the current generation, with its numbers underneath
    const s = Math.max(1, Math.ceil(800 / Math.max(lat.width, lat.height)));
    const w = lat.width * s, lineH = 20, perLine = K > 8 ? 4 : K;
    const rows = 1 + Math.ceil(K / perLine), h = lat.height * s + 10 + rows * lineH;
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    const c = cv.getContext("2d");
    c.fillStyle = "#fff"; c.fillRect(0, 0, w, h);
    c.imageSmoothingEnabled = false;
    c.drawImage(lat, 0, 0, w, lat.height * s);
    c.font = "14px Helvetica, Arial, sans-serif"; c.textBaseline = "middle";
    let y = lat.height * s + 10 + lineH / 2;
    const r = D.r[frame];
    c.fillStyle = "#222";
    c.fillText("generation " + GENS[frame] + "    assortment r = " +
               (r === null ? "undefined" : r.toFixed(3)), 6, y);
    const colW = w / perLine;
    D.names.forEach((name, i) => {
      if (i % perLine === 0) y += lineH;
      const x = 6 + (i % perLine) * colW;
      c.fillStyle = D.colors[i]; c.fillRect(x, y - 6, 12, 12);
      c.fillStyle = "#222";
      c.fillText(name + "  " + (100 * D.freq[frame][i]).toFixed(1) + "%", x + 18, y);
    });
    const a = document.createElement("a");
    a.download = "lattice-generation-" + GENS[frame] + ".png";
    a.href = cv.toDataURL("image/png");
    document.body.append(a); a.click(); a.remove();
  };

  // ---- load ---------------------------------------------------------------
  async function inflate(b64) {
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  // relayout when the frame's width changes; the resize event alone is not
  // always delivered to an iframe whose host resizes it
  let laidOutWidth = -1;
  function relayoutIfWidthChanged() {
    const W = document.documentElement.clientWidth;
    if (W !== laidOutWidth) { laidOutWidth = W; layout(); }
  }
  window.addEventListener("resize", relayoutIfWidthChanged);
  if (typeof ResizeObserver !== "undefined")
    new ResizeObserver(relayoutIfWidthChanged).observe(document.documentElement);
  relayoutIfWidthChanged();
  if (typeof DecompressionStream === "undefined") {
    $("msg").textContent = "This browser is too old to play the run (it lacks " +
      "DecompressionStream). Any current Chrome, Firefox, Safari or Edge will work.";
    return;
  }
  inflate(D.grids).then((g) => {
    if (g.length !== (LAST + 1) * CELLS) throw new Error("lattice data is the wrong size");
    grids = g; prepareLattice(); layout();
  }).catch((e) => { $("msg").textContent = "Could not load the run: " + e.message; });
})();
</script>
</body></html>
"""
