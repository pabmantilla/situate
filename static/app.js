const PX = 5, H = 140, MID = H / 2;
const BASES = ["A", "C", "G", "T"];
const COL = { A: "#2ca02c", C: "#1f77b4", G: "#ff7f0e", T: "#d62728" };
let N = 0, L = 0, cur = 0, genK = 0, HAS_OHE = false, MODE = "hyp", PP = 10;
let lastData = [];
const SVGNS = "http://www.w3.org/2000/svg";

async function init() {
  const m = await (await fetch("/meta")).json();
  N = m.n_seqs; L = m.seq_len; HAS_OHE = !!m.has_ohe;
  document.getElementById("prev").onclick = () => { cur = Math.max(0, cur - PP); load(); };
  document.getElementById("next").onclick = () => { cur = Math.min(Math.max(0, N - PP), cur + PP); load(); };
  const pp = document.getElementById("pp");
  PP = parseInt(pp.value) || 10;
  pp.onchange = () => { PP = parseInt(pp.value) || 10; load(); };
  document.getElementById("go").onclick = () => {
    let v = parseInt(document.getElementById("jump").value) || 0;
    cur = Math.max(0, Math.min(N - 1, v)); load();
  };
  for (const r of document.querySelectorAll('input[name="mode"]')) {
    if (r.value === "obs") r.disabled = !HAS_OHE;
    r.onchange = () => { if (r.checked) { MODE = r.value; render(); } };
  }
  load();
}

async function load() {
  lastData = await (await fetch(`/batch?start=${cur}&n=${PP}`)).json();
  const last = cur + lastData.length - 1;
  document.getElementById("range").textContent = `idx ${cur}–${last} of ${N}`;
  document.getElementById("jump").value = cur;
  render();
}

function render() {
  const root = document.getElementById("rows");
  root.innerHTML = "";
  for (const d of lastData) root.appendChild(makeRow(d));
}

function hypExtent(track) {
  let mx = 1e-9;
  for (const t of track) {
    let pos = 0, neg = 0;
    for (const v of t.v) (v >= 0 ? pos += v : neg += -v);
    mx = Math.max(mx, pos, neg);
  }
  return mx;
}

function makeRow(d) {
  const row = document.createElement("div");
  row.className = "row";
  const lbl = document.createElement("div");
  lbl.className = "lbl"; lbl.textContent = d.idx;
  const scroll = document.createElement("div");
  scroll.className = "scroll";
  const plot = document.createElement("div");
  plot.className = "plot";
  const W = L * PX;
  plot.style.width = W + "px";

  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  drawTrack(cv.getContext("2d"), d.track);
  plot.appendChild(cv);

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  plot.appendChild(svg);

  const spans = d.spans.map(s => ({ ...s }));
  const ctx = { svg, spans, W, sel: null };
  renderSpans(ctx);
  attachCreate(ctx);

  scroll.appendChild(plot);
  const setBtn = document.createElement("button");
  setBtn.textContent = "Set";
  setBtn.onclick = async () => {
    const r = await fetch(`/set/${d.idx}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spans: ctx.spans }),
    });
    if ((await r.json()).ok) {
      row.classList.add("flash");
      setTimeout(() => row.classList.remove("flash"), 800);
    }
  };
  row.tabIndex = 0;
  row.addEventListener("keydown", e => {
    if ((e.key === "Delete" || e.key === "Backspace") && ctx.sel != null) {
      ctx.spans.splice(ctx.sel, 1); ctx.sel = null; renderSpans(ctx);
    }
  });
  row.appendChild(lbl); row.appendChild(scroll); row.appendChild(setBtn);
  return row;
}

function drawTrack(g, track) {
  g.clearRect(0, 0, track.length * PX, H);
  g.strokeStyle = "#ddd"; g.beginPath();
  g.moveTo(0, MID); g.lineTo(track.length * PX, MID); g.stroke();
  g.textAlign = "center"; g.font = "10px monospace"; g.textBaseline = "alphabetic";
  if (MODE === "obs" && HAS_OHE) {
    let mx = 1e-9;
    for (const t of track) mx = Math.max(mx, Math.abs(t.h));
    for (let p = 0; p < track.length; p++) {
      const t = track[p], frac = t.h / mx, hpx = Math.abs(frac) * MID;
      if (hpx < 0.5) continue;
      const x = p * PX + PX / 2, scale = hpx / 8;
      g.save();
      g.translate(x, MID);
      g.scale(1, frac >= 0 ? -scale : scale);
      g.fillStyle = COL[t.b] || "#000";
      g.fillText(t.b, 0, 8);
      g.restore();
    }
    return;
  }
  // Hypothetical: stack all 4 letters, +ve up / -ve down, biggest near line.
  const mx = hypExtent(track);
  for (let p = 0; p < track.length; p++) {
    const x = p * PX + PX / 2;
    const ch = BASES.map((b, c) => ({ b, val: track[p].v[c] }));
    const pos = ch.filter(o => o.val > 0).sort((a, b) => a.val - b.val);
    const neg = ch.filter(o => o.val < 0).sort((a, b) => b.val - a.val);
    let y = MID;  // upward stack
    for (const o of pos) {
      const hpx = (o.val / mx) * MID;
      if (hpx >= 0.5) drawLetter(g, o.b, x, y, hpx, -1);
      y -= hpx;
    }
    y = MID;      // downward stack
    for (const o of neg) {
      const hpx = (-o.val / mx) * MID;
      if (hpx >= 0.5) drawLetter(g, o.b, x, y, hpx, 1);
      y += hpx;
    }
  }
}

function drawLetter(g, b, x, yBase, hpx, dir) {
  // dir -1 grows upward (letter sits above yBase), +1 downward
  g.save();
  g.translate(x, yBase);
  g.scale(1, dir * (hpx / 8));
  g.fillStyle = COL[b] || "#000";
  g.fillText(b, 0, dir < 0 ? 0 : 8);
  g.restore();
}

function renderSpans(ctx) {
  const { svg, spans } = ctx;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  spans.forEach((s, i) => {
    const x0 = s.start * PX, x1 = s.end * PX;
    const r = document.createElementNS(SVGNS, "rect");
    r.setAttribute("class", "span-rect" + (ctx.sel === i ? " sel" : ""));
    r.setAttribute("x", x0); r.setAttribute("y", 0);
    r.setAttribute("width", Math.max(1, x1 - x0)); r.setAttribute("height", H);
    r.onmousedown = e => startDrag(e, ctx, i, "move");
    svg.appendChild(r);
    for (const [edge, ex] of [["start", x0], ["end", x1]]) {
      const hd = document.createElementNS(SVGNS, "rect");
      hd.setAttribute("class", "handle");
      hd.setAttribute("x", ex - 3); hd.setAttribute("y", 0);
      hd.setAttribute("width", 6); hd.setAttribute("height", H);
      hd.onmousedown = e => startDrag(e, ctx, i, edge);
      svg.appendChild(hd);
    }
    const tx = document.createElementNS(SVGNS, "text");
    tx.setAttribute("class", "span-lbl");
    tx.setAttribute("x", x0 + 2); tx.setAttribute("y", 12);
    tx.textContent = s.motif_name;
    svg.appendChild(tx);
  });
}

function svgX(ctx, e) {
  const rb = ctx.svg.getBoundingClientRect();
  return Math.round(Math.max(0, Math.min(L, (e.clientX - rb.left) / PX)));
}

function startDrag(e, ctx, i, mode) {
  e.stopPropagation(); e.preventDefault();
  ctx.sel = i; renderSpans(ctx);
  const s = ctx.spans[i];
  const startPos = svgX(ctx, e), s0 = s.start, e0 = s.end;
  function mv(ev) {
    const p = svgX(ctx, ev), dp = p - startPos;
    if (mode === "start") s.start = Math.min(s.end - 1, Math.max(0, p));
    else if (mode === "end") s.end = Math.max(s.start + 1, Math.min(L, p));
    else {
      let ns = s0 + dp, ne = e0 + dp, w = e0 - s0;
      ns = Math.max(0, Math.min(L - w, ns)); ne = ns + w;
      s.start = ns; s.end = ne;
    }
    renderSpans(ctx);
  }
  function up() {
    document.removeEventListener("mousemove", mv);
    document.removeEventListener("mouseup", up);
  }
  document.addEventListener("mousemove", mv);
  document.addEventListener("mouseup", up);
}

function attachCreate(ctx) {
  ctx.svg.addEventListener("mousedown", e => {
    if (e.target !== ctx.svg) return;       // started on bg only
    e.preventDefault();
    const p0 = svgX(ctx, e);
    const sp = { motif_name: `user pattern ${genK++}`, start: p0, end: p0 + 1, strand: "+" };
    ctx.spans.push(sp);
    const i = ctx.spans.length - 1;
    ctx.sel = i;
    function mv(ev) {
      const p = svgX(ctx, ev);
      sp.start = Math.max(0, Math.min(p0, p));
      sp.end = Math.max(sp.start + 1, Math.min(L, Math.max(p0, p)));
      renderSpans(ctx);
    }
    function up() {
      document.removeEventListener("mousemove", mv);
      document.removeEventListener("mouseup", up);
    }
    document.addEventListener("mousemove", mv);
    document.addEventListener("mouseup", up);
    renderSpans(ctx);
  });
}

init();
