/* Protein Annotation Workbench client.
   Sidebar navigator, tabbed Geneious-style detail views, inspector, export. */

"use strict";

/** Demo FASTA (two short synthetic bacterial fragments) for first-run trials. */
const DEMO_FASTA = `>demo_kinase_fragment
MKTAYIAKQRQISFVKSHFSRQDILDLWQKETLQKAVAGAPYKVTVLLVGDGAGKTTLLKRL
>demo_porin_fragment
MKKTAIAIAVALAGFATVAQAAPKDNTWYTGAKLGWSQYHDTGFINNNGPTHENQLGAGAF
`;

/** @type {{records: object[], tsv: string, columns: string[], meta: object|null}} */
const state = { records: [], tsv: "", columns: [], meta: null };
let selectedIdx = 0;
let selAnchor = null; // 0-based anchor for sequence selection
let selFocus = null;  // 0-based focus for sequence selection

const $ = (id) => document.getElementById(id);

/* ---------- formatting helpers ---------- */

/** Format an E-value the way HMMER/BLAST tables do (exponential when small). */
function fmtEvalue(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "n/a";
  if (v === 0) return "0";
  if (v < 0.001 || v >= 10000) return v.toExponential(2);
  return String(Math.round(v * 10000) / 10000);
}

/** Format a probability/confidence to 3 decimals. */
function fmtConf(x) {
  return Number(x).toFixed(3);
}

/** Escape text for safe HTML interpolation. */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ---------- backend health ---------- */

async function checkHealth() {
  const pill = $("backend-status");
  try {
    const r = await fetch("/health");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const h = await r.json();
    pill.textContent = `Backend OK · ${h.go_terms} GO terms · HMMER ${h.hmmer_available ? "available" : "unavailable"}`;
    pill.classList.add("ok");
  } catch {
    pill.textContent = "Backend unreachable";
    pill.classList.add("down");
  }
}

/* ---------- run ---------- */

function showError(msg) {
  const el = $("error");
  if (!msg) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false;
  el.textContent = msg;
}

function setExportEnabled(on) {
  for (const id of ["export-tsv", "export-json", "export-fasta"]) $(id).disabled = !on;
}

async function runAnnotate() {
  const btn = $("annotate-btn");
  const status = $("status");
  showError(null);
  btn.disabled = true;
  const t0 = performance.now();
  const timer = setInterval(() => {
    status.textContent = `Annotating… ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 200);
  try {
    const fd = new FormData();
    const f = $("fasta-file").files[0];
    if (f) fd.append("file", f);
    fd.append("fasta", $("fasta-text").value);
    fd.append("top_k", $("top-k").value);
    const mode = $("embed-mode").value;
    fd.append("no_esm", mode === "hashed" ? "true" : "false");
    fd.append("esm_model", mode === "hashed" ? "esm2_t6_8M_UR50D" : mode);
    fd.append("hmmer_db", $("hmmer-db").value);
    const r = await fetch("/annotate", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    state.records = data.records;
    state.tsv = data.tsv;
    state.columns = data.columns;
    state.meta = data.meta || null;
    selectedIdx = 0;
    setExportEnabled(true);
    $("protein-filter").disabled = false;
    $("protein-filter").value = "";
    renderList("");
    renderDetail(currentRecord());
    renderProvenance();
    const n = state.records.length;
    $("run-summary").textContent =
      `Annotated ${n} protein${n === 1 ? "" : "s"} in ${((performance.now() - t0) / 1000).toFixed(1)}s.`;
    status.textContent = `Done: ${n} protein${n === 1 ? "" : "s"}.`;
  } catch (e) {
    status.textContent = "";
    showError(`Annotation failed: ${e.message}`);
  } finally {
    clearInterval(timer);
    btn.disabled = false;
  }
}

/* ---------- protein navigator ---------- */

function currentRecord() {
  return state.records[selectedIdx];
}

function recordMatches(rec, q) {
  if (!q) return true;
  const hay = `${rec.protein_id} ${rec.predicted_function}`.toLowerCase();
  return hay.includes(q.toLowerCase());
}

function renderList(filter) {
  const ul = $("protein-list");
  ul.innerHTML = "";
  const visible = state.records
    .map((rec, i) => ({ rec, i }))
    .filter(({ rec }) => recordMatches(rec, filter));
  $("protein-count").textContent = state.records.length ? `(${visible.length}/${state.records.length})` : "";
  if (!visible.length) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="psub">${state.records.length ? "No matches." : "No proteins yet."}</span>`;
    ul.appendChild(li);
    return;
  }
  if (!visible.some(({ i }) => i === selectedIdx)) selectedIdx = visible[0].i;
  for (const { rec, i } of visible) {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", i === selectedIdx ? "true" : "false");
    if (i === selectedIdx) li.classList.add("active");
    const top = rec.go_terms && rec.go_terms.length
      ? `${rec.go_terms[0].id} · ${fmtConf(rec.go_terms[0].score)}` : "no GO hit";
    li.innerHTML = `<div class="pid">${esc(rec.protein_id)}</div>` +
      `<div class="psub">${rec.length} aa · ${esc(top)}</div>`;
    li.title = rec.predicted_function;
    li.onclick = () => selectProtein(i);
    ul.appendChild(li);
  }
}

function selectProtein(i) {
  selectedIdx = i;
  selAnchor = selFocus = null;
  renderList($("protein-filter").value);
  renderDetail(currentRecord());
}

/* ---------- detail views ---------- */

function renderDetail(rec) {
  if (!rec) return;
  $("empty").hidden = true;
  $("detail").hidden = false;
  $("prot-title").textContent = rec.protein_id;
  $("prot-badges").innerHTML =
    `<span class="tag">${rec.length} aa</span>` +
    `<span class="tag">${rec.molecular_weight.toFixed(2)} Da</span>` +
    `<span class="tag">pI ${rec.isoelectric_point.toFixed(2)}</span>` +
    (rec.n_masked ? `<span class="tag">${rec.n_masked} masked</span>` : "");
  $("prot-func").textContent = rec.predicted_function;
  renderGoTable(rec);
  renderProps(rec);
  renderSequence(rec);
  renderTracks(rec);
  renderComposition(rec);
  renderHmmer(rec);
  // Refresh visible charts (Plotly needs a nudge when its tab was hidden).
  requestAnimationFrame(() => {
    if (typeof Plotly !== "undefined") {
      for (const id of ["tracks-chart", "aa-chart"]) {
        const el = $(id);
        if (el && el.offsetParent !== null && el.data) {
          try { Plotly.Plots.resize(el); } catch { /* keep static */ }
        }
      }
    }
  });
}

function renderGoTable(rec) {
  const tb = document.querySelector("#go-table tbody");
  tb.innerHTML = rec.go_terms.map((g) => {
    const amigo = `https://amigo.geneontology.org/amigo/term/${encodeURIComponent(g.id)}`;
    const pct = Math.max(0, Math.min(1, g.score)) * 100;
    return `<tr><td>${esc(g.name)}</td>` +
      `<td><a href="${amigo}" target="_blank" rel="noopener">${esc(g.id)}</a></td>` +
      `<td><span class="ns ${esc(g.namespace)}">${esc(g.namespace)}</span></td>` +
      `<td><span class="conf"><span class="bar"><i style="width:${pct.toFixed(1)}%"></i></span>` +
      `<span class="val">${fmtConf(g.score)}</span></span></td></tr>`;
  }).join("");
}

function renderProps(rec) {
  const rows = [
    ["Length", `${rec.length} aa`],
    ["Molecular weight", `${rec.molecular_weight.toFixed(2)} Da`],
    ["Isoelectric point", rec.isoelectric_point.toFixed(2)],
    ["GRAVY", rec.gravy.toFixed(4)],
    ["Aromaticity", rec.aromaticity.toFixed(4)],
    ["Instability index", rec.instability_index.toFixed(2)],
    ["Aliphatic index", rec.aliphatic.toFixed(2)],
    ["Masked residues", String(rec.n_masked)],
  ];
  $("props-grid").innerHTML =
    rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
}

function renderHmmer(rec) {
  const body = $("hmmer-body");
  if (rec.hmmer) {
    body.innerHTML = `<dl class="kv">` +
      `<dt>Target</dt><dd>${esc(rec.hmmer.target)}</dd>` +
      `<dt>E-value</dt><dd>${esc(fmtEvalue(rec.hmmer.evalue))}</dd>` +
      `<dt>Bit score</dt><dd>${esc(String(rec.hmmer.bitscore))}</dd></dl>`;
  } else {
    body.innerHTML = `<p class="muted small">No HMMER homolog: no reference database was ` +
      `supplied or the <code>phmmer</code> binary is unavailable. Predictions rest on ` +
      `sequence features alone.</p>`;
  }
}

function renderProvenance() {
  const m = state.meta;
  const dl = $("provenance");
  if (!m) { dl.innerHTML = `<dt class="muted">No run yet.</dt>`; return; }
  const rows = [
    ["Proteins", String(m.n_proteins)],
    ["Embeddings", m.embeddings],
    ["ESM-2 model", m.esm_model],
    ["HMMER", m.hmmer_available ? (m.hmmer_db_provided ? `phmmer (${m.hmmer_hits} hits)` : "binary ready, no DB given") : "unavailable (skipped)"],
    ["GO label space", `${m.go_terms} terms`],
    ["Top-K", String(m.top_k)],
    ["Thresholds", "default 0.5 (demo weights, untuned)"],
    ["Server time", `${m.elapsed_s.toFixed(2)}s`],
  ];
  dl.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("run-meta").textContent = `${m.embeddings} · ${m.esm_model}`;
}

/* ---------- sequence viewer ---------- */

const SEQ_WIDTH = 60; // residues per row, Geneious-style blocks of 10

function renderSequence(rec) {
  const viewer = $("seq-viewer");
  viewer.innerHTML = "";
  const seq = rec.sequence;
  for (let s = 0; s < seq.length; s += SEQ_WIDTH) {
    const block = document.createElement("div");
    block.className = "seq-block";
    // Ruler row: tick label every 10 residues, right-aligned to its block.
    const ruler = document.createElement("div");
    ruler.className = "seq-ruler";
    ruler.setAttribute("aria-hidden", "true");
    ruler.appendChild(Object.assign(document.createElement("span"), { className: "corner" }));
    for (let t = 10; t <= SEQ_WIDTH && s + t <= seq.length + 9; t += 10) {
      const tick = document.createElement("span");
      tick.textContent = Math.min(s + t, seq.length);
      ruler.appendChild(tick);
    }
    block.appendChild(ruler);
    const row = document.createElement("div");
    row.className = "seq-row";
    const pos = document.createElement("span");
    pos.className = "pos";
    pos.textContent = s + 1;
    row.appendChild(pos);
    for (let i = s; i < Math.min(s + SEQ_WIDTH, seq.length); i++) {
      const sp = document.createElement("span");
      sp.className = `aa ${rec.aa_class[i]}`;
      sp.textContent = seq[i];
      sp.dataset.idx = i;
      sp.title = `${seq[i]}${i + 1}`;
      row.appendChild(sp);
    }
    block.appendChild(row);
    viewer.appendChild(block);
  }
  viewer.onmousedown = (e) => {
    const t = e.target.closest(".aa");
    if (!t) return;
    selAnchor = selFocus = Number(t.dataset.idx);
    paintSelection();
    e.preventDefault();
  };
  viewer.onmouseover = (e) => {
    if (selAnchor === null || !(e.buttons & 1)) return;
    const t = e.target.closest(".aa");
    if (!t) return;
    selFocus = Number(t.dataset.idx);
    paintSelection();
  };
  document.onmouseup = () => { /* keep selection painted after drag */ };
  updateSelReadout();
}

function selectionRange() {
  if (selAnchor === null || selFocus === null) return null;
  const lo = Math.min(selAnchor, selFocus);
  const hi = Math.max(selAnchor, selFocus);
  return [lo, hi];
}

function paintSelection() {
  const range = selectionRange();
  for (const el of document.querySelectorAll("#seq-viewer .aa")) {
    const i = Number(el.dataset.idx);
    el.classList.toggle("sel", !!range && i >= range[0] && i <= range[1]);
  }
  updateSelReadout();
}

function updateSelReadout() {
  const range = selectionRange();
  const rec = currentRecord();
  const readout = $("sel-readout");
  $("copy-sel").disabled = !range;
  if (!range || !rec) { readout.textContent = "No selection"; return; }
  const [lo, hi] = range;
  readout.textContent = `Selection ${lo + 1}–${hi + 1} (${hi - lo + 1} aa)`;
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const old = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = old; }, 1200);
  } catch {
    btn.textContent = "Copy failed";
  }
}

/* ---------- charts ---------- */

/** Run a Plotly render, tolerating the CDN script still loading or blocked. */
function withPlotly(fn) {
  if (typeof Plotly !== "undefined") { fn(); return; }
  // Retry once after full page load (deferred CDN script), else show fallback.
  window.addEventListener("load", () => {
    if (typeof Plotly !== "undefined") fn();
    else $("tracks-fallback").hidden = false;
  }, { once: true });
}

function renderTracks(rec) {
  withPlotly(() => {
    const x = rec.sequence.split("").map((_, i) => i + 1);
    const layout = {
      margin: { t: 20, l: 48, r: 12, b: 42 },
      grid: { rows: 2, columns: 1, pattern: "independent" },
      xaxis: { title: "position", matches: "x2" },
      yaxis: { title: "KD", domain: [0.55, 1] },
      xaxis2: { title: "position", anchor: "y2" },
      yaxis2: { title: "charge", domain: [0, 0.32], tickvals: [-1, 0, 1] },
      legend: { orientation: "h", y: 1.06 },
      hovermode: "x unified",
    };
    Plotly.newPlot("tracks-chart", [
      { x, y: rec.tracks.hydro, xaxis: "x", yaxis: "y", type: "scatter",
        mode: "lines", name: "KD raw", line: { color: "#9aa7b3", width: 1 } },
      { x, y: rec.tracks.hydro_smooth, xaxis: "x", yaxis: "y", type: "scatter",
        mode: "lines", name: "KD smoothed (w=5)", line: { color: "#0b63ce", width: 2 } },
      { x, y: rec.tracks.charge, xaxis: "x2", yaxis: "y2", type: "bar",
        name: "charge", marker: { color: "#1a7f4b" } },
    ], layout, { responsive: true, displaylogo: false });
  });
}

function renderComposition(rec) {
  const keys = Object.keys(rec.aa_freq);
  const vals = Object.values(rec.aa_freq);
  withPlotly(() => {
    Plotly.newPlot("aa-chart", [{
      x: keys, y: vals, type: "bar", marker: { color: "#0b63ce" },
    }], { margin: { t: 20, l: 48, r: 12, b: 44 },
      xaxis: { title: "residue" }, yaxis: { title: "fraction" } },
      { responsive: true, displaylogo: false });
  });
  // Accessible data table mirrors the chart (also the offline fallback).
  const total = rec.length || 1;
  document.querySelector("#aa-table tbody").innerHTML = keys.map((k, j) =>
    `<tr><td><code>${esc(k)}</code></td><td class="num">${Number(vals[j]).toFixed(4)}</td>` +
    `<td class="num">${Math.round(Number(vals[j]) * total)}</td></tr>`).join("");
}

/* ---------- tabs ---------- */

const TABS = ["overview", "sequence", "tracks", "composition"];

function activateTab(name, focus) {
  for (const t of TABS) {
    const tab = $(`tab-${t}`);
    const panel = $(`panel-${t}`);
    const on = t === name;
    tab.setAttribute("aria-selected", on ? "true" : "false");
    tab.tabIndex = on ? 0 : -1;
    panel.hidden = !on;
    if (on && focus) tab.focus();
  }
  // Charts in a newly shown tab need a resize nudge.
  requestAnimationFrame(() => {
    if (typeof Plotly === "undefined") return;
    for (const id of ["tracks-chart", "aa-chart"]) {
      const el = $(id);
      if (el && el.offsetParent !== null && el.data) {
        try { Plotly.Plots.resize(el); } catch { /* keep static */ }
      }
    }
  });
}

/* ---------- export ---------- */

function download(filename, text, mime) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function exportJson() {
  download("protein_annotations.json",
    JSON.stringify({ meta: state.meta, columns: state.columns, records: state.records }, null, 2),
    "application/json");
}

function exportFasta() {
  const fasta = state.records.map((r) =>
    `>${r.protein_id} ${r.predicted_function}\n${r.sequence.replace(/(.{60})/g, "$1\n").trim()}`
  ).join("\n");
  download("protein_cleaned.fasta", fasta + "\n", "text/plain");
}

/* ---------- wiring ---------- */

$("annotate-btn").onclick = runAnnotate;
$("demo-btn").onclick = () => {
  $("fasta-text").value = DEMO_FASTA;
  $("status").textContent = "Demo sequences loaded. Press Annotate.";
};
$("protein-filter").oninput = (e) => renderList(e.target.value);
$("protein-list").onkeydown = (e) => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();
  const d = e.key === "ArrowDown" ? 1 : -1;
  const next = Math.max(0, Math.min(state.records.length - 1, selectedIdx + d));
  if (next !== selectedIdx) selectProtein(next);
};
for (const t of TABS) {
  $(`tab-${t}`).onclick = () => activateTab(t, false);
}
document.querySelector(".tabs").onkeydown = (e) => {
  const i = TABS.findIndex((t) => $(`tab-${t}`) === document.activeElement);
  if (i < 0) return;
  let j = null;
  if (e.key === "ArrowRight") j = (i + 1) % TABS.length;
  if (e.key === "ArrowLeft") j = (i - 1 + TABS.length) % TABS.length;
  if (j !== null) { e.preventDefault(); activateTab(TABS[j], true); }
};
$("copy-seq").onclick = (e) => {
  const rec = currentRecord();
  if (rec) copyText(rec.sequence, e.currentTarget);
};
$("copy-sel").onclick = (e) => {
  const rec = currentRecord();
  const range = selectionRange();
  if (rec && range) copyText(rec.sequence.slice(range[0], range[1] + 1), e.currentTarget);
};
$("export-tsv").onclick = () => download("protein_annotations.tsv", state.tsv, "text/tab-separated-values");
$("export-json").onclick = exportJson;
$("export-fasta").onclick = exportFasta;
// Close the export menu after a choice (native <details> stays open otherwise).
document.querySelectorAll("#export-menu .menu-list button").forEach((b) =>
  b.addEventListener("click", () => { $("export-menu").open = false; }));

checkHealth();
