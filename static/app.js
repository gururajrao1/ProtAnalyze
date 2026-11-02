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
    pill.textContent = `Backend OK · ${h.go_terms} GO terms · HMMER ${h.hmmer_available ? "on" : "off"} · AMR ${h.amrfinder_available ? "on" : "off"}`;
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
  btn.setAttribute("aria-busy", "true");
  const label = btn.textContent;
  btn.textContent = "Annotating…";
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
    fd.append("eval_labels", $("eval-labels").value);
    fd.append("go_map", $("go-map").value);
    fd.append("blend_alpha", $("blend-alpha").value);
    fd.append("obo", $("obo-text").value);
    fd.append("run_amrfinder", $("run-amrfinder").checked ? "true" : "false");
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
    btn.removeAttribute("aria-busy");
    btn.textContent = label;
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
  const scrollTop = ul.scrollTop;
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
  ul.scrollTop = scrollTop;
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
  const embLabel = rec.embedding_source === "hashed-fallback" ? "hashed embeddings"
    : rec.embedding_source ? String(rec.embedding_source).replace("esm2:", "ESM-2 ") : "";
  const disBadge = (rec.disordered_fraction === null || rec.disordered_fraction === undefined) ? ""
    : `<span class="tag${rec.disordered_fraction >= 0.5 ? " warn" : ""}">disorder ${(rec.disordered_fraction * 100).toFixed(1)}%</span>`;
  const ssBadge = (!rec.ss || rec.ss.source === "skipped") ? ""
    : rec.ss.helix_f >= 0.5 ? `<span class="tag">mostly-helical ${(rec.ss.helix_f * 100).toFixed(0)}%</span>`
    : rec.ss.sheet_f >= 0.3 ? `<span class="tag">sheet-rich ${(rec.ss.sheet_f * 100).toFixed(0)}%</span>`
    : `<span class="tag">SS3 H${(rec.ss.helix_f * 100).toFixed(0)}/E${(rec.ss.sheet_f * 100).toFixed(0)}</span>`;
  $("prot-badges").innerHTML =
    `<span class="tag">${rec.length} aa</span>` +
    `<span class="tag">${rec.molecular_weight.toFixed(2)} Da</span>` +
    `<span class="tag">pI ${rec.isoelectric_point.toFixed(2)}</span>` +
    (rec.n_masked ? `<span class="tag">${rec.n_masked} masked</span>` : "") +
    (embLabel ? `<span class="tag">${esc(embLabel)}</span>` : "") +
    disBadge +
    ssBadge +
    (rec.vfdb ? `<span class="tag vfdb">VFDB virulence hit</span>` : "") +
    (rec.blended ? `<span class="tag">blended scores</span>` : "") +
    (rec.amr && rec.amr.length ? `<span class="tag warn">${rec.amr.length} AMR hit${rec.amr.length === 1 ? "" : "s"}</span>` : "");
  $("prot-func").textContent = rec.predicted_function;
  renderGoTable(rec);
  renderProps(rec);
  renderSequence(rec);
  renderSsStrip(rec);
  renderTracks(rec);
  renderComposition(rec);
  renderLandscape(rec);
  renderStructure(rec);
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
    ["Disordered fraction", (rec.disordered_fraction === null || rec.disordered_fraction === undefined)
      ? "n/a" : `${(rec.disordered_fraction * 100).toFixed(1)}%`],
    ["Embedding source", rec.embedding_source || "n/a"],
    ["SS3 helix / sheet", (!rec.ss || rec.ss.source === "skipped")
      ? "n/a" : `${(rec.ss.helix_f * 100).toFixed(1)}% / ${(rec.ss.sheet_f * 100).toFixed(1)}%`],
    ["SS3 source", (rec.ss && rec.ss.source) || "n/a"],
  ];
  $("props-grid").innerHTML =
    rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
}

function renderHmmer(rec) {
  const body = $("hmmer-body");
  if (rec.hmmer) {
    body.innerHTML =
      (rec.vfdb ? `<p><span class="tag vfdb">VFDB virulence factor</span></p>` : "") +
      `<dl class="kv">` +
      `<dt>Target</dt><dd>${esc(rec.hmmer.target)}</dd>` +
      `<dt>E-value</dt><dd>${esc(fmtEvalue(rec.hmmer.evalue))}</dd>` +
      `<dt>Bit score</dt><dd>${esc(String(rec.hmmer.bitscore))}</dd></dl>`;
  } else {
    body.innerHTML = `<p class="muted small">No HMMER homolog: no reference database was ` +
      `supplied or the <code>phmmer</code> binary is unavailable. Predictions rest on ` +
      `sequence features alone.</p>`;
  }
  // AMR hits from the optional AMRFinderPlus scan (per-protein list).
  if (rec.amr && rec.amr.length) {
    body.innerHTML += `<h3>AMRFinderPlus hits (${rec.amr.length})</h3>` +
      `<table class="data"><thead><tr><th scope="col">Gene</th><th scope="col">Class</th>` +
      `<th scope="col" class="num">Id%</th></tr></thead><tbody>` +
      rec.amr.map((a) =>
        `<tr><td>${esc(a.gene)} <span class="muted small">${esc(a.element_type)}</span></td>` +
        `<td>${esc(a.class || "-")}</td><td class="num">${Number(a.pct_identity).toFixed(1)}</td></tr>`
      ).join("") + `</tbody></table>`;
  }
}

function renderProvenance() {
  const m = state.meta;
  const dl = $("provenance");
  if (!m) { dl.innerHTML = `<dt class="muted">No run yet.</dt>`; return; }
  const hmmerDb = m.hmmer_db_kind === "vfdb" ? "VFDB virulence tier"
    : m.hmmer_db_kind === "custom" ? "custom FASTA" : "none";
  const rows = [
    ["Proteins", String(m.n_proteins)],
    ["Embeddings", m.embeddings],
    ["ESM-2 model", m.esm_model],
    ["HMMER", m.hmmer_available ? (m.hmmer_db_provided ? `phmmer (${m.hmmer_hits} hits, DB: ${hmmerDb})` : "binary ready, no DB given") : "unavailable (skipped)"],
    ["VFDB hits", String(m.vfdb_hits || 0)],
    ["GO label space", `${m.go_terms} terms`],
    ["GO weights", m.go_weights || "demo random-init (untrained)"],
    ["Top-K", String(m.top_k)],
    ["Thresholds", "default 0.5 (demo weights, untuned)"],
    ["CAFA eval", m.eval ? `Fmax ${m.eval.fmax} · Smin ${m.eval.smin} · AUPR ${m.eval.aupr_micro}` : "no labels supplied"],
    ["Homology blend", m.blend ? (m.blend.active ? `active (α=${m.blend.alpha}, map ${m.blend.map_targets}, OBO ${m.blend.obo_terms})` : `off (α=${m.blend.alpha})`) : "off"],
    ["AMR scan", m.amr ? (m.amr.requested ? (m.amr.available ? `${m.amr.hits} hits / ${m.amr.proteins_with_hits} proteins` : "requested, binary unavailable (skipped)") : "not requested") : "not requested"],
    ["SS3 structure", m.ss ? (m.ss.active ? (m.ss.head ? `active (ESM-2 head ${m.ss.head})` : "active (heuristic fallback)") : "skipped (--no-ss)") : "n/a"],
    ["Server time", `${m.elapsed_s.toFixed(2)}s`],
  ];
  dl.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("run-meta").textContent = `${m.embeddings} · ${m.esm_model}`;
  // About panel reflects whether scores came from trained or demo weights.
  $("about-text").textContent = (m.go_weights && m.go_weights.startsWith("trained"))
    ? `GO confidences come from a multilabel head (${m.go_weights}) over fused sequence and ProtParam features, evaluated at Fmax 0.74 on a held-out bacterial split. Confirm hits against HMMER homology and the literature before reporting a function.`
    : "GO confidences come from a multilabel head over fused ESM-2 and ProtParam features. This run used demonstration weights: treat scores as ranking signals and confirm hits against HMMER homology and the literature before reporting a function.";
  renderEval(m.eval);
}

function renderEval(ev) {
  const body = $("eval-body");
  if (!ev) {
    body.innerHTML = `<p class="muted small">No eval labels supplied for this run. ` +
      `Paste <code>protein_id&lt;TAB&gt;GO:1,GO:2</code> lines in Run setup to score ` +
      `Fmax / Smin / AUPR.</p>`;
    return;
  }
  const perTerm = ev.per_term.map((t) =>
    `<tr><td><code>${esc(t.go_id)}</code></td><td class="num">${t.f1.toFixed(4)}</td>` +
    `<td class="num">${t.threshold.toFixed(2)}</td></tr>`).join("");
  body.innerHTML = `<dl class="kv">` +
    `<dt>Fmax (protein-centric)</dt><dd>${ev.fmax.toFixed(4)} @ ${ev.fmax_threshold.toFixed(2)}</dd>` +
    `<dt>Smin (semantic distance)</dt><dd>${ev.smin.toFixed(4)} @ ${ev.smin_threshold.toFixed(2)}</dd>` +
    `<dt>Micro-AUPR</dt><dd>${ev.aupr_micro.toFixed(4)}</dd>` +
    `<dt>Proteins / terms</dt><dd>${ev.n_proteins} / ${ev.n_terms}</dd></dl>` +
    `<table class="data compact"><thead><tr><th scope="col">Term</th>` +
    `<th scope="col" class="num">Best F1</th><th scope="col" class="num">Thr</th></tr></thead>` +
    `<tbody>${perTerm}</tbody></table>`;
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

/** Render the H/E/C letter strip under the sequence viewer (toggle-gated). */
function renderSsStrip(rec) {
  const strip = $("ss-strip");
  const ss3 = rec.tracks && Array.isArray(rec.tracks.ss3) ? rec.tracks.ss3 : [];
  strip.innerHTML = "";
  strip.hidden = !$("ss-strip-toggle").checked || ss3.length !== rec.sequence.length;
  if (strip.hidden) return;
  const cls = { H: "ss-h", E: "ss-e", C: "ss-c" };
  for (let s = 0; s < ss3.length; s += SEQ_WIDTH) {
    const block = document.createElement("div");
    block.className = "seq-block";
    const row = document.createElement("div");
    row.className = "seq-row";
    const pos = document.createElement("span");
    pos.className = "pos";
    pos.textContent = s + 1;
    row.appendChild(pos);
    for (let i = s; i < Math.min(s + SEQ_WIDTH, ss3.length); i++) {
      const sp = document.createElement("span");
      sp.className = `aa ${cls[ss3[i]] || "ss-c"}`;
      sp.textContent = ss3[i];
      sp.title = `${ss3[i]}${i + 1}`;
      row.appendChild(sp);
    }
    block.appendChild(row);
    strip.appendChild(block);
  }
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
    const disorder = Array.isArray(rec.disorder) ? rec.disorder : [];
    const ss3 = rec.tracks && Array.isArray(rec.tracks.ss3) ? rec.tracks.ss3 : [];
    const layout = {
      margin: { t: 20, l: 48, r: 12, b: 42 },
      height: 660,
      grid: { rows: 4, columns: 1, pattern: "independent" },
      xaxis: { matches: "x4" },
      yaxis: { title: "KD", domain: [0.77, 1] },
      xaxis2: { matches: "x4" },
      yaxis2: { title: "charge", domain: [0.58, 0.7], tickvals: [-1, 0, 1] },
      xaxis3: { matches: "x4" },
      yaxis3: { title: "disorder", domain: [0.32, 0.5], range: [0, 1] },
      xaxis4: { title: "position", anchor: "y4" },
      yaxis4: { title: "SS3", domain: [0, 0.22], tickvals: [0, 1, 2],
        ticktext: ["C", "E", "H"], range: [-0.3, 2.3] },
      legend: { orientation: "h", y: 1.03 },
      hovermode: "x unified",
    };
    const traces = [
      { x, y: rec.tracks.hydro, xaxis: "x", yaxis: "y", type: "scatter",
        mode: "lines", name: "KD raw", line: { color: "#9aa7b3", width: 1 } },
      { x, y: rec.tracks.hydro_smooth, xaxis: "x", yaxis: "y", type: "scatter",
        mode: "lines", name: "KD smoothed (w=5)", line: { color: "#0b63ce", width: 2 } },
      { x, y: rec.tracks.charge, xaxis: "x2", yaxis: "y2", type: "bar",
        name: "charge", marker: { color: "#1a7f4b" } },
    ];
    if (disorder.length === x.length) {
      traces.push(
        { x, y: disorder, xaxis: "x3", yaxis: "y3", type: "scatter",
          mode: "lines", name: "disorder (w=9)", line: { color: "#b35409", width: 2 } },
        { x: [x[0], x[x.length - 1]], y: [0.5, 0.5], xaxis: "x3", yaxis: "y3",
          type: "scatter", mode: "lines", name: "disordered ≥ 0.5",
          line: { color: "#b4232a", width: 1, dash: "dash" } },
      );
    }
    if (ss3.length === x.length) {
      const code = { C: 0, E: 1, H: 2 };
      traces.push(
        { x, y: ss3.map((s) => code[s] ?? 0), xaxis: "x4", yaxis: "y4",
          type: "scatter", mode: "lines", name: "SS3 (H/E/C)",
          line: { color: "#6a3fc0", width: 2, shape: "hv" } },
      );
    }
    Plotly.newPlot("tracks-chart", traces, layout, { responsive: true, displaylogo: false });
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

/* ---------- 3D residue landscape (Three.js, optional CDN) ---------- */

let landscape = null; // { renderer, raf, group, spinning } while mounted
let landscapeLoadHook = false;

function disposeLandscape() {
  if (!landscape) return;
  cancelAnimationFrame(landscape.raf);
  if (landscape.renderer) landscape.renderer.dispose();
  const el = $("landscape-3d");
  if (el) el.innerHTML = "";
  landscape = null;
}

/** Soft round sprite so points read as glowing dots, not hard squares. */
function landscapeSprite() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.8)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function ssColor(s) {
  if (s === "H") return new THREE.Color("#ff9ebb");
  if (s === "E") return new THREE.Color("#7cc4ff");
  return new THREE.Color("#93a3ae");
}

/**
 * Residue property landscape: chain position (x), smoothed KD
 * hydrophobicity (y), charge (z); points glow by SS3 state with a faint
 * backbone thread. Drag to rotate; gentle auto-spin unless reduced motion.
 */
function renderLandscape(rec) {
  const fallback = $("landscape-fallback");
  disposeLandscape();
  if (typeof THREE === "undefined") {
    // Retry once after full page load (deferred CDN script), else fallback.
    if (!landscapeLoadHook) {
      landscapeLoadHook = true;
      window.addEventListener("load", () => {
        if (typeof THREE !== "undefined") {
          fallback.hidden = true;
          if (!$("panel-landscape").hidden) renderLandscape(currentRecord());
        } else fallback.hidden = false;
      }, { once: true });
    }
    fallback.hidden = false;
    return;
  }
  fallback.hidden = true;
  const el = $("landscape-3d");
  // Hidden tab has no width yet; render when the tab is activated instead.
  if (!el || el.clientWidth === 0) return;
  const n = rec.sequence.length;
  if (!n) return;
  const hydro = rec.tracks.hydro_smooth;
  const charge = rec.tracks.charge;
  const ss3 = (rec.tracks && Array.isArray(rec.tracks.ss3)) ? rec.tracks.ss3 : [];
  if (hydro.length !== n || charge.length !== n) return;

  const W = el.clientWidth, H = 420;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  camera.position.set(0, 2.6, 11.5);
  camera.lookAt(0, 0.2, 0);

  const group = new THREE.Group();
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = (n === 1 ? 0 : i / (n - 1) - 0.5) * 8;
    pos[i * 3] = x;
    pos[i * 3 + 1] = hydro[i] / 1.4;
    pos[i * 3 + 2] = charge[i] * 2.5;
    const c = ssColor(ss3[i]);
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const dots = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.3, map: landscapeSprite(), vertexColors: true,
    transparent: true, depthWrite: false, sizeAttenuation: true,
  }));
  const thread = new THREE.Line(geo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.45,
  }));
  group.add(thread);
  group.add(dots);
  scene.add(group);

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let dragging = false, px = 0, py = 0;
  renderer.domElement.style.touchAction = "none";
  renderer.domElement.addEventListener("pointerdown", (e) => {
    dragging = true; px = e.clientX; py = e.clientY;
    renderer.domElement.setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    group.rotation.y += (e.clientX - px) * 0.006;
    group.rotation.x = Math.max(-1.2, Math.min(1.2, group.rotation.x + (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY;
  });
  const stopDrag = () => { dragging = false; };
  renderer.domElement.addEventListener("pointerup", stopDrag);
  renderer.domElement.addEventListener("pointercancel", stopDrag);

  const clock = new THREE.Clock();
  const st = { renderer, raf: 0, group, camera };
  landscape = st;
  (function tick() {
    st.raf = requestAnimationFrame(tick);
    if (!reduceMotion && !dragging) group.rotation.y += clock.getDelta() * 0.22;
    else clock.getDelta();
    renderer.render(scene, camera);
  })();
}

/* ---------- tabs ---------- */

const TABS = ["overview", "sequence", "tracks", "composition", "landscape", "structure"];

function activateTab(name, focus) {
  const detail = $("detail");
  detail.classList.remove("tab-switch");
  // Force reflow so the panel fade replays on every switch.
  void detail.offsetWidth;
  detail.classList.add("tab-switch");
  for (const t of TABS) {
    const tab = $(`tab-${t}`);
    const panel = $(`panel-${t}`);
    const on = t === name;
    tab.setAttribute("aria-selected", on ? "true" : "false");
    tab.tabIndex = on ? 0 : -1;
    panel.hidden = !on;
    if (on && focus) tab.focus();
  }
  // The 3D tabs render lazily: hidden tabs have no width to size to.
  if (name === "landscape") renderLandscape(currentRecord());
  if (name === "structure") renderStructure(currentRecord());
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
let filterTimer = null;
$("protein-filter").oninput = (e) => {
  // Debounce so fast typing does not rebuild the list on every keystroke.
  clearTimeout(filterTimer);
  const value = e.target.value;
  filterTimer = setTimeout(() => renderList(value), 140);
};
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
window.addEventListener("resize", () => {
  if (landscape && !$("panel-landscape").hidden) {
    const w = $("landscape-3d").clientWidth;
    if (w) {
      landscape.camera.aspect = w / 420;
      landscape.camera.updateProjectionMatrix();
      landscape.renderer.setSize(w, 420);
    }
  }
  if (structure3d && !$("panel-structure").hidden) {
    const w = $("structure-3d").clientWidth;
    if (w) {
      structure3d.camera.aspect = w / 420;
      structure3d.camera.updateProjectionMatrix();
      structure3d.renderer.setSize(w, 420);
    }
  }
});

/* ---------- 3D fold schematic (SS3 trace, not atomic coordinates) ---------- */

let structure3d = null;
let structureLoadHook = false;

function disposeStructure() {
  if (!structure3d) return;
  cancelAnimationFrame(structure3d.raf);
  if (structure3d.renderer) structure3d.renderer.dispose();
  const el = $("structure-3d");
  if (el) el.innerHTML = "";
  structure3d = null;
}

/**
 * Pseudo-C-alpha layout from the SS3 string: helix residues spiral
 * (3.6 per turn), sheet residues pleat up/down, coil wanders gently.
 * Schematic only — it shows fold layout, not measured coordinates.
 */
function tracePoints(n, ss3) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const s = ss3[i] || "C";
    const x = (n === 1 ? 0 : i / (n - 1) - 0.5) * n * 0.3;
    let y, z;
    if (s === "H") {
      const a = i * (Math.PI * 2 / 3.6);
      y = Math.sin(a); z = Math.cos(a);
    } else if (s === "E") {
      y = (i % 2 === 0 ? 0.7 : -0.7); z = 0;
    } else {
      y = Math.sin(i * 0.7) * 0.35; z = Math.cos(i * 0.5) * 0.35;
    }
    pts.push(new THREE.Vector3(x, y, z));
  }
  return pts;
}

function renderStructure(rec) {
  const fallback = $("structure-fallback");
  disposeStructure();
  if (typeof THREE === "undefined") {
    if (!structureLoadHook) {
      structureLoadHook = true;
      window.addEventListener("load", () => {
        if (typeof THREE !== "undefined") {
          fallback.hidden = true;
          if (!$("panel-structure").hidden) renderStructure(currentRecord());
        } else fallback.hidden = false;
      }, { once: true });
    }
    fallback.hidden = false;
    return;
  }
  fallback.hidden = true;
  const el = $("structure-3d");
  if (!el || el.clientWidth === 0) return;
  const n = rec.sequence.length;
  if (n < 2) return;
  const ss3 = (rec.tracks && Array.isArray(rec.tracks.ss3)) ? rec.tracks.ss3 : [];
  const pts = tracePoints(n, ss3);

  const W = el.clientWidth, H = 420;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(5, 8, 6);
  scene.add(key);
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 200);
  const span = n * 0.3;
  camera.position.set(0, span * 0.18, span * 0.9 + 6);
  camera.lookAt(0, 0, 0);

  const group = new THREE.Group();
  const curve = new THREE.CatmullRomCurve3(pts);
  const tub = Math.max(24, n * 6), rad = 10;
  const geo = new THREE.TubeGeometry(curve, tub, 0.16, rad, false);
  // One colour per tubular ring, from the nearest residue's SS3 state.
  const count = geo.attributes.position.count;
  const col = new Float32Array(count * 3);
  const ringSize = rad + 1;
  for (let v = 0; v < count; v++) {
    const ring = Math.floor(v / ringSize);
    const ri = Math.min(n - 1, Math.round(ring / tub * (n - 1)));
    const c = ssColor(ss3[ri]);
    col[v * 3] = c.r; col[v * 3 + 1] = c.g; col[v * 3 + 2] = c.b;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  group.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.35, metalness: 0.1,
  })));
  // Termini markers: blue N, red C.
  const cap = new THREE.SphereGeometry(0.32, 16, 16);
  const nTerm = new THREE.Mesh(cap, new THREE.MeshBasicMaterial({ color: 0x22d3ee }));
  nTerm.position.copy(pts[0]);
  const cTerm = new THREE.Mesh(cap, new THREE.MeshBasicMaterial({ color: 0xf87171 }));
  cTerm.position.copy(pts[n - 1]);
  group.add(nTerm);
  group.add(cTerm);
  scene.add(group);

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let dragging = false, px = 0, py = 0;
  renderer.domElement.style.touchAction = "none";
  renderer.domElement.addEventListener("pointerdown", (e) => {
    dragging = true; px = e.clientX; py = e.clientY;
    renderer.domElement.setPointerCapture(e.pointerId);
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    group.rotation.y += (e.clientX - px) * 0.006;
    group.rotation.x = Math.max(-1.2, Math.min(1.2, group.rotation.x + (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY;
  });
  const stopDrag = () => { dragging = false; };
  renderer.domElement.addEventListener("pointerup", stopDrag);
  renderer.domElement.addEventListener("pointercancel", stopDrag);

  const clock = new THREE.Clock();
  const st = { renderer, raf: 0, group, camera };
  structure3d = st;
  (function tick() {
    st.raf = requestAnimationFrame(tick);
    if (!reduceMotion && !dragging) group.rotation.y += clock.getDelta() * 0.22;
    else clock.getDelta();
    renderer.render(scene, camera);
  })();
}
document.querySelector(".tabs").onkeydown = (e) => {
  const i = TABS.findIndex((t) => $(`tab-${t}`) === document.activeElement);
  if (i < 0) return;
  let j = null;
  if (e.key === "ArrowRight") j = (i + 1) % TABS.length;
  if (e.key === "ArrowLeft") j = (i - 1 + TABS.length) % TABS.length;
  if (j !== null) { e.preventDefault(); activateTab(TABS[j], true); }
};
$("ss-strip-toggle").onchange = () => {
  const rec = currentRecord();
  if (rec) renderSsStrip(rec);
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
