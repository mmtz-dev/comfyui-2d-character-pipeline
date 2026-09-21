"""The pages the API serves until `web/dist/` (the web pixel editor) exists.

Two views in one self-contained HTML file - no CDN, no build step, no
framework - driven entirely by the JSON endpoints, so they are also the
smallest possible client of the API:

    /                    every job with its stages, and the recent runs
    /jobs/<job>/review   the click-to-pick sheet that used to live on :8710

Same look as the old review page on purpose: dark ground because the sprites
are judged against a near-black board, `image-rendering: pixelated` because a
browser's default smoothing makes a 64px sprite look like a decision it is
not. A click PUTs the whole picks list; "Request regenerate" only flags
job.json - it never starts a render (anim.stages.review's docstring).
"""

from __future__ import annotations

import html

STYLE = """
  :root { color-scheme: dark; }
  body { margin:0; background:#15171c; color:#d7dae0;
         font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; background:#1d2027; border-bottom:1px solid #2c3038;
           padding:12px 16px; display:flex; gap:12px; align-items:center;
           flex-wrap:wrap; z-index:2; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  h2 { font-size:14px; margin:24px 0 8px; font-weight:600; color:#a9b0bc; }
  a { color:#f0c070; text-decoration:none; }
  a:hover { text-decoration:underline; }
  .muted { color:#8b929e; }
  .tag { font-size:12px; padding:2px 8px; border-radius:999px;
         background:#2c3038; color:#a9b0bc; }
  .tag.warn { background:#4a3312; color:#f0c070; }
  .tag.ok { background:#1f3a2a; color:#8fd3a5; }
  .tag.bad { background:#4a1f1f; color:#f08080; }
  button.act { font:inherit; background:#2c3038; color:#d7dae0; border:1px solid #3a3f49;
               border-radius:6px; padding:6px 12px; cursor:pointer; }
  button.act:hover { background:#363b45; }
  section { padding:0 16px 24px; }
  table { border-collapse:collapse; width:100%; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid #2c3038;
           vertical-align:top; }
  th { color:#8b929e; font-weight:500; font-size:12px; }
  main.sheet { display:grid; gap:10px; padding:16px;
               grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); }
  .card { position:relative; padding:0; border:2px solid #2c3038; border-radius:6px;
          background:#0e1013; cursor:pointer; overflow:hidden; }
  .card img { display:block; width:100%; height:auto;
              image-rendering:pixelated; image-rendering:crisp-edges; }
  .card.picked { border-color:#f0c070; box-shadow:0 0 0 2px #f0c07044; }
  .card .n { position:absolute; left:4px; bottom:2px; font-size:11px;
             color:#8b929e; background:#000a; padding:0 4px; border-radius:3px; }
  .ph { position:absolute; left:4px; top:4px; font-size:10px; padding:1px 5px; border-radius:999px;
    background:#1f3a2a; color:#8fd3a5; }
  .badge { position:absolute; right:4px; top:4px; background:#f0c070; color:#15171c;
           font-size:11px; font-weight:700; border-radius:999px;
           width:18px; height:18px; line-height:18px; text-align:center; }
  footer { padding:0 16px 24px; }
  pre.log { background:#0e1013; border:1px solid #2c3038; border-radius:6px;
            padding:8px; max-height:240px; overflow:auto; font-size:12px; }
  @media (max-width:420px) { main.sheet { grid-template-columns:repeat(auto-fill,minmax(90px,1fr)); } }
"""

INDEX_SCRIPT = """
async function getJSON(path) { const r = await fetch(path); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
async function loadJobs() {
  const jobs = await getJSON('/jobs');
  const rows = jobs.map(j => `<tr>
    <td><a href="/jobs/${esc(j.name)}/review">${esc(j.name)}</a></td>
    <td>${esc(j.slug)} / ${esc(j.action)}</td><td>${esc(j.direction)}</td>
    <td>${j.stages.map(s => `<span class="tag">${esc(s)}</span>`).join(' ')}</td>
    <td class="muted">${esc(j.updated)}</td></tr>`).join('');
  document.getElementById('jobs').innerHTML = rows ||
    '<tr><td colspan="5" class="muted">no jobs in work/ yet - run a stage from the CLI or POST /jobs</td></tr>';
}
async function loadRuns() {
  const runs = await getJSON('/runs');
  const tag = s => ({done:'ok', failed:'bad', lost:'bad', running:'warn'}[s] || '');
  const rows = runs.slice(0, 20).map(r => `<tr>
    <td><a href="/runs/${esc(r.id)}">${esc(r.id)}</a></td>
    <td>${esc(r.stage)} ${r.gpu ? '<span class="tag warn">GPU</span>' : ''}</td>
    <td><span class="tag ${tag(r.status)}">${esc(r.status)}</span>
        ${r.percent !== null ? esc(r.percent) + '%' : ''}</td>
    <td class="muted">${esc(r.argv.join(' '))}</td></tr>`).join('');
  document.getElementById('runs').innerHTML = rows ||
    '<tr><td colspan="4" class="muted">nothing has run through the API yet</td></tr>';
}
loadJobs().catch(e => alert(e)); loadRuns().catch(e => alert(e));
setInterval(() => loadRuns().catch(() => {}), 5000);
"""

REVIEW_SCRIPT = """
const JOB = document.body.dataset.job;
let state = null;
async function getJSON(path, init) {
  const r = await fetch(path, init);
  if (!r.ok) { const t = await r.text(); throw new Error(t); }
  return r.json();
}
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
// The keyframe phase feature's header tags: the plan in one line and, for a
// loop, whether it closes - the picker's verdict, never the page's.
function phaseTags(block) {
  if (!block) return '';
  const plan = block.plan || {}, frames = plan.frames || {};
  const shape = Object.entries(frames).map(([k, v]) => `${k} ${v}`).join(' / ');
  let out = `<span class="tag">phases: ${esc(plan.action)} ${esc(plan.budget)} \u00b7 ${esc(shape)}</span>`;
  const c = block.closure;
  if (c) {
    const [cls, word] = c.ok ? ['ok', 'loop closes'] : c.ok === false ? ['bad', 'loop does not close']
                        : ['warn', 'loop closure unknown'];
    const detail = c.distance != null ? ` (${c.metric} ${c.distance} vs ${c.tolerance})` : '';
    out += `<span class="tag ${cls}">${word}${esc(detail)}</span>`;
  }
  return out;
}
function render() {
  document.getElementById('title').textContent = state.job;
  document.getElementById('meta').textContent =
    `${state.slug} / ${state.action} \\u00b7 ${state.frames.length} frames @ ${state.fps} fps`;
  document.getElementById('edited').outerHTML = state.edited
    ? '<span id="edited" class="tag warn">hand-edited</span>' : '<span id="edited" class="tag">auto</span>';
  const flag = document.getElementById('flag');
  flag.textContent = 'regenerate requested';
  flag.style.display = state.rerender_requested ? '' : 'none';
  document.getElementById('picks').textContent = 'picks: ' + (state.picks.join(', ') || '(none)');
  document.getElementById('phases').innerHTML = phaseTags(state.phases);
  document.getElementById('sheet').innerHTML = state.frames.map(f => `
    <button class="card${f.picked ? ' picked' : ''}" data-index="${f.index}" onclick="toggle(${f.index})">
      ${f.order !== null ? `<span class="badge">${f.order + 1}</span>` : ''}
      ${f.phase ? `<span class="ph">${esc(f.phase)}</span>` : ''}
      <img src="/jobs/${esc(JOB)}/frames/frames/${f.index}.png" alt="frame ${f.index}" loading="lazy">
      <span class="n">${f.index}</span></button>`).join('');
}
async function load() { state = await getJSON(`/jobs/${JOB}/picks`); render(); }
async function toggle(i) {
  const picks = state.picks.includes(i) ? state.picks.filter(p => p !== i) : [...state.picks, i];
  state = await getJSON(`/jobs/${JOB}/picks`, {method: 'PUT',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify({picks})});
  render();
}
async function regen() {
  state = await getJSON(`/jobs/${JOB}/regenerate`, {method: 'POST'}); render();
}
load().catch(e => { document.getElementById('sheet').innerHTML = '<p class="muted">' + esc(e.message) + '</p>'; });
"""


def index_page() -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>anim - jobs</title>
<style>{STYLE}</style></head><body>
<header>
  <h1>anim pipeline</h1>
  <span class="muted">work/ as the API sees it &middot; <a href="/docs">API docs</a></span>
</header>
<section>
  <h2>Jobs</h2>
  <table><thead><tr><th>job</th><th>unit / action</th><th>dir</th><th>stages done</th><th>updated</th></tr></thead>
  <tbody id="jobs"><tr><td colspan="5" class="muted">loading&hellip;</td></tr></tbody></table>
  <h2>Runs</h2>
  <table><thead><tr><th>run</th><th>stage</th><th>status</th><th>argv</th></tr></thead>
  <tbody id="runs"><tr><td colspan="4" class="muted">loading&hellip;</td></tr></tbody></table>
</section>
<footer class="muted">Click a job for its click-to-pick sheet. A GPU stage runs only
from an explicit <code>POST /jobs</code>, one at a time; nothing on this page starts one.</footer>
<script>{INDEX_SCRIPT}</script></body></html>
"""


def review_page(job_name: str) -> str:
    safe = html.escape(job_name, quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>anim review - {safe}</title>
<style>{STYLE}</style></head><body data-job="{safe}">
<header>
  <a href="/" class="muted">&larr; jobs</a>
  <h1 id="title">{safe}</h1>
  <span class="muted" id="meta"></span>
  <span id="edited" class="tag">auto</span>
  <span id="flag" class="tag warn" style="display:none"></span>
  <span class="muted" id="picks"></span>
  <span id="phases"></span>
  <button class="act" onclick="regen()">Request regenerate</button>
</header>
<main class="sheet" id="sheet"></main>
<footer class="muted">Click a frame to keep or drop it. Saved straight to
picks.json. "Request regenerate" only flags job.json - it never starts a render.
</footer>
<script>{REVIEW_SCRIPT}</script></body></html>
"""
