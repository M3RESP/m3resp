// ================= 4 · Results: side panel, summary, values, plots, files =================
// The side panel chooses one part at a time: 1 Summary, 2 Values (every
// modality or one), 3 Plots (every modality or one) and 4 Files. Back / Next
// walk the same order.
// A click on a value opens its details: what it is, the step and settings
// that made it, the file that holds it and, for values per breath, one bar
// per breath. Breaths that failed a quality check of 3 · Check are yellow,
// and the row shows how many there are.

const RES_ITEM_WORD = {breath:'breath', occlusion:'occlusion'};
const resValuesOf = mod=> RES_VALUES.filter(v=> v.mod===mod);
const resPlotsOf = mod=> RES_PLOTS.filter(p=> p.mod===mod);
const resModOf = id=> RES_MODS.find(m=> m.id===id);
const resNum = (x, digits)=> (+x).toFixed(digits).replace('-', '−');
const RES_ROWS_SHOWN = 6;            // breath rows in a table before "Show all"

function resValueText(v){
  if(v.show) return v.show;
  if(v.scope==='occlusion') return v.items.map(x=> resNum(x.value, v.digits)).join(' · ');
  return resNum(v.value, v.digits);
}
const resUnit = v=> v.show ? '' : v.unit;
function resStatText(v){
  if(v.scope==='run') return 'one value for the run';
  if(v.scope==='occlusion') return `one value per occlusion (${v.items.length})`;
  return `${v.stat} of ${v.items.length} ${resModOf(v.mod).name} breaths`;
}
const resLimit = v=> v.limit===undefined ? '' : `${v.dir==='max' ? '≤' : '≥'} ${resNum(v.limit, v.limit % 1 ? 1 : 0)}${v.unit && v.unit!=='%' ? ' '+v.unit : v.unit}`;

// the breaths or occlusions of a value that failed a quality check of the
// same modality: {n, total, failedBy: id -> [checks], checks}
function resFlags(v){
  if(!v.items) return null;
  const checks = QC_CHECKS.filter(c=> c.mod===v.mod && c.scope!=='run');
  const ids = new Set(v.items.map(x=> x.id)), failedBy = {};
  checks.forEach(c=> c.items.forEach(x=>{
    if(!x.passed && ids.has(x.id)) (failedBy[x.id] = failedBy[x.id] || []).push(c);
  }));
  const n = Object.keys(failedBy).length;
  if(!n) return null;
  return {n, total:v.items.length, failedBy, checks: checks.filter(c=> c.items.some(x=> !x.passed && ids.has(x.id)))};
}
function resFlagText(v, f){
  const w = RES_ITEM_WORD[v.scope];
  return `${f.n} of ${f.total} ${w}${f.total===1?'':'s'} failed a quality check`;
}

// one small bar per breath; failed breaths in yellow
function resMiniBars(v, w=96, h=20){
  const vals = v.items.map(x=> x.value), f = resFlags(v);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = (hi-lo) || 1;
  const bw = w/vals.length, zero = h - (0-lo)/span*h;
  const bars = v.items.map((x, i)=>{
    const top = h - (Math.max(x.value, 0)-lo)/span*h, bot = h - (Math.min(x.value, 0)-lo)/span*h;
    const cls = f && f.failedBy[x.id] ? 'flag' : '';
    return `<rect class="${cls}" x="${(i*bw+bw*0.15).toFixed(1)}" y="${top.toFixed(1)}" width="${(bw*0.7).toFixed(1)}" height="${Math.max(1, bot-top).toFixed(1)}" rx="1"/>`;
  }).join('');
  return `<svg class="res-mini mod-${v.mod}" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${bars}<line x1="0" x2="${w}" y1="${zero}" y2="${zero}"/></svg>`;
}

// ---- side panel ----
function renderResSidePanel(){
  document.getElementById('resValuesCount').textContent = RES_VALUES.length;
  document.getElementById('resPlotsCount').textContent = RES_PLOTS.length;
  document.getElementById('resFilesCount').textContent = RES_FILES.length;
  const row = (m, target, sub)=> `<div class="mod-list-row" data-target="${target}" title="Show the ${m.name} ${sub.split(' ')[1]}">
      <span class="swatch ${m.id}"></span>
      <div class="info"><div class="nm">${m.name}</div><div class="file">${sub}</div></div>
    </div>`;
  document.getElementById('resValModList').innerHTML = RES_MODS.map(m=>{
    const n = resValuesOf(m.id).length;
    return row(m, m.valCard, `${n} value${n===1?'':'s'}`);
  }).join('');
  document.getElementById('resPlotModList').innerHTML = RES_MODS.filter(m=> m.plotCard).map(m=>{
    const n = resPlotsOf(m.id).length;
    return row(m, m.plotCard, `${n} plot${n===1?'':'s'}`);
  }).join('');
}

// ---- 1 Summary: one card per modality with its main values ----
const resSummaryEl = document.getElementById('resSummary');
function renderResSummary(){
  resSummaryEl.innerHTML = RES_MODS.map(m=>{
    const vals = resValuesOf(m.id).filter(v=> v.top);
    let foot;
    if(m.id==='session'){
      foot = `<span>Offsets set in 1 · Prepare</span><button class="tb-btn small quiet" data-prep="cardSync">Show the alignment</button>`;
    } else {
      const cs = qcChecksOf(m.id), n = qcCounts(cs);
      const flagged = vals.map(v=> resFlags(v)).filter(Boolean).sort((a, b)=> b.n-a.n)[0];
      const txt = !cs.length ? 'No quality checks in this run' : n.review ? `${pl(n.review, 'check')} to review` : 'All checks passed';
      foot = `<span class="st ${qcModState(m.id)}"></span><span>${txt}${flagged ? ` · <b>${flagged.n} of ${flagged.total}</b> ${m.id==='vent'?'occlusions':'breaths'} failed a check` : ''}</span>`
        + (cs.length ? `<button class="tb-btn small quiet" data-check-mod="${m.id}">Open in 3 · Check</button>` : '');
    }
    return `<div class="card mod-card mod-${m.id} res-sum-card">
      <div class="card-head">
        <div class="modal-info"><div class="name"><span class="mod-name">${m.name}</span></div><div class="meta">${m.meta}</div></div>
        <button class="tb-btn small quiet" data-part="${m.valCard}">All ${resValuesOf(m.id).length} values</button>
      </div>
      <div class="card-body">
        <div class="res-sum-vals">${vals.map(v=>`
          <button type="button" class="res-sum-val" data-open="${v.id}" title="Open the details of ${v.name}">
            <span class="k">${v.name}</span>
            <span class="v${v.show ? ' txt' : ''}">${resValueText(v)}${resUnit(v) ? `<small>${resUnit(v)}</small>` : ''}</span>
            ${v.items ? `<span class="b">${resMiniBars(v, 120, 22)}<span>${v.scope==='occlusion' ? 'per occlusion' : 'mean · per breath'}</span></span>` : ''}
          </button>`).join('')}</div>
        <div class="res-sum-foot">${foot}</div>
      </div>
    </div>`;
  }).join('');
}
resSummaryEl.addEventListener('click', e=>{
  const open = e.target.closest('[data-open]');
  if(open){ resOpenValue(open.dataset.open); return; }
  const part = e.target.closest('[data-part]');
  if(part){ showResPart(part.dataset.part); return; }
  const chk = e.target.closest('[data-check-mod]');
  if(chk){ resOpenCheckMod(chk.dataset.checkMod); return; }
  const prep = e.target.closest('[data-prep]');
  if(prep){ goToView('prepare'); showSection(prep.dataset.prep); }
});

// ---- 2 Values: search, modality chips, one card per modality ----
const resValueCards = document.getElementById('resValueCards');
const resSearch = document.getElementById('resSearch');
const resChips = document.getElementById('resChips');
const resOpen = new Set();             // values with open details
const resPick = {};                    // value id -> breath id picked on the bars
const resModOn = {eit:true, emg:true, vent:true, session:true};
const resMatches = v=>{
  const q = resSearch.value.trim().toLowerCase();
  return !q || `${v.name} ${v.code} ${v.step}`.toLowerCase().includes(q);
};

function renderResChips(){
  resChips.innerHTML = RES_MODS.map(m=> `<button type="button" class="res-chip mod-${m.id}${resModOn[m.id] ? ' on' : ''}" data-mod="${m.id}" aria-pressed="${resModOn[m.id]}"><span class="dot"></span>${m.name}</button>`).join('');
}
resChips.addEventListener('click', e=>{
  const b = e.target.closest('[data-mod]'); if(!b) return;
  resModOn[b.dataset.mod] = !resModOn[b.dataset.mod];
  renderResChips(); renderResValueCards(); showResPart(currentResPart, {keepScroll:true});
});
resSearch.addEventListener('input', ()=>{ renderResValueCards(); showResPart(currentResPart, {keepScroll:true}); });

function renderResValueCards(){
  resValueCards.innerHTML = '';
  let shown = 0;
  RES_MODS.forEach(m=>{
    const all = resValuesOf(m.id), vals = all.filter(resMatches);
    const card = document.createElement('div');
    card.className = `card mod-card mod-${m.id} res-val-card`;
    card.id = m.valCard;
    card.dataset.mod = m.id;
    card.dataset.hits = vals.length;
    card.innerHTML = `
      <div class="card-head">
        <div class="mod-head-left"><div class="modal-info">
          <div class="name"><span class="mod-name">${m.name}</span><span class="mod-device">values</span></div>
          <div class="meta">${m.meta}</div>
        </div></div>
        <span class="res-head-num">${vals.length===all.length ? pl(all.length, 'value') : `${vals.length} of ${all.length} values`}</span>
      </div>
      <div class="card-body"><div class="qc-list res-list" role="list"></div></div>`;
    const list = card.querySelector('.res-list');
    if(!vals.length) list.outerHTML = `<div class="windows-empty">No ${m.name} value matches “${resSearch.value.trim()}”.</div>`;
    vals.forEach(v=> list.appendChild(resRow(v)));
    if(resModOn[m.id]) shown += vals.length;
    resValueCards.appendChild(card);
  });
  document.getElementById('resCount').textContent = `${shown} of ${RES_VALUES.length} values`;
}

function resRow(v){
  const f = resFlags(v), open = resOpen.has(v.id);
  const row = document.createElement('div');
  row.className = 'qc-check res-value' + (open ? ' open' : '');
  row.id = v.id;
  row.setAttribute('role', 'listitem');
  const word = RES_ITEM_WORD[v.scope];
  row.innerHTML = `
    <button type="button" class="qc-row res-row" aria-expanded="${open}" aria-controls="${v.id}-d">
      <span class="qc-name"><span class="qc-plain">${v.name}</span><code>${v.code}</code></span>
      <span class="res-breaths">${v.items ? `${resMiniBars(v)}<span>${v.items.length} ${word}s</span>` : '<span class="res-one">one value</span>'}</span>
      <span class="res-val">${resValueText(v)}${resUnit(v) ? `<small>${resUnit(v)}</small>` : ''}${v.stat ? `<em>${v.stat}</em>` : ''}</span>
      <span class="res-flag-cell">${f ? `<span class="res-flag" title="${resFlagText(v, f)}">${f.n} / ${f.total}</span>` : ''}</span>
      <span class="qc-chev">${prepIcon('chevron-down')}</span>
    </button>
    <div class="qc-detail" id="${v.id}-d"><div class="qc-detail-inner">${resDetail(v, f)}</div></div>`;
  row.querySelector('.res-row').addEventListener('click', ()=>{
    const isOpen = row.classList.toggle('open');
    row.querySelector('.res-row').setAttribute('aria-expanded', String(isOpen));
    if(isOpen) resOpen.add(v.id); else resOpen.delete(v.id);
  });
  wireResDetail(row, v, f);
  return row;
}

function resDetail(v, f){
  const word = RES_ITEM_WORD[v.scope];
  const facts = [
    ['Value', resStatText(v)],
    ['Produced by', `<code>${v.step}</code> · ${v.stepMs.toFixed(1)} ms`],
    ['Settings', v.settings.length ? v.settings.map(([k, x])=> `<code>${k}</code> = ${x}`).join('<br>') : 'the defaults of the step'],
    ['Saved to', `<code>${v.file}</code><br><span class="res-key">${v.key}</span>`],
  ];
  let html = `
    <div class="qc-what"><p>${v.what}</p></div>
    <dl class="qc-facts">${facts.map(([k, x])=> `<div><dt>${k}</dt><dd>${x}</dd></div>`).join('')}</dl>`;
  if(v.mask) html += `<div class="res-mask-one">${resMaskSvg(v.mask, 'var(--eit)')}<span>${v.show}: the blue pixels are in the lungs.</span></div>`;
  if(f){
    html += `<div class="res-flag-box">
      <b>${resFlagText(v, f)}.</b> The value includes ${f.n===1 ? 'it' : 'them'}.
      <div class="res-flag-checks">${f.checks.map(c=>{
        const k = v.items.filter(x=> (f.failedBy[x.id] || []).includes(c)).length;
        return `<button type="button" class="res-flag-check" data-check="${c.code}" title="Open this check in 3 · Check"><span class="qc-plain">${c.name}</span><span class="mono">${k} ${word}${k===1?'':'s'}</span></button>`;
      }).join('')}</div>
    </div>`;
  }
  (v.checks || []).filter(code=> !f || !f.checks.some(c=> c.code===code)).forEach(code=>{
    const c = QC_CHECKS.find(x=> x.code===code), st = qcStatus(c);
    html += `<div class="res-check-line"><span class="qc-dot ${st}"></span>Quality check <b>${c.name}</b>: ${QC_STATUS_TEXT[st].toLowerCase()}<button class="tb-btn small quiet" data-check="${c.code}">Open in 3 · Check</button></div>`;
  });
  if(v.items){
    const vals = v.items.map(x=> x.value);
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const showLimit = v.limit!==undefined && v.limit>=lo && v.limit<=hi*1.25;
    if(showLimit) hi = Math.max(hi, v.limit);
    const span = (hi-lo) || 1, pct = x=> (x-lo)/span*100;
    html += `
      <div class="qc-strip-head"><span class="ct-label">${word}s</span><span class="qc-strip-sum">${v.items.length} ${word}s${v.limit!==undefined ? ` · limit ${resLimit(v)}` : ''}${v.stat ? ` · ${v.stat} ${resValueText(v)} ${v.unit}` : ''}</span></div>
      <div class="res-bars mod-${v.mod}" role="group" aria-label="One bar per ${word}">
        ${showLimit ? `<span class="res-limit" style="bottom:${(16 + pct(v.limit)*0.8).toFixed(1)}px"><span>limit ${resLimit(v)}</span></span>` : ''}
        ${v.items.map(x=>{
          const b = pct(Math.min(x.value, 0)), h = Math.max(1.5, Math.abs(x.value)/span*100);
          const fl = f && f.failedBy[x.id];
          return `<button type="button" class="res-bar${fl ? ' flag' : ''}" data-item="${x.id}" title="${resItemTitle(v, x, f)}" aria-label="${resItemTitle(v, x, f)}"><span style="bottom:${b.toFixed(1)}%;height:${h.toFixed(1)}%"></span><i>${x.id.replace('Pocc ', 'P')}</i></button>`;
        }).join('')}
      </div>
      <div class="qc-pick" hidden></div>
      <table class="qc-fails res-table">
        <thead><tr><th>${word}</th><th>Time</th><th>${v.name}</th><th>Failed checks</th><th></th></tr></thead>
        <tbody>${v.items.map((x, k)=>`<tr data-item="${x.id}"${k>=RES_ROWS_SHOWN ? ' class="more"' : ''}>
          <td class="mono">${x.id}</td><td class="mono">${fmtRange(x.t0, x.t1)}</td>
          <td class="mono">${resNum(x.value, v.digits)}${v.unit ? ' '+v.unit : ''}</td>
          <td>${f && f.failedBy[x.id] ? f.failedBy[x.id].map(c=> `<span class="res-failed">${c.name}</span>`).join(' ') : '<span class="res-none">none</span>'}</td>
          <td><button class="tb-btn small quiet" data-show-item="${x.id}" title="Show this ${word} in 1 · Prepare">Show</button></td></tr>`).join('')}</tbody>
      </table>`
      + (v.items.length>RES_ROWS_SHOWN ? `<button class="tb-btn small quiet qc-more" data-more>Show all ${v.items.length} ${word}s</button>` : '');
  }
  html += `<div class="qc-actions">
    <button class="tb-btn small" data-find-step>Find step in 2 · Design</button>
    <button class="tb-btn small" data-show-prepare>Show in 1 · Prepare</button>
  </div>`;
  return html;
}
function resItemTitle(v, x, f){
  const fl = f && f.failedBy[x.id];
  return `${RES_ITEM_WORD[v.scope]} ${x.id} · ${fmtRange(x.t0, x.t1)} · ${resNum(x.value, v.digits)} ${v.unit}${fl ? ' · failed: '+fl.map(c=> c.name).join(', ') : ''}`;
}

function wireResDetail(row, v, f){
  const pick = row.querySelector('.qc-pick');
  const showPick = id=>{
    const x = v.items.find(i=> i.id===id); if(!x || !pick) return;
    resPick[v.id] = id;
    row.querySelectorAll('.res-bar').forEach(b=> b.classList.toggle('picked', b.dataset.item===id));
    const fl = f && f.failedBy[id];
    pick.hidden = false;
    pick.className = 'qc-pick ' + (fl ? 'flag' : 'ok');
    pick.innerHTML = `<b>${RES_ITEM_WORD[v.scope]} ${x.id}</b><span class="mono">${fmtRange(x.t0, x.t1)}</span>
      <span class="mono">${resNum(x.value, v.digits)} ${v.unit}</span>
      <span>${fl ? 'Failed: '+fl.map(c=> c.name).join(', ')+'.' : 'Passed every check.'}</span>
      <button class="tb-btn small quiet" data-show-item="${x.id}">Show in 1 · Prepare</button>`;
  };
  row.querySelectorAll('.res-bar').forEach(b=> b.addEventListener('click', ()=> showPick(b.dataset.item)));
  if(resPick[v.id]) showPick(resPick[v.id]);
  row.querySelector('.qc-detail').addEventListener('click', e=>{
    const chk = e.target.closest('[data-check]');
    if(chk){ resOpenCheck(chk.dataset.check); return; }
    const show = e.target.closest('[data-show-item]');
    if(show){ resShowInPrepare(v, v.items.find(i=> i.id===show.dataset.showItem)); return; }
    if(e.target.closest('[data-show-prepare]')){ resShowInPrepare(v, v.items && (v.items.find(x=> f && f.failedBy[x.id]) || v.items[0])); return; }
    if(e.target.closest('[data-find-step]')){ qcFindStep(v); return; }
    const more = e.target.closest('[data-more]');
    if(more){
      const all = row.querySelector('.res-table').classList.toggle('all');
      more.textContent = all ? `Show the first ${RES_ROWS_SHOWN}` : `Show all ${v.items.length} ${RES_ITEM_WORD[v.scope]}s`;
    }
  });
}

function resShowInPrepare(v, x){
  if(v.mod==='session'){ goToView('prepare'); showSection('cardSync'); return; }
  qcShowInPrepare(v, x);
}
// a value from the summary: its card, with the details open
function resOpenValue(id){
  const v = RES_VALUES.find(x=> x.id===id); if(!v) return;
  resOpen.add(id);
  resSearch.value = '';
  resModOn[v.mod] = true;
  renderResChips(); renderResValueCards();
  showResPart(resModOf(v.mod).valCard);
  document.getElementById(id).scrollIntoView({block:'nearest'});
}
// a check, or the checks of a modality, on 3 · Check
function resOpenCheck(code){
  const c = QC_CHECKS.find(x=> x.code===code); if(!c) return;
  goToView('check');
  qcOpen.add(code);
  qcFilter[c.mod] = 'all';
  renderCheckCards();
  showCheckPart(QC_MODS.find(m=> m.id===c.mod).card);
  document.getElementById('qc-d-'+code).closest('.qc-check').scrollIntoView({block:'center'});
}
function resOpenCheckMod(mod){
  goToView('check');
  showCheckPart(QC_MODS.find(m=> m.id===mod).card);
}

// ---- 3 Plots: a grid per modality; a click opens a plot at full width ----
const resPlotCards = document.getElementById('resPlotCards');
const resPlotOpen = new Set();
const resPlotState = {emgRange:'all', emgPeaks:true, emgWin:true, emgBase:true, bvValue:'percentage_under_baseline', imgLarge:{}};

function renderResPlotCards(){
  resPlotCards.innerHTML = '';
  RES_MODS.filter(m=> m.plotCard).forEach(m=>{
    const ps = resPlotsOf(m.id), saved = ps.filter(p=> p.kind==='img').length;
    const card = document.createElement('div');
    card.className = `card mod-card mod-${m.id} res-plot-card`;
    card.id = m.plotCard;
    card.innerHTML = `
      <div class="card-head">
        <div class="mod-head-left"><div class="modal-info">
          <div class="name"><span class="mod-name">${m.name}</span><span class="mod-device">plots</span></div>
          <div class="meta">${pl(ps.length, 'plot')}${saved ? ` · ${saved} saved by the run` : ''}</div>
        </div></div>
      </div>
      <div class="card-body"><div class="res-plot-grid"></div></div>`;
    const grid = card.querySelector('.res-plot-grid');
    ps.forEach(p=> grid.appendChild(resPlotTile(p)));
    resPlotCards.appendChild(card);
  });
}

function resPlotTag(p){
  return p.kind==='img' ? `<span class="res-tag saved">saved</span>` : `<span class="res-tag">drawn</span>`;
}
function resPlotTile(p){
  const tile = document.createElement('div');
  const open = resPlotOpen.has(p.id);
  tile.className = 'res-tile' + (open ? ' open' : '');
  tile.dataset.plot = p.id;
  if(!open){
    tile.innerHTML = `
      <button type="button" class="res-tile-btn" title="Open this plot">
        <span class="res-thumb">${resPlotBody(p, true)}</span>
        <span class="res-tile-text">
          <span class="nm">${p.name}</span>
          <span class="src"><code>${p.step}</code></span>
          <span class="tags">${resPlotTag(p)}<span class="file">${p.src}</span></span>
        </span>
      </button>`;
    tile.querySelector('.res-tile-btn').addEventListener('click', ()=>{ resPlotOpen.add(p.id); tile.replaceWith(resPlotTile(p)); });
    return tile;
  }
  tile.innerHTML = `
    <div class="res-view-head">
      <div class="res-view-title"><span class="nm">${p.name}</span><span class="tags">${resPlotTag(p)}<code>${p.step}</code><span class="file">${p.src}</span></span></div>
      <button class="tb-btn small" data-find-step>Find step in 2 · Design</button>
      <button class="tb-btn small icon quiet" data-close title="Close this plot" aria-label="Close this plot">${prepIcon('close')}</button>
    </div>
    <div class="res-view-tools">${resPlotTools(p)}</div>
    <div class="res-view-body">${resPlotBody(p, false)}</div>`;
  tile.addEventListener('click', e=>{
    if(e.target.closest('[data-close]')){ resPlotOpen.delete(p.id); tile.replaceWith(resPlotTile(p)); return; }
    if(e.target.closest('[data-find-step]')){ qcFindStep(p); return; }
    const b = e.target.closest('[data-set]');
    if(b){
      const [k, x] = b.dataset.set.split(':');
      if(k==='imgLarge') resPlotState.imgLarge[p.id] = x==='1';
      else if(x==='toggle') resPlotState[k] = !resPlotState[k];
      else resPlotState[k] = x;
      tile.replaceWith(resPlotTile(p));
    }
  });
  return tile;
}

const resSeg = (key, cur, opts)=> `<div class="tb-seg small" role="group">${opts.map(([x, label])=> `<button data-set="${key}:${x}" class="${cur===x ? 'on' : ''}">${label}</button>`).join('')}</div>`;
const resToggle = (key, label)=> `<button class="res-toggle${resPlotState[key] ? ' on' : ''}" data-set="${key}:toggle" aria-pressed="${resPlotState[key]}"><span class="box">${prepIcon('check', 12)}</span>${label}</button>`;
function resPlotTools(p){
  if(p.kind==='img') return `<span class="ct-label">Size</span>${resSeg('imgLarge', resPlotState.imgLarge[p.id] ? '1' : '0', [['0', 'Fit'], ['1', 'Full width']])}`;
  if(p.kind==='emgBreaths') return `<span class="ct-label">Time</span>${resSeg('emgRange', resPlotState.emgRange, [['all', '0–60 s'], ['0', '0–20 s'], ['20', '20–40 s'], ['40', '40–60 s']])}
    <span class="tb-sep"></span>${resToggle('emgPeaks', 'Breath peaks')}${resToggle('emgWin', 'Onset to offset')}${resToggle('emgBase', 'Baseline')}`;
  if(p.kind==='breathValues') return `<span class="ct-label">Value</span>${resSeg('bvValue', resPlotState.bvValue, [['percentage_under_baseline', 'Time under baseline'], ['snr_pseudo', 'Signal-to-noise ratio']])}`;
  if(p.kind==='masks') return `<span class="res-tools-note">Each square is one of the 32 × 32 EIT pixels.</span>`;
  if(p.kind==='pocc') return `<span class="res-tools-note">The green band is where a value passes.</span>`;
  return '';
}
function resPlotBody(p, thumb){
  if(p.kind==='img'){
    const large = !thumb && resPlotState.imgLarge[p.id];
    return `<img class="res-img${large ? ' large' : ''}" src="data:image/png;base64,${p.img}" alt="${p.name}">`;
  }
  if(p.kind==='emgBreaths') return resEmgPlot(thumb);
  if(p.kind==='breathValues') return resBreathValuesPlot(thumb);
  if(p.kind==='pocc') return resPoccPlot(thumb);
  if(p.kind==='masks') return `<div class="res-masks">${RES_VALUES.filter(v=> v.mask).map(v=>
    `<figure>${resMaskSvg(v.mask, 'var(--eit)')}${thumb ? '' : `<figcaption><b>${v.name}</b><span>${v.show}</span></figcaption>`}</figure>`).join('')}</div>`;
  return '';
}

// axes: x ticks in seconds and y ticks, inside a W × H drawing with margins
function resAxes(W, H, m, x0, x1, y0, y1, xStep, yTicks, yUnit, thumb){
  if(thumb) return '';
  const X = t=> m.l + (t-x0)/(x1-x0)*(W-m.l-m.r), Y = y=> H-m.b - (y-y0)/(y1-y0)*(H-m.t-m.b);
  let s = '';
  for(let t=Math.ceil(x0/xStep)*xStep; t<=x1+1e-9; t+=xStep)
    s += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H-m.b}"/><text class="tick" x="${X(t)}" y="${H-m.b+15}" text-anchor="middle">${+t.toFixed(1)}</text>`;
  yTicks.forEach(y=>{ s += `<line class="grid" x1="${m.l}" x2="${W-m.r}" y1="${Y(y)}" y2="${Y(y)}"/><text class="tick" x="${m.l-6}" y="${Y(y)+3.5}" text-anchor="end">${y}</text>`; });
  s += `<text class="unit" x="${m.l-6}" y="${m.t-6}" text-anchor="end">${yUnit}</text>`;
  return s;
}

// EMGdi envelope (RMS, ECG removed) drawn from the 19 breaths of the run:
// one burst per breath peak, the baseline under it
function resEmgEnvAt(t){
  let v = 5.2 + 0.9*Math.sin(t*1.7) + 0.7*Math.sin(t*4.3+1);
  RUN_EMG_PEAKS.forEach((p, i)=>{ const w = qcEmgBreath(i); const s = (w.t1-w.t0)/2.6; v += (17 + 13*qcRand(i+5)) * Math.exp(-Math.pow((t-p)/s, 2)); });
  return v;
}
const resEmgBaseAt = t=> 8.2 + 0.8*Math.sin(t/7);
function resEmgPlot(thumb){
  const W = 1000, H = thumb ? 150 : 250, m = thumb ? {l:4, r:4, t:8, b:8} : {l:46, r:14, t:22, b:28};
  const r = resPlotState.emgRange, x0 = thumb || r==='all' ? 0 : +r, x1 = thumb || r==='all' ? REC_SECONDS : +r+20;
  const y0 = 0, y1 = 42;
  const X = t=> m.l + (t-x0)/(x1-x0)*(W-m.l-m.r), Y = y=> H-m.b - (y-y0)/(y1-y0)*(H-m.t-m.b);
  const n = 900;
  let env = '', base = '';
  for(let i=0;i<=n;i++){ const t = x0 + i/n*(x1-x0); env += `${X(t).toFixed(1)},${Y(resEmgEnvAt(t)).toFixed(1)} `; base += `${X(t).toFixed(1)},${Y(resEmgBaseAt(t)).toFixed(1)} `; }
  const aub = qcItemsOf('percentage_under_baseline');
  let wins = '', peaks = '';
  RUN_EMG_PEAKS.forEach((p, i)=>{
    if(p<x0-1 || p>x1+1) return;
    const b = qcEmgBreath(i), flag = !aub[i].passed;
    if(thumb || resPlotState.emgWin) wins += `<rect class="win${flag ? ' flag' : ''}" x="${X(Math.max(x0, b.t0))}" y="${m.t}" width="${Math.max(0, X(Math.min(x1, b.t1))-X(Math.max(x0, b.t0)))}" height="${H-m.t-m.b}"><title>breath ${i+1} · ${fmtRange(b.t0, b.t1)}</title></rect>`;
    if(!thumb && resPlotState.emgPeaks) peaks += `<line class="peak" x1="${X(p)}" x2="${X(p)}" y1="${Y(resEmgEnvAt(p))-4}" y2="${H-m.b}"/><circle class="peak-dot" cx="${X(p)}" cy="${Y(resEmgEnvAt(p))}" r="3"/>`
      + (x1-x0<=20 ? `<text class="peak-lab" x="${X(p)}" y="${Y(resEmgEnvAt(p))-9}" text-anchor="middle">${i+1}</text>` : '');
  });
  return `<svg class="res-plot mod-emg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" width="100%" height="${H}" role="img" aria-label="EMGdi envelope with 19 breaths">
    ${resAxes(W, H, m, x0, x1, y0, y1, x1-x0>20 ? 10 : 2, [0, 10, 20, 30, 40], 'µV', thumb)}
    ${wins}
    ${!thumb && resPlotState.emgBase ? `<polyline class="base" points="${base}"/>` : ''}
    <polyline class="sig" points="${env}"/>
    ${peaks}
    ${thumb ? '' : `<text class="tick" x="${W-m.r}" y="${H-4}" text-anchor="end">time (s)</text>`}
  </svg>`
  + (thumb ? '' : `<div class="res-plot-key"><span><i class="k-sig"></i>EMGdi envelope (RMS, ECG removed)</span><span><i class="k-base"></i>baseline</span><span><i class="k-win"></i>onset to offset</span><span><i class="k-flag"></i>failed: time under baseline</span></div>`);
}

// one bar per EMG breath, with the limit of the check when it has one
function resBreathValuesPlot(thumb){
  const v = RES_VALUES.find(x=> x.code===(thumb ? 'percentage_under_baseline' : resPlotState.bvValue));
  const lim = v.code==='percentage_under_baseline' ? 40 : null;
  const W = 1000, H = thumb ? 150 : 240, m = thumb ? {l:4, r:4, t:8, b:8} : {l:46, r:14, t:22, b:30};
  const vals = v.items.map(x=> x.value), y1 = Math.max(...vals, lim || 0)*1.15;
  const Y = y=> H-m.b - y/y1*(H-m.t-m.b), bw = (W-m.l-m.r)/vals.length;
  const step = y1>20 ? 10 : y1>2 ? 0.5 : 0.25, ticks = [];
  for(let y=0; y<=y1; y+=step) ticks.push(+y.toFixed(2));
  const f = resFlags(v);
  let s = '';
  if(!thumb) ticks.forEach(y=>{ s += `<line class="grid" x1="${m.l}" x2="${W-m.r}" y1="${Y(y)}" y2="${Y(y)}"/><text class="tick" x="${m.l-6}" y="${Y(y)+3.5}" text-anchor="end">${y}</text>`; });
  v.items.forEach((x, i)=>{
    const fail = lim!==null && x.value>lim;
    s += `<rect class="bar${fail ? ' fail' : ''}" x="${m.l+i*bw+bw*0.18}" y="${Y(x.value)}" width="${bw*0.64}" height="${H-m.b-Y(x.value)}" rx="2"><title>breath ${x.id} · ${resNum(x.value, v.digits)} ${v.unit}${f && f.failedBy[x.id] ? ' · failed: '+f.failedBy[x.id].map(c=> c.name).join(', ') : ''}</title></rect>`;
    if(!thumb) s += `<text class="tick" x="${m.l+i*bw+bw/2}" y="${H-m.b+15}" text-anchor="middle">${x.id}</text>`;
  });
  if(lim!==null) s += `<line class="limit" x1="${m.l}" x2="${W-m.r}" y1="${Y(lim)}" y2="${Y(lim)}"/>` + (thumb ? '' : `<text class="limit-lab" x="${W-m.r}" y="${Y(lim)-5}" text-anchor="end">limit ≤ 40 %</text>`);
  const mean = v.value;
  s += `<line class="mean" x1="${m.l}" x2="${W-m.r}" y1="${Y(mean)}" y2="${Y(mean)}"/>` + (thumb ? '' : `<text class="mean-lab" x="${m.l+4}" y="${Y(mean)-5}">mean ${resNum(mean, v.digits)} ${v.unit}</text>`);
  if(!thumb) s += `<text class="unit" x="${m.l-6}" y="${m.t-6}" text-anchor="end">${v.unit || 'ratio'}</text><text class="tick" x="${W-m.r}" y="${H-2}" text-anchor="end">breath</text>`;
  return `<svg class="res-plot mod-emg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" width="100%" height="${H}" role="img" aria-label="${v.name} per breath">${s}</svg>`;
}

// the two Pocc criteria of the run, each on its own scale with its pass band
function resPoccPlot(thumb){
  const rows = [
    {v:RES_VALUES.find(x=> x.code==='pocc_quality_dp_up_10'), lo:-0.05, hi:0.05, ticks:[-0.04, -0.02, 0, 0.02, 0.04]},
    {v:RES_VALUES.find(x=> x.code==='pocc_quality_dp_up_90'), lo:-0.2, hi:2.4, ticks:[0, 0.5, 1, 1.5, 2]},
  ];
  const W = 1000, rowH = thumb ? 70 : 92, H = rows.length*rowH + (thumb ? 10 : 16), l = thumb ? 10 : 210, r = thumb ? 10 : 30;
  let s = '';
  rows.forEach((row, k)=>{
    const {v} = row, y = k*rowH + rowH/2 + 6, X = x=> l + (x-row.lo)/(row.hi-row.lo)*(W-l-r);
    const passFrom = v.dir==='min' ? v.limit : row.lo, passTo = v.dir==='min' ? row.hi : v.limit;
    s += `<rect class="pass" x="${X(passFrom)}" y="${y-16}" width="${X(passTo)-X(passFrom)}" height="32" rx="4"/>`;
    s += `<line class="axis" x1="${l}" x2="${W-r}" y1="${y}" y2="${y}"/>`;
    s += `<line class="limit" x1="${X(v.limit)}" x2="${X(v.limit)}" y1="${y-22}" y2="${y+22}"/>`;
    if(!thumb){
      s += `<text class="row-lab" x="0" y="${y-4}">${v.name}</text><text class="tick" x="0" y="${y+12}" text-anchor="start">passes when ${resLimit(v)}</text>`;
      row.ticks.forEach(t=>{ s += `<text class="tick" x="${X(t)}" y="${y+36}" text-anchor="middle">${resNum(t, 2).replace(/\.?0+$/, '') || '0'}</text>`; });
    }
    v.items.forEach((x, i)=>{
      const ok = v.dir==='min' ? x.value>=v.limit : x.value<=v.limit;
      const dy = i ? 9 : -9;
      s += `<circle class="pt ${ok ? 'ok' : 'bad'}" cx="${X(x.value)}" cy="${y+dy}" r="${thumb ? 6 : 7}"><title>${x.id} · ${resNum(x.value, v.digits)} ${v.unit} · ${ok ? 'passes' : 'fails'}</title></circle>`;
      if(!thumb) s += `<text class="pt-lab" x="${X(x.value)+12}" y="${y+dy+4}">${x.id}: ${resNum(x.value, v.digits)}</text>`;
    });
  });
  return `<svg class="res-plot mod-vent res-pocc" viewBox="0 0 ${W} ${H}" preserveAspectRatio="${thumb ? 'none' : 'xMinYMid meet'}" width="100%" height="${H}" role="img" aria-label="Pocc upslope criteria against the limits">${s}</svg>`;
}

// a lung mask: 32 × 32 EIT pixels, the px pixels closest to two lung shapes
const RES_MASKS = {};
function resMaskCells(px, seed){
  const key = px+'_'+seed;
  if(!RES_MASKS[key]){
    const score = [];
    for(let y=0;y<32;y++) for(let x=0;x<32;x++){
      const d = Math.min(Math.pow((x-9.6)/6.0, 2) + Math.pow((y-16)/9.2, 2), Math.pow((x-22.2)/6.6, 2) + Math.pow((y-15.5)/9.6, 2));
      score.push({i:y*32+x, d: d + 0.35*(qcRand(x*37+y*101+seed*7)-0.5)});
    }
    score.sort((a, b)=> a.d-b.d);
    RES_MASKS[key] = new Set(score.slice(0, px).map(s=> s.i));
  }
  return RES_MASKS[key];
}
function resMaskSvg(mask, color){
  const cells = resMaskCells(mask.px, mask.seed);
  let s = '';
  for(let i=0;i<1024;i++) s += `<rect x="${(i%32)*9}" y="${Math.floor(i/32)*9}" width="8" height="8" rx="1"${cells.has(i) ? ` class="in"` : ''}/>`;
  return `<svg class="res-mask" viewBox="0 0 287 287" style="--mask:${color}" role="img" aria-label="${mask.px} of 1024 pixels in the lungs">${s}</svg>`;
}

// ---- 4 Files ----
const resFilesState = document.getElementById('resFilesState');
function renderResFiles(){
  document.getElementById('resFolderPath').textContent = RES_OUT_DIR + '/';
  document.getElementById('resFilesHint').textContent = `${RES_FILES.length} files`;
  const groups = [...new Set(RES_FILES.map(f=> f.group))];
  document.getElementById('resFileList').innerHTML = groups.map(g=>`
    <div class="res-file-group">
      <div class="res-file-head"><span class="ct-label">${g}</span></div>
      ${RES_FILES.filter(f=> f.group===g).map(f=>`
        <div class="res-file">
          <code class="nm">${f.name}</code>
          <span class="what">${f.what}</span>
          <span class="rows">${f.rows ? f.rows + (/^\d+$/.test(f.rows) ? ' rows' : '') : ''}</span>
          <span class="size">${f.size}</span>
          <span class="go">${f.part ? `<button class="tb-btn small quiet" data-part="${f.part}">Show</button>` : f.check ? `<button class="tb-btn small quiet" data-check-page>Open in 3 · Check</button>` : ''}</span>
        </div>`).join('')}
    </div>`).join('');
}
document.getElementById('resFileList').addEventListener('click', e=>{
  const part = e.target.closest('[data-part]');
  if(part){ showResPart(part.dataset.part); return; }
  if(e.target.closest('[data-check-page]')){ goToView('check'); showCheckPart('checkAll'); }
});
document.getElementById('resCopyPath').addEventListener('click', ()=>{
  const path = RES_OUT_DIR + '/';
  const done = ok=> resFilesState.textContent = ok ? 'Folder path copied.' : 'Select the path above and copy it.';
  try{ navigator.clipboard.writeText(path).then(()=> done(true), ()=> done(false)); }catch(err){ done(false); }
});
document.getElementById('resOpenFolder').addEventListener('click', ()=>{
  resFilesState.textContent = 'The app opens this folder in the file manager.';
});
document.getElementById('resZip').addEventListener('click', ()=>{
  resFilesState.textContent = `The app saves the ${RES_FILES.length} files as 2026-06-10_153009_multimodal-full.zip.`;
});
document.getElementById('resNewRun').addEventListener('click', ()=> goToView('design'));

// ---- one part at a time ----
const resMainEl = document.getElementById('resultsMain');
const resSteps = document.getElementById('resSteps');
const resCrumb = document.getElementById('resCrumb');
const resBackBtn = document.getElementById('resBackBtn');
const resNextBtn = document.getElementById('resNextBtn');
const resNextHint = document.getElementById('resNextHint');
const RES_PLOT_MODS = RES_MODS.filter(m=> m.plotCard);
const RES_ORDER = ['resSummary', 'resValues', ...RES_MODS.map(m=> m.valCard), 'resPlots', ...RES_PLOT_MODS.map(m=> m.plotCard), 'resFiles'];
let currentResPart = 'resSummary';

function resPartLabel(id){
  if(id==='resSummary') return {num:1, step:'Summary'};
  if(id==='resFiles') return {num:4, step:'Files'};
  const vm = RES_MODS.find(m=> m.valCard===id), pm = RES_PLOT_MODS.find(m=> m.plotCard===id);
  if(id==='resValues' || vm) return {num:2, step:'Values', sub: vm ? vm.name : 'all modalities', group:'resValues'};
  return {num:3, step:'Plots', sub: pm ? pm.name : 'all modalities', group:'resPlots'};
}
function resPartText(id){
  const l = resPartLabel(id);
  return l.sub && l.sub!=='all modalities' ? `${l.step}, ${l.sub}` : l.step;
}

function showResPart(id, {keepScroll=false}={}){
  if(!RES_ORDER.includes(id)) id = 'resSummary';
  currentResPart = id;
  const l = resPartLabel(id), i = RES_ORDER.indexOf(id);
  resSummaryEl.classList.toggle('is-shown', id==='resSummary');
  document.getElementById('resFiles').classList.toggle('is-shown', id==='resFiles');
  const valAll = id==='resValues', plotAll = id==='resPlots';
  document.getElementById('resValues').classList.toggle('is-shown', l.group==='resValues');
  document.getElementById('resValues').classList.toggle('one-mod', l.group==='resValues' && !valAll);
  resValueCards.querySelectorAll('.res-val-card').forEach(c=> c.classList.toggle('is-shown', valAll ? resModOn[c.dataset.mod] && (+c.dataset.hits>0 || !resSearch.value.trim()) : c.id===id));
  document.getElementById('resPlots').classList.toggle('is-shown', l.group==='resPlots');
  resPlotCards.querySelectorAll('.res-plot-card').forEach(c=> c.classList.toggle('is-shown', plotAll || c.id===id));
  resSteps.querySelectorAll('.step-item').forEach(b=> b.classList.toggle('current', b.dataset.part===id || b.dataset.part===l.group));
  resSteps.querySelectorAll('.mod-list-row').forEach(r=> r.classList.toggle('current', r.dataset.target===id));
  resCrumb.innerHTML = `<span class="step-num">${l.num}</span><span class="crumb-step">${l.step}</span>`
    + (l.sub ? `<span class="crumb-sep">›</span><span class="crumb-sub">${l.sub}</span>` : '')
    + `<span class="crumb-pos">${i+1} of ${RES_ORDER.length}</span>`;
  resBackBtn.disabled = i===0;
  const next = RES_ORDER[i+1];
  resNextBtn.hidden = !next;
  resNextHint.textContent = next ? 'Next: '+resPartText(next) : '';
  if(!keepScroll) resMainEl.scrollTop = 0;
}

resSteps.addEventListener('click', e=>{
  const item = e.target.closest('.step-item');
  if(item){ showResPart(item.dataset.part); return; }
  const row = e.target.closest('.mod-list-row');
  if(row) showResPart(row.dataset.target);
});
resBackBtn.addEventListener('click', ()=>{
  const i = RES_ORDER.indexOf(currentResPart);
  if(i>0) showResPart(RES_ORDER[i-1]);
});
resNextBtn.addEventListener('click', ()=>{
  const i = RES_ORDER.indexOf(currentResPart);
  if(i < RES_ORDER.length-1) showResPart(RES_ORDER[i+1]);
});

// called by checkConfirmed() in js/check/check-page.js
function resultsChecked(){
  const d = new Date();
  document.getElementById('resCheckedAt').textContent = `today ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

wireSidePanelCollapse('resultsSidebar', 'resultsSidebarCollapseBtn', 'r', 'l');
placeIcons(document.getElementById('view-results'));
renderResSidePanel();
renderResSummary();
renderResChips();
renderResValueCards();
renderResPlotCards();
renderResFiles();
showResPart('resSummary');
