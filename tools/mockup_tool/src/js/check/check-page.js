// ================= 3 · Check: side panel, modality cards, details =================
// The side panel chooses one part at a time: "1 Quality checks" (every
// modality), one modality, the run log or provenance. Back / Next walk the
// same order; the last Next confirms the check and opens 4 · Results
// (js/core/tabs.js).
// Each modality card lists its checks, the ones to look at first on top.
// A click on a check opens its details: what it checks, the rule, one
// square per breath or occlusion, the failed ones in a table, and the step
// that produced it.

const qcPassed = c=> c.items.filter(x=>x.passed).length;
// 'pass' every item passed, 'fail' none passed, 'part' some failed
function qcStatus(c){
  const p = qcPassed(c);
  return p===c.items.length ? 'pass' : p===0 ? 'fail' : 'part';
}
const QC_STATUS_TEXT = {pass:'Passed', part:'Some failed', fail:'Failed'};
const QC_SEV_CLASS = {info:'info', warning:'warn', error:'crit'};
const QC_SCOPE_TEXT = {breath:'each EMG breath', occlusion:'each occlusion manoeuvre', pair:'each EMG breath with its ventilator breath', run:'the whole run'};
const QC_ITEM_WORD = {breath:'breath', occlusion:'occlusion', pair:'breath pair', run:'run'};
const qcChecksOf = mod=> QC_CHECKS.filter(c=> c.mod===mod);
const QC_FAILS_SHOWN = 6;            // failed rows in the table before "Show all"
const pl = (n, w)=> `${n} ${w}${n===1?'':'s'}`;

// side panel dot of a modality: green all passed, yellow only info checks
// failed, red a warning or error check failed, grey no checks
function qcModState(mod){
  const cs = qcChecksOf(mod);
  if(!cs.length) return 'none';
  const failing = cs.filter(c=> qcStatus(c)!=='pass');
  if(!failing.length) return 'on';
  return failing.some(c=> c.severity!=='info') ? 'fail' : 'part';
}
function qcCounts(cs){
  const total = cs.reduce((s,c)=> s + c.items.length, 0);
  const passed = cs.reduce((s,c)=> s + qcPassed(c), 0);
  return {total, passed, review: cs.filter(c=> qcStatus(c)!=='pass').length};
}
// the passed share as a bar
function qcBar(passed, total, cls=''){
  const pct = total ? passed/total*100 : 0;
  return `<span class="qc-bar ${cls}" role="img" aria-label="${passed} of ${total} passed"><span style="width:${pct.toFixed(1)}%"></span></span>`;
}
const qcFmt = (c, v)=> v===null || v===undefined ? '–' : `${(+v).toFixed(c.digits)}${c.unit && c.unit!=='%' ? ' '+c.unit : c.unit}`;
const qcLimit = c=> c.limit===null ? c.limitText : `${c.dir==='max' ? '≤' : '≥'} ${qcFmt(c, c.limit)}`;

// ---- side panel: one row per modality under "1 Quality checks" ----
const checkModList = document.getElementById('checkModList');
function renderCheckSidePanel(){
  const all = qcCounts(QC_CHECKS);
  document.getElementById('checkAllCount').textContent = `${all.passed}/${all.total}`;
  checkModList.innerHTML = QC_MODS.map(m=>{
    const cs = qcChecksOf(m.id), n = qcCounts(cs), st = qcModState(m.id);
    const sub = cs.length ? `${pl(cs.length, 'check')} · ${n.passed}/${n.total} passed` : 'no checks';
    const tip = {on:'All checks passed', part:'Info checks have fails', fail:'A warning or error check has fails', none:'No checks'}[st];
    return `<div class="mod-list-row" data-target="${m.card}" title="Show the ${m.name} checks">
      <span class="swatch ${m.id}"></span>
      <div class="info"><div class="nm">${m.name}</div><div class="file">${sub}</div></div>
      <span class="st ${st}" title="${tip}"></span>
    </div>`;
  }).join('');
}

// ---- one card per modality ----
const checkModCards = document.getElementById('checkModCards');
const qcOpen = new Set();            // checks with open details
const qcPick = {};                   // check code -> item id picked on the strip
const qcFilter = {};                 // modality -> 'all' | 'review'
const QC_ORDER = {fail:0, part:1, pass:2}, QC_SEV_ORDER = {error:0, warning:1, info:2};

function renderCheckCards(){
  checkModCards.innerHTML = '';
  QC_MODS.forEach(m=>{
    const cs = qcChecksOf(m.id), n = qcCounts(cs), filter = qcFilter[m.id] || 'all';
    const card = document.createElement('div');
    card.className = `card mod-card mod-${m.id} check-mod`;
    card.id = m.card;
    card.innerHTML = `
      <div class="card-head">
        <div class="mod-head-left">
          <div class="modal-info">
            <div class="name"><span class="mod-name">${m.name}</span><span class="mod-device">quality checks</span></div>
            <div class="meta">${cs.length ? `${pl(cs.length, 'check')} · ${n.review} to review` : 'no checks in this run'}</div>
          </div>
        </div>
        ${cs.length ? `<div class="qc-head-sum">${qcBar(n.passed, n.total, 'wide')}<span class="qc-head-num"><b>${n.passed}</b> of ${n.total} passed</span></div>` : ''}
      </div>
      <div class="card-body"></div>`;
    const body = card.querySelector('.card-body');
    if(!cs.length){
      body.innerHTML = `<div class="qc-empty">${prepIcon('minus', 22)}<div><b>${m.empty}</b><span>Add a quality step for ${m.name} in <a href="#" data-goto-design>2 · Design</a>; its checks then show here after the next run.</span></div></div>`;
      body.querySelector('[data-goto-design]').addEventListener('click', e=>{ e.preventDefault(); goToView('design'); });
      checkModCards.appendChild(card);
      return;
    }
    body.innerHTML = `
      <div class="card-toolbar">
        <div class="ct-left">
          <span class="ct-label">Show</span>
          <div class="tb-seg small" role="group" aria-label="Checks to show">
            <button data-filter="all" class="${filter==='all'?'on':''}">All ${cs.length}</button>
            <button data-filter="review" class="${filter==='review'?'on':''}">To review ${n.review}</button>
          </div>
        </div>
        <div class="ct-right"><span class="qc-sort-note">Failed first, then by severity</span></div>
      </div>
      <div class="qc-list" role="list"></div>`;
    const list = body.querySelector('.qc-list');
    const shown = cs.filter(c=> filter==='all' || qcStatus(c)!=='pass')
      .sort((a,b)=> QC_ORDER[qcStatus(a)]-QC_ORDER[qcStatus(b)] || QC_SEV_ORDER[a.severity]-QC_SEV_ORDER[b.severity]);
    if(!shown.length) list.innerHTML = '<div class="windows-empty">Every check passed.</div>';
    shown.forEach(c=> list.appendChild(qcRow(c)));
    body.querySelector('.tb-seg').addEventListener('click', e=>{
      const b = e.target.closest('[data-filter]'); if(!b) return;
      qcFilter[m.id] = b.dataset.filter;
      renderCheckCards(); showCheckPart(currentCheckPart, {keepScroll:true});
    });
    checkModCards.appendChild(card);
  });
}

function qcRow(c){
  const st = qcStatus(c), p = qcPassed(c), open = qcOpen.has(c.code);
  const row = document.createElement('div');
  row.className = `qc-check st-${st}` + (open ? ' open' : '');
  row.setAttribute('role', 'listitem');
  const result = c.scope==='run'
    ? `<span class="qc-run-val">${qcFmt(c, c.items[0].value)}</span>`
    : `${qcBar(p, c.items.length, st)}<span class="qc-count"><b>${p}</b> / ${c.items.length}</span>`;
  row.innerHTML = `
    <button type="button" class="qc-row" aria-expanded="${open}" aria-controls="qc-d-${c.code}">
      <span class="qc-dot ${st}" title="${QC_STATUS_TEXT[st]}"></span>
      <span class="qc-name"><span class="qc-plain">${c.name}</span><code>${c.code}</code></span>
      <span class="qc-result">${result}</span>
      <span class="qc-status">${QC_STATUS_TEXT[st]}</span>
      <span class="sev-pill ${QC_SEV_CLASS[c.severity]}">${c.severity}</span>
      <span class="qc-chev">${prepIcon('chevron-down')}</span>
    </button>
    <div class="qc-detail" id="qc-d-${c.code}"><div class="qc-detail-inner">${qcDetail(c)}</div></div>`;
  row.querySelector('.qc-row').addEventListener('click', ()=>{
    const isOpen = row.classList.toggle('open');
    row.querySelector('.qc-row').setAttribute('aria-expanded', String(isOpen));
    if(isOpen) qcOpen.add(c.code); else qcOpen.delete(c.code);
  });
  wireQcDetail(row, c);
  return row;
}

function qcDetail(c){
  const st = qcStatus(c), failed = c.items.filter(x=>!x.passed);
  const facts = [
    ['Checks', QC_SCOPE_TEXT[c.scope]],
    ['Signal', c.signal],
    ['Produced by', `<code>${c.step}</code> · ${c.stepMs.toFixed(1)} ms`],
    ...(c.params.length ? [['Parameters', c.params.map(([k,v])=>`<code>${k}</code> = ${v}`).join('<br>')]] : []),
  ];
  let html = `
    <div class="qc-what">
      <p>${c.what}</p>
      <p class="qc-rule"><span>Rule</span>${c.rule}</p>
    </div>
    <dl class="qc-facts">${facts.map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>`;
  if(c.scope==='run'){
    const it = c.items[0];
    html += `<div class="qc-run-box ${st}">
      <div><span class="k">${c.valueName}</span><span class="v">${qcFmt(c, it.value)}</span></div>
      <div><span class="k">limit</span><span class="v">${qcLimit(c)}</span></div>
      <div class="msg">${it.passed ? 'Passed.' : it.message}</div>
    </div>`;
  } else {
    const word = QC_ITEM_WORD[c.scope];
    html += `
      <div class="qc-strip-head"><span class="ct-label">${word}s</span><span class="qc-strip-sum">${qcPassed(c)} of ${pl(c.items.length, word)} passed · limit ${qcLimit(c)}</span></div>
      <div class="qc-strip" role="group" aria-label="One square per ${word}">
        ${c.items.map(x=>`<button type="button" class="qc-sq ${x.passed?'ok':'bad'}" data-item="${x.id}" title="${qcItemTitle(c, x)}" aria-label="${qcItemTitle(c, x)}"><span>${x.id.replace('Pocc ','P')}</span></button>`).join('')}
      </div>
      <div class="qc-pick" hidden></div>`;
    if(failed.length){
      html += `<table class="qc-fails">
        <thead><tr><th>Failed ${word}</th><th>Time</th><th>${c.valueName}</th><th>Limit</th><th>Message</th><th></th></tr></thead>
        <tbody>${failed.map((x, k)=>`<tr data-item="${x.id}"${k>=QC_FAILS_SHOWN ? ' class="more"' : ''}>
          <td class="mono">${x.id}</td><td class="mono">${fmtRange(x.t0, x.t1)}</td>
          <td class="mono">${qcFmt(c, x.value)}</td><td class="mono">${qcLimit(c)}</td><td>${x.message}</td>
          <td><button class="tb-btn small quiet" data-show-item="${x.id}" title="Show this ${word} in 1 · Prepare">Show</button></td></tr>`).join('')}</tbody>
      </table>`
      + (failed.length>QC_FAILS_SHOWN ? `<button class="tb-btn small quiet qc-more" data-more>Show all ${pl(failed.length, 'failed '+word)}</button>` : '');
    }
  }
  html += `<div class="qc-actions">
    <button class="tb-btn small" data-find-step>Find step in 2 · Design</button>
    ${c.scope!=='run' ? `<button class="tb-btn small" data-show-prepare>Show in 1 · Prepare</button>` : ''}
  </div>`;
  return html;
}
function qcItemTitle(c, x){
  return `${QC_ITEM_WORD[c.scope]} ${x.id} · ${fmtRange(x.t0, x.t1)} · ${qcFmt(c, x.value)} · ${x.passed ? 'passed' : 'failed'}`;
}

function wireQcDetail(row, c){
  const pick = row.querySelector('.qc-pick');
  const showPick = id=>{
    const x = c.items.find(i=> i.id===id); if(!x || !pick) return;
    qcPick[c.code] = id;
    row.querySelectorAll('.qc-sq').forEach(s=> s.classList.toggle('picked', s.dataset.item===id));
    pick.hidden = false;
    pick.className = 'qc-pick ' + (x.passed ? 'ok' : 'bad');
    pick.innerHTML = `<b>${QC_ITEM_WORD[c.scope]} ${x.id}</b><span class="mono">${fmtRange(x.t0, x.t1)}</span>
      <span class="mono">${c.valueName} ${qcFmt(c, x.value)}</span><span class="mono">limit ${qcLimit(c)}</span>
      <span>${x.passed ? 'Passed.' : x.message}</span>
      <button class="tb-btn small quiet" data-show-item="${x.id}">Show in 1 · Prepare</button>`;
  };
  row.querySelectorAll('.qc-sq').forEach(sq=> sq.addEventListener('click', ()=> showPick(sq.dataset.item)));
  if(qcPick[c.code]) showPick(qcPick[c.code]);
  row.querySelector('.qc-detail').addEventListener('click', e=>{
    const show = e.target.closest('[data-show-item]');
    if(show){ qcShowInPrepare(c, c.items.find(i=> i.id===show.dataset.showItem)); return; }
    if(e.target.closest('[data-show-prepare]')){ qcShowInPrepare(c, c.items.find(i=>!i.passed) || c.items[0]); return; }
    if(e.target.closest('[data-find-step]')) qcFindStep(c);
    const more = e.target.closest('[data-more]');
    if(more){
      const all = row.querySelector('.qc-fails').classList.toggle('all');
      more.textContent = all ? `Show the first ${QC_FAILS_SHOWN}` : `Show all ${pl(c.items.filter(x=>!x.passed).length, 'failed '+QC_ITEM_WORD[c.scope])}`;
    }
  });
}

// "Show in 1 · Prepare": the modality card of 1 · Prepare, zoomed on the
// breath or occlusion (file 1 of the modality, the recording of the run)
function qcShowInPrepare(c, x){
  const m = QC_MODS.find(q=> q.id===c.mod);
  goToView('prepare');
  showSection(m.prepCard);
  const merge = {emg: typeof EMG_MERGE!=='undefined' && EMG_MERGE, vent: typeof VENT_MERGE!=='undefined' && VENT_MERGE}[c.mod];
  if(merge && merge.shown!=='f1') merge.showSource('f1');
  const z = ZOOM[c.mod];
  if(z && x && x.t0!==undefined) z.setWindow(Math.max(0, x.t0-3), x.t1+3);
}
// "Find step in 2 · Design": the step in the Operations list, and on the
// canvas when the example graph has it
function qcFindStep(c){
  goToView('design');
  document.querySelector('#palTabs [data-paltab="ops"]')?.click();
  const search = document.getElementById('paletteSearchInput');
  if(search){ search.value = c.step; search.dispatchEvent(new Event('input', {bubbles:true})); }
  const node = NODES.find(n=> n.op===c.step);
  if(node && typeof focusNode==='function') focusNode(node.id);
}

// ---- one part at a time ----
const checkMainEl = document.getElementById('checkMain');
const checkSteps = document.getElementById('checkSteps');
const checkCrumb = document.getElementById('checkCrumb');
const checkBackBtn = document.getElementById('checkBackBtn');
const checkNextBtn = document.getElementById('checkNextBtn');
const checkNextHint = document.getElementById('checkNextHint');
const CHECK_ORDER = ['checkAll', ...QC_MODS.map(m=>m.card), 'checkLog', 'checkProv', 'checkConfirm'];
let currentCheckPart = 'checkAll';

function checkPartLabel(id){
  if(id==='checkLog') return {num:2, step:'Run log'};
  if(id==='checkProv') return {num:3, step:'Provenance'};
  if(id==='checkConfirm') return {num:4, step:'Confirm'};
  const m = QC_MODS.find(q=> q.card===id);
  return {num:1, step:'Quality checks', sub: m ? m.name : 'all modalities'};
}
function checkPartText(id){
  const l = checkPartLabel(id);
  return QC_MODS.some(m=> m.card===id) ? l.sub : l.step;
}

function showCheckPart(id, {keepScroll=false}={}){
  if(!CHECK_ORDER.includes(id)) id = 'checkAll';
  currentCheckPart = id;
  const isMod = QC_MODS.some(m=> m.card===id);
  checkModCards.classList.toggle('is-shown', id==='checkAll' || isMod);
  checkModCards.querySelectorAll('.check-mod').forEach(c=> c.classList.toggle('is-shown', id==='checkAll' || c.id===id));
  ['checkLog', 'checkProv', 'checkConfirm'].forEach(k=> document.getElementById(k).classList.toggle('is-shown', k===id));
  checkSteps.querySelectorAll('.step-item').forEach(b=> b.classList.toggle('current', b.dataset.part===id || (isMod && b.dataset.part==='checkAll')));
  checkModList.querySelectorAll('.mod-list-row').forEach(r=> r.classList.toggle('current', r.dataset.target===id));
  const l = checkPartLabel(id), i = CHECK_ORDER.indexOf(id);
  checkCrumb.innerHTML = `<span class="step-num">${l.num}</span><span class="crumb-step">${l.step}</span>`
    + (l.sub ? `<span class="crumb-sep">›</span><span class="crumb-sub">${l.sub}</span>` : '')
    + `<span class="crumb-pos">${i+1} of ${CHECK_ORDER.length}</span>`;
  checkBackBtn.disabled = i===0;
  const next = CHECK_ORDER[i+1];
  checkNextBtn.hidden = !next;
  checkNextHint.textContent = next ? 'Next: '+checkPartText(next) : '';
  if(!keepScroll) checkMainEl.scrollTop = 0;
}

checkSteps.addEventListener('click', e=>{
  const item = e.target.closest('.step-item');
  if(item){ showCheckPart(item.dataset.part); return; }
  const row = e.target.closest('.mod-list-row');
  if(row) showCheckPart(row.dataset.target);
});
checkBackBtn.addEventListener('click', ()=>{
  const i = CHECK_ORDER.indexOf(currentCheckPart);
  if(i>0) showCheckPart(CHECK_ORDER[i-1]);
});
checkNextBtn.addEventListener('click', ()=>{
  const i = CHECK_ORDER.indexOf(currentCheckPart);
  if(i < CHECK_ORDER.length-1) showCheckPart(CHECK_ORDER[i+1]);
});

// ---- run log: a filter by modality, and the time of each step as a bar ----
const LOG_MOD = {eit:'eit', emg:'emg', ventilator:'vent', session:'session'};
(function prepareRunLog(){
  const lines = [...document.querySelectorAll('#runLog .log-line')];
  const steps = lines.map(el=>{
    const m = el.textContent.match(/^\s*\+\s*[\d.]+s\s*([a-z_]+)\.([a-z_]+)\s+([\d.]+) ms/);
    return m ? {el, mod: LOG_MOD[m[1]] || 'session', ms: +m[3]} : {el};
  });
  const max = Math.max(...steps.filter(s=>s.ms!==undefined).map(s=>s.ms));
  steps.forEach(s=>{
    if(s.ms===undefined){ s.el.classList.add('log-summary'); return; }
    s.el.innerHTML = `<span class="log-text">${s.el.innerHTML}</span>`;
    s.el.dataset.mod = s.mod;
    s.el.insertAdjacentHTML('beforeend', `<span class="log-bar mod-${s.mod}" title="${s.ms.toFixed(1)} ms"><span style="width:${Math.max(0.5, s.ms/max*100).toFixed(1)}%"></span></span>`);
  });
  const count = document.getElementById('logCount');
  const filter = mod=>{
    let n = 0;
    steps.forEach(s=>{ if(s.ms===undefined) return; const on = mod==='all' || s.mod===mod; s.el.hidden = !on; if(on) n++; });
    count.textContent = `${n} step${n===1?'':'s'}`;
  };
  document.getElementById('logFilter').addEventListener('click', e=>{
    const b = e.target.closest('[data-mod]'); if(!b) return;
    document.querySelectorAll('#logFilter button').forEach(x=> x.classList.toggle('on', x===b));
    filter(b.dataset.mod);
  });
  filter('all');
})();

// ---- 4 Confirm: what is still open, and the state after confirming ----
function renderConfirmSummary(){
  const all = qcCounts(QC_CHECKS);
  const warn = QC_CHECKS.filter(c=> c.severity!=='info' && qcStatus(c)!=='pass').length;
  const rows = QC_MODS.map(m=>{
    const cs = qcChecksOf(m.id), n = qcCounts(cs), st = qcModState(m.id);
    const txt = !cs.length ? 'no checks' : n.review ? `${pl(n.review, 'check')} to review` : 'all passed';
    return `<button type="button" class="confirm-mod" data-target="${m.card}"><span class="st ${st}"></span><b>${m.name}</b><span>${txt}</span></button>`;
  }).join('');
  document.getElementById('confirmSummary').innerHTML = `
    <div class="confirm-nums">
      <div><span class="v">${all.passed}<small> / ${all.total}</small></span><span class="k">passed</span></div>
      <div><span class="v">${all.review}</span><span class="k">checks to review</span></div>
      <div class="${warn?'warn':''}"><span class="v">${warn}</span><span class="k">warning${warn===1?'':'s'}</span></div>
    </div>
    <div class="confirm-mods">${rows}</div>`;
}
document.getElementById('confirmSummary').addEventListener('click', e=>{
  const b = e.target.closest('[data-target]'); if(b) showCheckPart(b.dataset.target);
});
// called by js/core/tabs.js once the run is confirmed
function checkConfirmed(){
  document.getElementById('confirmState').textContent = 'Confirmed. 4 · Results is open.';
  document.getElementById('confirmStepMeta').textContent = 'done';
  checkSteps.querySelector('[data-part="checkConfirm"]').classList.add('done');
}
renderConfirmSummary();

wireSidePanelCollapse('checkSidebar', 'checkSidebarCollapseBtn', 'r', 'l');
placeIcons(document.getElementById('view-check'));
renderCheckSidePanel();
renderCheckCards();
showCheckPart('checkAll');
