// ---- help box at the bottom of the right panel: what the palette item under
// the mouse is. It shows a step from Operations, a workflow from Workflows,
// or a data row from Available data, and keeps the last item until the next
// one. ----
const helpBoxEl = document.getElementById('helpBox');

function escHtml(text){
  return String(text).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
// one chip per port, with a dot in the colour of its artifact type
function portChipsHtml(list){
  if(!list.length) return '<span class="hb-none">none</span>';
  return list.map(([name, type])=>
    `<span class="hb-chip" title="${escHtml(name)} · ${escHtml(type)}"><span class="type-dot" style="background:${tc(type)}"></span>${escHtml(name)}</span>`
  ).join('');
}

const HELP_IDLE_TEXT = {
  data: 'The data made by the steps on the canvas, and the sequences saved in 1 · Prepare. Point at an item to see where it comes from; click it to see the data.',
  ops: 'Point at a step to see what it does. Click it, or drag it onto the canvas, to add it.',
  blocks: 'Ready-made chains of steps. Point at one to see its steps; click it to add all of them as one box. Click the box to open it, and Pack to close it again.',
};
function showHelpIdle(tab){
  helpBoxEl.innerHTML = `<div class="hb-idle">${HELP_IDLE_TEXT[tab] || HELP_IDLE_TEXT.ops}</div>`;
}

function showStepHelp(op){
  const info = STEP_INFO[op];
  const ports = stepPorts(op);
  const mod = stepMod(op);
  const cat = (info && info.category) || 'other';
  const modLabel = (MODALITIES.find(m=>m.mod===mod) || {}).label || mod;
  // the session input is shared by almost every step, so it is not listed
  const ins = ports.ins.filter(([name])=>name!=='session');
  const note = MOCK_UNAVAILABLE.has(op)
    ? `<div class="hb-warn">Not available: the optional package ${escHtml((info && info.packages[0]) || '')} is not installed.</div>`
    : '';
  helpBoxEl.innerHTML = `
    <div class="hb-head">
      <span class="hb-icon mod-${mod}">${categoryIconSvg(cat, 16)}</span>
      <div class="hb-title">
        <div class="hb-name">${escHtml(stepShortName(op))}</div>
        <div class="hb-sub">${modLabel} · ${cat} · <code>${escHtml(op)}</code></div>
      </div>
    </div>
    <p class="hb-summary">${escHtml((info && info.summary) || 'No summary in the registry.')}</p>
    ${note}
    <div class="hb-ports"><span class="hb-label">In</span>${portChipsHtml(ins)}</div>
    <div class="hb-ports"><span class="hb-label">Out</span>${portChipsHtml(ports.outs)}</div>
    ${sharedChipsHtml(op)}`;
}
// the shared session data (M3Session) that a step reads and writes
function sharedChipsHtml(op){
  const {reads, writes} = sharedData(op);
  const chips = list=> list.map(name=> `<span class="hb-chip" title="${escHtml(sharedDataText(name))}">${sharedIconSvg(10)}${escHtml(name)}</span>`).join('');
  return (reads.length ? `<div class="hb-ports"><span class="hb-label">Reads</span>${chips(reads)}</div>` : '')
    + (writes.length ? `<div class="hb-ports"><span class="hb-label">Writes</span>${chips(writes)}</div>` : '');
}

function showBlockHelp(block){
  helpBoxEl.innerHTML = `
    <div class="hb-head">
      <span class="hb-icon mod-${block.mod}">${categoryIconSvg('workflow', 16)}</span>
      <div class="hb-title">
        <div class="hb-name">${escHtml(block.name)}</div>
        <div class="hb-sub">workflow · ${block.ops.length} steps</div>
      </div>
    </div>
    <p class="hb-summary">${escHtml(block.desc)}</p>
    <div class="hb-steps">${block.ops.map(op=>`<code>${escHtml(stepShortName(op))}</code>`).join(' → ')}</div>`;
}

function showReferenceHelp(w){
  helpBoxEl.innerHTML = `
    <div class="hb-head">
      <span class="hb-icon mod-generic">${categoryIconSvg('workflow', 16)}</span>
      <div class="hb-title">
        <div class="hb-name">${escHtml(w.name)}</div>
        <div class="hb-sub">full example pipeline · yaml</div>
      </div>
    </div>
    <p class="hb-summary">${escHtml(w.desc)}</p>
    <div class="hb-warn">Not in this mockup yet.</div>`;
}

function showDataHelp(key, type, nodeId, op){
  helpBoxEl.innerHTML = `
    <div class="hb-head">
      <span class="hb-icon hb-icon-data"><span class="type-dot" style="background:${tc(type)}"></span></span>
      <div class="hb-title">
        <div class="hb-name">${escHtml(key)}</div>
        <div class="hb-sub"><code>${escHtml(type)}</code></div>
      </div>
    </div>
    <p class="hb-summary">Made by <b>${escHtml(nodeId)}</b> (<code>${escHtml(op)}</code>). Click it to see the data.</p>`;
}

function showSequenceHelp(seq){
  helpBoxEl.innerHTML = `
    <div class="hb-head">
      <span class="hb-icon mod-generic">${categoryIconSvg('sequence', 16)}</span>
      <div class="hb-title">
        <div class="hb-name">${escHtml(seq.name)}</div>
        <div class="hb-sub">saved sequence · ${seq.windows.length} window${seq.windows.length===1?'':'s'}${seq.modalities ? ' · '+seqModalityText(seq) : ''}</div>
      </div>
    </div>
    <p class="hb-summary">Saved in 1 · Prepare. Click it to put it on the canvas, or 👁 to see the signals.</p>`;
}

// the Available data and Workflows lists are drawn again often, so listen on
// their containers
document.getElementById('dataTab').addEventListener('mouseover', e=>{
  const row = e.target.closest('[data-seq], [data-key]');
  if(!row) return;
  if(row.dataset.seq){
    const seq = (typeof SAVED_SEQUENCES!=='undefined' ? SAVED_SEQUENCES : []).find(s=>s.id===row.dataset.seq);
    if(seq) showSequenceHelp(seq);
  } else if(row.dataset.key){
    showDataHelp(row.dataset.key, row.dataset.type, row.dataset.node, row.dataset.op);
  }
});
document.getElementById('blocksTab').addEventListener('mouseover', e=>{
  const item = e.target.closest('[data-block], [data-ref]');
  if(!item) return;
  if(item.dataset.ref){ showReferenceHelp(REFERENCE_WORKFLOWS[Number(item.dataset.ref)]); return; }
  const block = PREDEFINED_BLOCKS.find(b=>b.id===item.dataset.block);
  if(block) showBlockHelp(block);
});
document.getElementById('palTabs').addEventListener('click', e=>{
  const btn = e.target.closest('.pal-tab');
  if(btn) showHelpIdle(btn.dataset.paltab);
});
showHelpIdle(document.querySelector('.pal-tab.active').dataset.paltab);
