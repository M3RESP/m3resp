// ---- saved sequences (SAVED_SEQUENCES itself lives in js/core/tabs.js, which
// loads first — see the note there for why) ----

function defaultSequenceName(){
  return 'sequence_'+(SAVED_SEQUENCES.length+1);
}
(function primeSequenceNameInput(){
  const input = document.getElementById('sequenceNameInput');
  if(input && !input.value) input.value = defaultSequenceName();
})();

let lastSavedSeqId = null;   // the row of the newest sequence lights up once
// one list in the side panel and one at the bottom of the working window
const SEQ_LISTS = [
  {list:'savedSequencesList', count:'savedSequencesCount', countText:n=> n ? String(n) : '',
   empty:()=> `<div class="seq-empty">${prepIcon('seq', 20)}No sequences yet. Save one from the working window; 2 · Design lists it under Available data.</div>`},
  {list:'windowSequencesList', count:'windowSequencesCount', countText:n=> n ? `${n} saved · listed under Available data on 2 · Design` : '',
   empty:()=> '<div class="windows-empty">No sequences yet. "Save as new sequence" adds one here; 2 · Design lists it under Available data.</div>'},
];
// the modalities of a sequence, for example "EIT, ventilator"
function seqModalityText(seq){
  return (seq.modalities || []).map(m=> SLICE_MOD_NAMES[m] || m).join(', ');
}
function seqRow(seq){
  const row = document.createElement('div');
  row.className = 'seq-row' + (seq.id===lastSavedSeqId ? ' is-new' : '');
  row.title = `${seq.name}: ${seq.windows.length} window${seq.windows.length===1?'':'s'}, ${fmtDuration(seq.total)}`;
  row.innerHTML = `
    <span class="seq-icon">${prepIcon('seq', 14)}</span>
    <div class="seq-text"><div class="seq-name">${seq.name}</div>
      <div class="seq-meta">${seq.windows.length} window${seq.windows.length===1?'':'s'} · ${fmtDuration(seq.total)}${seq.modalities ? ' · '+seqModalityText(seq) : ''}</div></div>
    <button class="tb-btn small icon quiet remove" data-remove-seq title="Remove this sequence" aria-label="Remove this sequence">${prepIcon('close')}</button>`;
  row.querySelector('[data-remove-seq]').addEventListener('click', ()=>{
    SAVED_SEQUENCES = SAVED_SEQUENCES.filter(s=>s.id!==seq.id);
    renderSavedSequences();
    // the windows of a removed sequence are unlocked and can be saved again
    windows.forEach(w=>{ if(w.seqId===seq.id) delete w.seqId; });
    renderWindows();
  });
  return row;
}
function renderSavedSequences(){
  SEQ_LISTS.forEach(ids=>{
    const list = document.getElementById(ids.list);
    document.getElementById(ids.count).textContent = ids.countText(SAVED_SEQUENCES.length);
    if(SAVED_SEQUENCES.length===0){
      list.innerHTML = ids.empty();
      return;
    }
    list.innerHTML = '';
    SAVED_SEQUENCES.forEach(seq=> list.appendChild(seqRow(seq)));
  });
}
renderSavedSequences();

document.getElementById('saveSequenceBtn').addEventListener('click', ()=>{
  // what gets saved is the resolved selection of the windows without a
  // sequence: kept spans minus removed spans
  const sorted = resolvedSegments();
  const cutCount = openCuts().length;
  const summary = document.getElementById('saveSummary');
  if(sorted.length===0){ summary.classList.add('hidden'); return; }
  const nameInput = document.getElementById('sequenceNameInput');
  let name = nameInput.value.trim() || defaultSequenceName();
  // keep names unique so several saved sequences never collide in "Available data"
  let uniqueName = name, i = 2;
  while(SAVED_SEQUENCES.some(s=>s.name===uniqueName)){ uniqueName = name+'_'+i; i++; }
  name = uniqueName;

  const total = sorted.reduce((s,w)=>s+(w.end-w.start), 0);
  seqSaveSeq++;
  SAVED_SEQUENCES.push({
    id: 'seq'+seqSaveSeq,
    name,
    windows: sorted.map(w=>({start:w.start, end:w.end})),
    total,
    modalities: sliceModalities(),
    savedAt: Date.now(),
  });
  lastSavedSeqId = 'seq'+seqSaveSeq;
  renderSavedSequences();
  lastSavedSeqId = null;
  // the saved windows turn grey and are locked
  openWindows().forEach(w=>{ w.seqId = 'seq'+seqSaveSeq; });
  renderWindows();

  const rows = sorted.map((w,idx)=>
    `${idx+1}) ${fmtRange(w.start, w.end)} <span style="color:var(--text-dim);">(${fmtDuration(w.end-w.start)})</span>`
  ).join(' &nbsp;·&nbsp; ');
  summary.innerHTML = `
    <div class="ok-pill">✓ Saved as <b>${name}</b></div>
    <div><b>${sorted.length}</b> segment${sorted.length===1?'':'s'}, <b>${fmtDuration(total)}</b> combined${cutCount?` — after cutting out <b>${cutCount}</b> removed span${cutCount===1?'':'s'}`:''} — concatenated in time order via <code>Sequence.concatenate()</code>. Now listed under <b>Available data</b> on <b>2 · Design</b>, alongside any other sequences you save:</div>
    <div style="margin-top:8px;">${rows}</div>`;
  summary.classList.remove('hidden');
  nameInput.value = defaultSequenceName();
});

// time axis under the working-window lanes, over the length of the ventilator recording
function drawSliceAxis(){
  const el=document.getElementById('sliceAxis');
  let out='';
  for(let k=0;k<=6;k++){
    const t=sliceDur()*(k/6);
    out+=`<span>${fmtSec(t)}</span>`;
  }
  el.innerHTML=out;
}
drawSliceAxis();

renderWindows();
