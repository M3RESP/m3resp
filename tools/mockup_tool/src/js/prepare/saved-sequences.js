// ---- saved sequences (SAVED_SEQUENCES itself lives in js/core/tabs.js, which
// loads first — see the note there for why) ----

function defaultSequenceName(){
  return 'sequence_'+(SAVED_SEQUENCES.length+1);
}
(function primeSequenceNameInput(){
  const input = document.getElementById('sequenceNameInput');
  if(input && !input.value) input.value = defaultSequenceName();
})();

function renderSavedSequences(){
  const list = document.getElementById('savedSequencesList');
  const countEl = document.getElementById('savedSequencesCount');
  countEl.textContent = SAVED_SEQUENCES.length
    ? `${SAVED_SEQUENCES.length} sequence${SAVED_SEQUENCES.length===1?'':'s'} saved`
    : '';
  if(SAVED_SEQUENCES.length===0){
    list.innerHTML = '<div class="windows-empty">No sequences saved yet — name one above and click "Save as new sequence".</div>';
    return;
  }
  list.innerHTML = '';
  SAVED_SEQUENCES.forEach(seq=>{
    const row = document.createElement('div');
    row.className = 'seq-row';
    row.innerHTML = `
      <span class="seq-icon">▤</span>
      <span class="seq-name">${seq.name}</span>
      <span class="seq-meta">${seq.windows.length} window${seq.windows.length===1?'':'s'} · ${fmtDuration(seq.total)}</span>
      <button data-remove-seq>Remove</button>`;
    row.querySelector('[data-remove-seq]').addEventListener('click', ()=>{
      SAVED_SEQUENCES = SAVED_SEQUENCES.filter(s=>s.id!==seq.id);
      renderSavedSequences();
    });
    list.appendChild(row);
  });
}
renderSavedSequences();

document.getElementById('saveSequenceBtn').addEventListener('click', ()=>{
  // what gets saved is the resolved selection: kept spans minus removed spans
  const sorted = resolvedSegments();
  const cutCount = sortedCuts().length;
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
    savedAt: Date.now(),
  });
  renderSavedSequences();

  const rows = sorted.map((w,idx)=>
    `${idx+1}) ${fmtTime(w.start)}–${fmtTime(w.end)} <span style="color:var(--text-faint);">(${fmtDuration(w.end-w.start)})</span>`
  ).join(' &nbsp;·&nbsp; ');
  summary.innerHTML = `
    <div class="ok-pill">✓ Saved as "<b>${name}</b>"</div>
    <div><b>${sorted.length}</b> segment${sorted.length===1?'':'s'}, <b>${fmtDuration(total)}</b> combined${cutCount?` — after cutting out <b>${cutCount}</b> removed span${cutCount===1?'':'s'}`:''} — concatenated in time order via <code>Sequence.concatenate()</code>. Now listed under <b>Available data</b> on <b>2 · Design</b>, alongside any other sequences you save:</div>
    <div style="margin-top:8px;">${rows}</div>`;
  summary.classList.remove('hidden');

  nameInput.value = defaultSequenceName();
});

(function drawSliceAxis(){
  const el=document.getElementById('sliceAxis');
  let out='';
  for(let k=0;k<=6;k++){
    const t=REC_SECONDS*(k/6);
    out+=`<span>${t.toFixed(1)}s</span>`;
  }
  el.innerHTML=out;
})();

renderWindows();

// ---- sync offset overrides (placeholder = estimated value; typing overrides it) ----
const SYNC_SCALE = 4.8; // seconds of offset represented as 10% of the illustrative track width
function wireSyncOffset(inputId, barId, label, placeholderSeconds){
  const input = document.getElementById(inputId);
  const bar = document.getElementById(barId);
  input.addEventListener('input', ()=>{
    const raw = input.value.trim();
    const sec = raw==='' ? placeholderSeconds : parseFloat(raw);
    const shiftPct = Math.max(-60, Math.min(60, (sec/SYNC_SCALE)*10));
    const left = Math.max(0, shiftPct);
    const width = 100 - Math.abs(shiftPct);
    bar.style.left = left+'%';
    bar.style.width = width+'%';
    bar.textContent = `offset ${sec>=0?'':''}${sec.toFixed(1)} s${raw===''?' (estimated)':' (override)'}`;
  });
}
wireSyncOffset('syncOffsetEit', 'syncBarEit', 'EIT', 0.0);
wireSyncOffset('syncOffsetEmg', 'syncBarEmg', 'EMG', -2.0);

