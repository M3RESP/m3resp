// ---- multiple selection windows: each {id, start, end, type} in seconds.
// type 'keep' selects a span to save; type 'cut' is an INVERSE window — that
// span is removed and whatever surrounds it is saved instead. With no keep
// windows at all the whole recording is treated as kept, so cut windows on
// their own mean "save everything except these". ----
let winSeq = 1;
let windows = [{id:'w1', start:4, end:22, type:'keep'}];
let activeWindowId = 'w1';
const CUT_COLOR = '#d1554f'; // --crit, needed as a literal for SVG fill/stroke

function fmtTime(sec){
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec/60), s = sec%60;
  return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
}
function fmtDuration(sec){
  const m = Math.floor(sec/60), s = Math.round(sec%60);
  return m+'m '+s+'s';
}
function clampSeconds(v){ return Math.min(Math.max(v,0), REC_SECONDS); }
function sortedWindows(){ return [...windows].sort((a,b)=>a.start-b.start); }
function isCut(w){ return w.type==='cut'; }
function sortedKeeps(){ return sortedWindows().filter(w=>!isCut(w)); }
function sortedCuts(){ return sortedWindows().filter(isCut); }
function seqIndexOf(id){
  const w = windows.find(x=>x.id===id);
  if(!w) return 0;
  const pool = isCut(w) ? sortedCuts() : sortedKeeps();
  return pool.findIndex(x=>x.id===id)+1;
}
function windowColor(w){
  if(!w) return NEUTRAL_COLOR;
  if(isCut(w)) return CUT_COLOR;                 // removed → red
  if(w.type==='keep') return KEEP_COLOR;         // kept → green
  return NEUTRAL_COLOR;                          // plain, unclassified band → blue
}

// Merge the keep spans (or the whole recording, if only cuts were drawn), then
// subtract every cut span. The result is what "Save as new sequence" writes.
function resolvedSegments(){
  const keeps = sortedKeeps().map(w=>({start:w.start, end:w.end}));
  const cuts = sortedCuts().map(w=>({start:w.start, end:w.end}));
  if(keeps.length===0 && cuts.length===0) return [];
  let merged = [];
  const base = keeps.length ? keeps : [{start:0, end:REC_SECONDS}];
  base.forEach(k=>{
    const last = merged[merged.length-1];
    if(last && k.start<=last.end) last.end = Math.max(last.end, k.end);
    else merged.push({...k});
  });
  cuts.forEach(c=>{
    const next = [];
    merged.forEach(s=>{
      if(c.end<=s.start || c.start>=s.end){ next.push(s); return; }   // no overlap
      if(c.start>s.start) next.push({start:s.start, end:c.start});     // head survives
      if(c.end<s.end)     next.push({start:c.end,   end:s.end});       // tail survives
    });
    merged = next;
  });
  return merged.filter(s=>s.end-s.start > 0.05);
}
function resolvedTotal(){ return resolvedSegments().reduce((s,x)=>s+(x.end-x.start), 0); }

const sliceTrack = document.getElementById('sliceTrack');
const startInput = document.getElementById('sliceStartInput');
const endInput = document.getElementById('sliceEndInput');
const startLabel = document.getElementById('sliceStartLabel');
const endLabel = document.getElementById('sliceEndLabel');
const durationChip = document.getElementById('sliceDuration');
const activeHint = document.getElementById('sliceActiveHint');

// custom modalities register their plot's <svg> here (see js/prepare/sidebar.js)
// so updateMasks() can (re)draw one crop overlay per selected window into it.
const MASK_SVGS = {};

function updateMasks(){
  const keeps = sortedKeeps();
  const cuts = sortedCuts();
  Object.values(MASK_SVGS).forEach(svgEl=>{
    if(!svgEl) return;
    svgEl.querySelectorAll('.slice-mask, .slice-mask-edge').forEach(el=>el.remove());
    const vb = (svgEl.getAttribute('viewBox')||'0 0 700 118').split(' ').map(Number);
    const h = vb[3];
    // cut spans first, so a keep window drawn over one still reads on top
    cuts.forEach(w=>{
      const x1=(w.start/REC_SECONDS)*700, x2=(w.end/REC_SECONDS)*700;
      const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
      rect.setAttribute('class','slice-mask'); rect.setAttribute('x',x1); rect.setAttribute('y',0);
      rect.setAttribute('width', Math.max(0,x2-x1)); rect.setAttribute('height', h);
      rect.setAttribute('fill', CUT_COLOR); rect.style.opacity = 0.20;
      svgEl.appendChild(rect);
      [x1,x2].forEach(x=>{
        const ln=document.createElementNS('http://www.w3.org/2000/svg','line');
        ln.setAttribute('class','slice-mask-edge'); ln.setAttribute('x1',x); ln.setAttribute('x2',x);
        ln.setAttribute('y1',0); ln.setAttribute('y2',h);
        ln.setAttribute('stroke', CUT_COLOR); ln.setAttribute('stroke-dasharray','4 3');
        svgEl.appendChild(ln);
      });
    });
    keeps.forEach(w=>{
      const color = windowColor(w);
      const x1=(w.start/REC_SECONDS)*700, x2=(w.end/REC_SECONDS)*700;
      const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
      rect.setAttribute('class','slice-mask'); rect.setAttribute('x',x1); rect.setAttribute('y',0);
      rect.setAttribute('width', Math.max(0,x2-x1)); rect.setAttribute('height', h);
      rect.setAttribute('fill', color); rect.style.opacity = 0.14;
      const l=document.createElementNS('http://www.w3.org/2000/svg','line');
      l.setAttribute('class','slice-mask-edge'); l.setAttribute('x1',x1); l.setAttribute('x2',x1); l.setAttribute('y1',0); l.setAttribute('y2',h);
      l.setAttribute('stroke', color);
      const r=document.createElementNS('http://www.w3.org/2000/svg','line');
      r.setAttribute('class','slice-mask-edge'); r.setAttribute('x1',x2); r.setAttribute('x2',x2); r.setAttribute('y1',0); r.setAttribute('y2',h);
      r.setAttribute('stroke', color);
      svgEl.appendChild(rect); svgEl.appendChild(l); svgEl.appendChild(r);
    });
  });
}

function positionWindowEl(el, w){
  const lp=(w.start/REC_SECONDS)*100, rp=(w.end/REC_SECONDS)*100;
  el.style.left = lp+'%';
  el.style.width = Math.max(0, rp-lp)+'%';
}

function secondsAtClientX(clientX){
  const rect = sliceTrack.getBoundingClientRect();
  const frac = (clientX-rect.left)/rect.width;
  return clampSeconds(frac*REC_SECONDS);
}

function applyActiveClasses(){
  sliceTrack.querySelectorAll('.slice-select').forEach(el=> el.classList.toggle('active', el.dataset.id===activeWindowId));
  document.querySelectorAll('.window-row').forEach(row=> row.classList.toggle('active', row.dataset.id===activeWindowId));
  updateActiveFields();
}

function updateActiveFields(){
  const w = windows.find(x=>x.id===activeWindowId);
  if(!w){
    startInput.value=''; endInput.value='';
    activeHint.textContent = windows.length ? 'click a window to edit it' : 'no windows selected';
    durationChip.textContent = '—';
    return;
  }
  activeHint.textContent = isCut(w)
    ? 'editing removed span '+seqIndexOf(w.id)
    : 'editing window '+seqIndexOf(w.id);
  if(sliceMode==='time'){ startInput.value=w.start.toFixed(1); endInput.value=w.end.toFixed(1); }
  else { startInput.value=Math.round(w.start*EIT_HZ); endInput.value=Math.round(w.end*EIT_HZ); }
  durationChip.textContent = (isCut(w) ? 'removing: ' : 'active window: ')+fmtDuration(w.end-w.start);
}

function wireWindowDrag(el, w){
  function startDrag(mode, startClientX){
    el.classList.add('dragging');
    const handleEl = mode!=='move' ? el.querySelector(`[data-h="${mode}"]`) : null;
    if(handleEl) handleEl.classList.add('dragging');
    const origStart=w.start, origEnd=w.end;
    function onMove(ev){
      if(mode==='left'){ w.start = clampSeconds(Math.min(secondsAtClientX(ev.clientX), w.end-1)); }
      else if(mode==='right'){ w.end = clampSeconds(Math.max(secondsAtClientX(ev.clientX), w.start+1)); }
      else {
        const rect=sliceTrack.getBoundingClientRect();
        const deltaSec=((ev.clientX-startClientX)/rect.width)*REC_SECONDS;
        const width=origEnd-origStart;
        let lo=origStart+deltaSec, hi=origEnd+deltaSec;
        if(lo<0){ lo=0; hi=width; } if(hi>REC_SECONDS){ hi=REC_SECONDS; lo=REC_SECONDS-width; }
        w.start=lo; w.end=hi;
      }
      positionWindowEl(el, w);
      updateMasks();
      if(w.id===activeWindowId) updateActiveFields();
    }
    function onUp(){
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      el.classList.remove('dragging');
      if(handleEl) handleEl.classList.remove('dragging');
      renderWindows(); // reconciles ordering/badges/list after the drag settles
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }
  el.querySelector('[data-h="left"]').addEventListener('mousedown', e=>{
    e.stopPropagation(); e.preventDefault();
    activeWindowId=w.id; applyActiveClasses();
    startDrag('left');
  });
  el.querySelector('[data-h="right"]').addEventListener('mousedown', e=>{
    e.stopPropagation(); e.preventDefault();
    activeWindowId=w.id; applyActiveClasses();
    startDrag('right');
  });
  el.addEventListener('mousedown', e=>{
    if(e.target.closest('.slice-handle') || e.target.closest('.slice-select-remove')) return;
    e.preventDefault();
    activeWindowId=w.id; applyActiveClasses();
    startDrag('move', e.clientX);
  });
}

function buildWindowEl(w, seq){
  const cut = isCut(w);
  const el=document.createElement('div');
  el.className='slice-select'+(w.id===activeWindowId?' active':'')+(cut?' cut':'');
  el.dataset.id=w.id;
  el.style.setProperty('--win-color', windowColor(w));
  positionWindowEl(el, w);
  el.innerHTML = `
    <span class="slice-select-badge" title="${cut?'removed span '+seq:'kept window '+seq}">${cut?'✂':seq}</span>
    <button class="slice-select-remove" title="Delete this ${cut?'removed span':'window'}">×</button>
    <div class="slice-handle" data-h="left"></div>
    <div class="slice-handle" data-h="right"></div>
  `;
  el.querySelector('.slice-select-remove').addEventListener('click', e=>{
    e.stopPropagation(); removeWindow(w.id);
  });
  wireWindowDrag(el, w);
  return el;
}

function renderWindowsList(sorted){
  const list = document.getElementById('windowsList');
  const keeps = sortedKeeps(), cuts = sortedCuts();
  const segs = resolvedSegments();
  const parts = [];
  if(keeps.length) parts.push(`${keeps.length} kept`);
  if(cuts.length) parts.push(`${cuts.length} removed`);
  if(segs.length) parts.push(`→ ${segs.length} segment${segs.length===1?'':'s'} · ${fmtDuration(resolvedTotal())} saved`);
  document.getElementById('windowsCount').textContent = parts.join(' · ');
  if(sorted.length===0){
    list.innerHTML = '<div class="windows-empty">Nothing selected — click "⤢ Whole signal" to keep all of it, "+ Keep window" or "− Remove window", or drag directly on a lane above.</div>';
    return;
  }
  list.innerHTML = '';
  sorted.forEach(w=>{
    const cut = isCut(w);
    const seq = seqIndexOf(w.id);
    const row=document.createElement('div');
    row.className='window-row'+(w.id===activeWindowId?' active':'')+(cut?' cut':'');
    row.dataset.id=w.id;
    row.style.setProperty('--win-color', windowColor(w));
    row.innerHTML = `
      <span class="badge">${cut?'✂':seq}</span>
      <button class="win-kind" data-kind title="Switch this span between kept and removed">${cut?'remove':'keep'}</button>
      <span class="range">${fmtTime(w.start)}–${fmtTime(w.end)}</span>
      <span class="dur">${fmtDuration(w.end-w.start)}</span>
      <button data-remove>Delete</button>`;
    row.addEventListener('click', e=>{
      if(e.target.closest('button')) return;
      activeWindowId=w.id; applyActiveClasses();
    });
    row.querySelector('[data-kind]').addEventListener('click', e=>{
      e.stopPropagation();
      w.type = cut ? 'keep' : 'cut';
      activeWindowId = w.id;
      renderWindows();
    });
    row.querySelector('[data-remove]').addEventListener('click', e=>{
      e.stopPropagation(); removeWindow(w.id);
    });
    list.appendChild(row);
  });
  if(cuts.length && keeps.length===0){
    const note=document.createElement('div');
    note.className='windows-empty';
    note.style.textAlign='left';
    note.innerHTML = 'No keep windows — the <b>whole recording minus the removed spans</b> will be saved.';
    list.appendChild(note);
  }
}

function renderWindows(){
  sliceTrack.querySelectorAll('.slice-select').forEach(el=>el.remove());
  const sorted = sortedWindows();
  sorted.forEach(w=> sliceTrack.appendChild(buildWindowEl(w, seqIndexOf(w.id))));
  renderWindowsList(sorted);
  updateActiveFields();
  updateMasks();
  // already covering everything → nothing left for "Whole signal" to widen
  const fullBtn = document.getElementById('fullWindowBtn');
  const isFull = windows.filter(w=>!isCut(w)).length===1 && windows.some(isFullSpan);
  fullBtn.disabled = isFull;
  fullBtn.title = isFull
    ? 'The whole recording is already kept'+(sortedCuts().length ? ' — minus the remove windows below.' : '.')
    : 'Keep the whole signal — one keep window spanning the entire recording. Any remove windows stay, so this means "everything except the removed spans".';
  document.getElementById('saveSequenceBtn').disabled = resolvedSegments().length===0;
  document.getElementById('saveSummary').classList.add('hidden');
}

function removeWindow(id){
  const idx = windows.findIndex(w=>w.id===id);
  if(idx>=0) windows.splice(idx,1);
  if(activeWindowId===id) activeWindowId = windows[0]?.id ?? null;
  renderWindows();
}

function addWindow(type='keep'){
  winSeq++;
  const id = 'w'+winSeq;
  let start, end;
  if(type==='cut'){
    // drop it in the middle of the widest span that currently survives, so the
    // inverse selection visibly does something the moment it appears
    const segs = resolvedSegments();
    const host = segs.length
      ? segs.reduce((a,b)=> (b.end-b.start > a.end-a.start ? b : a))
      : {start:0, end:REC_SECONDS};
    const width = Math.max(1.5, Math.min(6, (host.end-host.start)/3));
    const mid = (host.start+host.end)/2;
    start = clampSeconds(mid-width/2);
    end = clampSeconds(start+width);
  } else if(windows.length===0){
    start=4; end=22;
  } else {
    const lastEnd = Math.max(...windows.map(w=>w.end));
    const widest = windows.reduce((a,b)=> (b.end-b.start > a.end-a.start ? b : a));
    const width = Math.max(2, Math.min(18, widest.end-widest.start));
    start = Math.min(REC_SECONDS-2, lastEnd+2);
    end = Math.min(REC_SECONDS, start+width);
    if(end-start < 2){ end = REC_SECONDS; start = Math.max(0, end-width); }
  }
  windows.push({id, start, end, type});
  activeWindowId = id;
  renderWindows();
}
// ---- "Whole signal": one keep window covering the entire recording ----
// Anything shorter is a crop; this is the "I want all of it" shortcut, and the
// starting point for "everything except…" — remove windows are left alone so
// they keep cutting into the full span.
const FULL_SPAN_EPS = 0.001;
function isFullSpan(w){ return !isCut(w) && w.start<=FULL_SPAN_EPS && w.end>=REC_SECONDS-FULL_SPAN_EPS; }
function keepWholeSignal(){
  // several keep windows would merge into the full span anyway, so widen one and
  // drop the rest instead of leaving redundant rows in the list
  const cuts = windows.filter(isCut);
  let full = windows.find(w=>!isCut(w));
  if(full){ full.start = 0; full.end = REC_SECONDS; }
  else { winSeq++; full = {id:'w'+winSeq, start:0, end:REC_SECONDS, type:'keep'}; }
  windows = [full, ...cuts];
  activeWindowId = full.id;
  renderWindows();
}
document.getElementById('fullWindowBtn').addEventListener('click', keepWholeSignal);
document.getElementById('addWindowBtn').addEventListener('click', ()=> addWindow('keep'));
document.getElementById('addCutWindowBtn').addEventListener('click', ()=> addWindow('cut'));

document.getElementById('sliceMode').addEventListener('click', e=>{
  const btn = e.target.closest('button'); if(!btn) return;
  sliceMode = btn.dataset.mode;
  document.querySelectorAll('#sliceMode button').forEach(b=>b.classList.toggle('on', b===btn));
  if(sliceMode==='time'){ startLabel.textContent='Start (s)'; endLabel.textContent='End (s)'; }
  else { startLabel.textContent='Start (idx)'; endLabel.textContent='End (idx)'; }
  updateActiveFields();
});

startInput.addEventListener('input', ()=>{
  const w = windows.find(x=>x.id===activeWindowId); if(!w) return;
  const raw = parseFloat(startInput.value)||0;
  const sec = sliceMode==='index' ? raw/EIT_HZ : raw;
  w.start = clampSeconds(Math.min(sec, w.end-1));
  renderWindows();
});
endInput.addEventListener('input', ()=>{
  const w = windows.find(x=>x.id===activeWindowId); if(!w) return;
  const raw = parseFloat(endInput.value)||0;
  const sec = sliceMode==='index' ? raw/EIT_HZ : raw;
  w.end = clampSeconds(Math.max(sec, w.start+1));
  renderWindows();
});

