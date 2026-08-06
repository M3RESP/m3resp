// ---- multiple selection windows: each {id, start, end} in seconds ----
let winSeq = 1;
let windows = [{id:'w1', start:4, end:22}];
let activeWindowId = 'w1';

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
function seqIndexOf(id){ return sortedWindows().findIndex(w=>w.id===id)+1; }
function windowColor(seq){ return WINDOW_COLORS[(seq-1)%WINDOW_COLORS.length]; }

const sliceTrack = document.getElementById('sliceTrack');
const startInput = document.getElementById('sliceStartInput');
const endInput = document.getElementById('sliceEndInput');
const startLabel = document.getElementById('sliceStartLabel');
const endLabel = document.getElementById('sliceEndLabel');
const durationChip = document.getElementById('sliceDuration');
const activeHint = document.getElementById('sliceActiveHint');

// custom modalities register their plot's <svg> here (see addCustomModBtn below)
// so updateMasks() can (re)draw one crop overlay per selected window into it.
const MASK_SVGS = {};

function updateMasks(){
  const sorted = sortedWindows();
  Object.values(MASK_SVGS).forEach(svgEl=>{
    if(!svgEl) return;
    svgEl.querySelectorAll('.slice-mask, .slice-mask-edge').forEach(el=>el.remove());
    const vb = (svgEl.getAttribute('viewBox')||'0 0 700 118').split(' ').map(Number);
    const h = vb[3];
    sorted.forEach((w,idx)=>{
      const color = windowColor(idx+1);
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
  activeHint.textContent = 'editing window '+seqIndexOf(w.id);
  if(sliceMode==='time'){ startInput.value=w.start.toFixed(1); endInput.value=w.end.toFixed(1); }
  else { startInput.value=Math.round(w.start*EIT_HZ); endInput.value=Math.round(w.end*EIT_HZ); }
  durationChip.textContent = 'active window: '+fmtDuration(w.end-w.start);
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
  const el=document.createElement('div');
  el.className='slice-select'+(w.id===activeWindowId?' active':'');
  el.dataset.id=w.id;
  el.style.setProperty('--win-color', windowColor(seq));
  positionWindowEl(el, w);
  el.innerHTML = `
    <span class="slice-select-badge">${seq}</span>
    <button class="slice-select-remove" title="Remove window ${seq}">×</button>
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
  const total = sorted.reduce((s,w)=>s+(w.end-w.start), 0);
  document.getElementById('windowsCount').textContent =
    sorted.length ? `${sorted.length} window${sorted.length===1?'':'s'} · ${fmtDuration(total)} total` : '';
  if(sorted.length===0){
    list.innerHTML = '<div class="windows-empty">No windows selected — click "+ Add window", or drag directly on a lane above.</div>';
    return;
  }
  list.innerHTML = '';
  sorted.forEach((w,idx)=>{
    const seq=idx+1;
    const row=document.createElement('div');
    row.className='window-row'+(w.id===activeWindowId?' active':'');
    row.dataset.id=w.id;
    row.style.setProperty('--win-color', windowColor(seq));
    row.innerHTML = `
      <span class="badge">${seq}</span>
      <span class="range">${fmtTime(w.start)}–${fmtTime(w.end)}</span>
      <span class="dur">${fmtDuration(w.end-w.start)}</span>
      <button data-remove>Remove</button>`;
    row.addEventListener('click', e=>{
      if(e.target.closest('button')) return;
      activeWindowId=w.id; applyActiveClasses();
    });
    row.querySelector('[data-remove]').addEventListener('click', e=>{
      e.stopPropagation(); removeWindow(w.id);
    });
    list.appendChild(row);
  });
}

function renderWindows(){
  sliceTrack.querySelectorAll('.slice-select').forEach(el=>el.remove());
  const sorted = sortedWindows();
  sorted.forEach((w,idx)=> sliceTrack.appendChild(buildWindowEl(w, idx+1)));
  renderWindowsList(sorted);
  updateActiveFields();
  updateMasks();
  document.getElementById('saveSequenceBtn').disabled = sorted.length===0;
  document.getElementById('saveSummary').classList.add('hidden');
}

function removeWindow(id){
  const idx = windows.findIndex(w=>w.id===id);
  if(idx>=0) windows.splice(idx,1);
  if(activeWindowId===id) activeWindowId = windows[0]?.id ?? null;
  renderWindows();
}

function addWindow(){
  winSeq++;
  const id = 'w'+winSeq;
  let start, end;
  if(windows.length===0){
    start=4; end=22;
  } else {
    const lastEnd = Math.max(...windows.map(w=>w.end));
    const widest = windows.reduce((a,b)=> (b.end-b.start > a.end-a.start ? b : a));
    const width = Math.max(2, Math.min(18, widest.end-widest.start));
    start = Math.min(REC_SECONDS-2, lastEnd+2);
    end = Math.min(REC_SECONDS, start+width);
    if(end-start < 2){ end = REC_SECONDS; start = Math.max(0, end-width); }
  }
  windows.push({id, start, end});
  activeWindowId = id;
  renderWindows();
}
document.getElementById('addWindowBtn').addEventListener('click', addWindow);

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

