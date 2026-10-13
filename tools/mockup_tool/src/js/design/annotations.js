// ---- notes and arrows on the canvas, as the annotations of Orange ----
// "✎ Note" in the toolbar, then a click (or a drag for a bigger box) on the
// canvas: a note appears, ready for text. "↗ Arrow", then a drag: an arrow
// from the press to the release. Notes move by their top bar and resize at
// their lower right corner; arrows move as a whole or by their end points.
// Delete (or ×) removes the selected note or arrow. They are part of the
// canvas, so they move and zoom with it; "Arrange" leaves them in place.
const NOTE_W = 190, NOTE_H = 76, ARROW_HEAD = 11;
let annotMode = null;        // 'note' | 'arrow' while a toolbar button is on
let selectedAnnot = null;    // id of the selected note or arrow
let annotSeq = 0;

// arrows have their own layer, above the links and under the steps
const annotSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
annotSvg.setAttribute('class', 'annot-layer');
svg.after(annotSvg);

function annotById(id){ return ANNOTATIONS.find(a=>a.id===id); }

// ---- drawing ----
function arrowHeadPath(a){
  const ang = Math.atan2(a.y2-a.y1, a.x2-a.x1);
  const p = (dAng, len)=> `${(a.x2-Math.cos(ang+dAng)*len).toFixed(1)} ${(a.y2-Math.sin(ang+dAng)*len).toFixed(1)}`;
  return `M ${a.x2} ${a.y2} L ${p(0.45, ARROW_HEAD)} L ${p(-0.45, ARROW_HEAD)} Z`;
}
function renderArrows(){
  annotSvg.innerHTML = ANNOTATIONS.filter(a=>a.kind==='arrow').map(a=>{
    const sel = a.id===selectedAnnot;
    return `<g class="annot-arrow${sel?' selected':''}" data-id="${a.id}">
      <path class="annot-hit" d="M ${a.x1} ${a.y1} L ${a.x2} ${a.y2}"/>
      <path class="annot-line" d="M ${a.x1} ${a.y1} L ${a.x2} ${a.y2}"/>
      <path class="annot-head" d="${arrowHeadPath(a)}"/>
      ${sel ? `<circle class="annot-handle" data-end="1" cx="${a.x1}" cy="${a.y1}" r="6"/>
               <circle class="annot-handle" data-end="2" cx="${a.x2}" cy="${a.y2}" r="6"/>` : ''}
    </g>`;
  }).join('');
}
// notes keep their element, so a note being typed in keeps its focus
function renderNotes(){
  const live = new Set();
  ANNOTATIONS.filter(a=>a.kind==='note').forEach(a=>{
    live.add(a.id);
    let el = inner.querySelector(`.annot-note[data-id="${a.id}"]`);
    if(!el){
      el = document.createElement('div');
      el.className = 'annot-note';
      el.dataset.id = a.id;
      el.innerHTML = `
        <div class="annot-bar" title="Drag to move the note"><span>Note</span>
          <button type="button" class="annot-del" title="Remove the note">×</button></div>
        <div class="annot-text" contenteditable="true" data-placeholder="Type a note…"></div>`;
      el.querySelector('.annot-text').textContent = a.text;
      // notes go under the steps
      inner.insertBefore(el, inner.querySelector('.node'));
      noteResizeWatch.observe(el);
    }
    el.style.left = a.x+'px'; el.style.top = a.y+'px';
    el.style.width = a.w+'px'; el.style.height = a.h+'px';
    el.classList.toggle('selected', a.id===selectedAnnot);
  });
  inner.querySelectorAll('.annot-note').forEach(el=>{
    if(!live.has(el.dataset.id)){ noteResizeWatch.unobserve(el); el.remove(); }
  });
}
function renderAnnotations(){
  renderArrows();
  renderNotes();
}
// the size a user gives a note at its corner is kept
const noteResizeWatch = new ResizeObserver(entries=> entries.forEach(en=>{
  const a = annotById(en.target.dataset.id);
  if(!a) return;
  a.w = en.target.offsetWidth; a.h = en.target.offsetHeight;
}));

// ---- select, remove ----
function selectAnnot(id){
  selectedAnnot = id;
  renderAnnotations();
}
function removeAnnot(id){
  const i = ANNOTATIONS.findIndex(a=>a.id===id);
  if(i>=0) ANNOTATIONS.splice(i, 1);
  if(selectedAnnot===id) selectedAnnot = null;
  renderAnnotations();
  updateCanvasSize();
}

// ---- toolbar modes ----
function setAnnotMode(mode){
  annotMode = annotMode===mode ? null : mode;
  document.getElementById('noteBtn').classList.toggle('on', annotMode==='note');
  document.getElementById('arrowBtn').classList.toggle('on', annotMode==='arrow');
  canvasScrollEl.classList.toggle('annot-drawing', !!annotMode);
}
document.getElementById('noteBtn').addEventListener('click', ()=> setAnnotMode('note'));
document.getElementById('arrowBtn').addEventListener('click', ()=> setAnnotMode('arrow'));

// a press on the canvas while a mode is on draws a note or an arrow; this
// listener runs before the box selection of the canvas
canvasScrollEl.addEventListener('mousedown', e=>{
  if(!annotMode || e.button!==0 || e.target.closest('.canvas-hud')) return;
  e.preventDefault();
  e.stopPropagation();
  const mode = annotMode;
  const start = canvasPoint(e);
  const draft = mode==='arrow'
    ? {id:'arrow_'+(++annotSeq), kind:'arrow', x1:start.x, y1:start.y, x2:start.x, y2:start.y}
    : {id:'note_'+(++annotSeq), kind:'note', x:start.x, y:start.y, w:NOTE_W, h:NOTE_H, text:''};
  ANNOTATIONS.push(draft);
  selectedAnnot = draft.id;
  renderAnnotations();
  const move = ev=>{
    const p = canvasPoint(ev);
    if(mode==='arrow'){ draft.x2 = p.x; draft.y2 = p.y; }
    else if(Math.hypot(p.x-start.x, p.y-start.y) > 8){
      draft.x = Math.min(start.x, p.x); draft.y = Math.min(start.y, p.y);
      draft.w = Math.max(120, Math.abs(p.x-start.x)); draft.h = Math.max(50, Math.abs(p.y-start.y));
    }
    renderAnnotations();
  };
  const up = ()=>{
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    setAnnotMode(mode);    // one note or arrow per click on the button
    if(mode==='arrow' && Math.hypot(draft.x2-draft.x1, draft.y2-draft.y1) < 12){ removeAnnot(draft.id); return; }
    updateCanvasSize();
    if(mode==='note') inner.querySelector(`.annot-note[data-id="${draft.id}"] .annot-text`).focus();
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}, true);

// ---- move a note (by its bar), an arrow, or one end of an arrow ----
function dragAnnot(e, apply){
  e.preventDefault();
  e.stopPropagation();
  const start = canvasPoint(e);
  const move = ev=>{
    const p = canvasPoint(ev);
    apply(p.x-start.x, p.y-start.y);
    renderAnnotations();
  };
  const up = ()=>{
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    updateCanvasSize();
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}
inner.addEventListener('mousedown', e=>{
  const note = e.target.closest('.annot-note');
  if(note){
    if(selectedAnnot!==note.dataset.id) selectAnnot(note.dataset.id);
    if(e.target.closest('.annot-del') || !e.target.closest('.annot-bar')) return;
    const a = annotById(note.dataset.id), x0 = a.x, y0 = a.y;
    dragAnnot(e, (dx, dy)=>{ a.x = Math.max(0, x0+dx); a.y = Math.max(0, y0+dy); });
    return;
  }
  const arrow = e.target.closest('.annot-arrow');
  if(arrow){
    const a = annotById(arrow.dataset.id);
    const k = {x1:a.x1, y1:a.y1, x2:a.x2, y2:a.y2};
    const end = e.target.closest('.annot-handle');
    if(selectedAnnot!==a.id) selectAnnot(a.id);
    dragAnnot(e, (dx, dy)=>{
      if(!end || end.dataset.end==='1'){ a.x1 = k.x1+dx; a.y1 = k.y1+dy; }
      if(!end || end.dataset.end==='2'){ a.x2 = k.x2+dx; a.y2 = k.y2+dy; }
    });
    return;
  }
  // a press anywhere else on the canvas ends the selection
  if(selectedAnnot) selectAnnot(null);
});
inner.addEventListener('click', e=>{
  const del = e.target.closest('.annot-del');
  if(del) removeAnnot(del.closest('.annot-note').dataset.id);
});
inner.addEventListener('input', e=>{
  const text = e.target.closest('.annot-text');
  if(!text) return;
  const el = text.closest('.annot-note'), a = annotById(el.dataset.id);
  a.text = text.innerText;
  // the note grows when the text needs more room
  const need = el.querySelector('.annot-bar').offsetHeight + text.scrollHeight + 2;
  if(need > a.h){ a.h = need; el.style.height = a.h+'px'; updateCanvasSize(); }
});
// a note left empty goes away when the user leaves it
inner.addEventListener('focusout', e=>{
  const text = e.target.closest('.annot-text');
  if(!text) return;
  const a = annotById(text.closest('.annot-note').dataset.id);
  if(a && !a.text.trim()) removeAnnot(a.id);
});
document.addEventListener('keydown', e=>{
  if(!document.getElementById('view-design').classList.contains('active')) return;
  const typing = e.target.closest && e.target.closest('.annot-text, input, textarea, select');
  if(e.key==='Escape'){
    if(typing && e.target.closest('.annot-text')) e.target.blur();
    if(annotMode) setAnnotMode(annotMode);
    return;
  }
  if((e.key==='Delete' || e.key==='Backspace') && selectedAnnot && !typing){
    e.preventDefault();
    removeAnnot(selectedAnnot);
  }
});
