// ---- multi-select: Shift/Ctrl-click nodes or drag a box over empty canvas,
// then bulk-delete with the toolbar button or the Delete key ----
let multiSelected = new Set();
function toggleMultiSelected(id){
  if(multiSelected.has(id)) multiSelected.delete(id); else multiSelected.add(id);
  refreshMultiSelectUI();
}
function clearMultiSelect(){
  if(multiSelected.size===0) return;
  multiSelected.clear();
  refreshMultiSelectUI();
}
function refreshMultiSelectUI(){
  inner.querySelectorAll('.node').forEach(el=> el.classList.toggle('multi-picked', multiSelected.has(el.dataset.id)));
  const btn = document.getElementById('deleteSelectedBtn');
  if(!btn) return;
  btn.disabled = multiSelected.size===0;
  btn.textContent = multiSelected.size ? `🗑 Delete selected (${multiSelected.size})` : '🗑 Delete selected';
}
function deleteMultiSelected(){
  if(multiSelected.size===0) return;
  const ids = new Set(multiSelected);
  for(let i=EDGES.length-1;i>=0;i--){ if(ids.has(EDGES[i].src) || ids.has(EDGES[i].dst)) EDGES.splice(i,1); }
  for(let i=NODES.length-1;i>=0;i--){ if(ids.has(NODES[i].id)) NODES.splice(i,1); }
  const wasSelected = ids.has(insp.dataset.selected);
  multiSelected.clear();
  renderAllNodes();
  drawEdges();
  refreshMultiSelectUI();
  if(wasSelected) renderInspectorEmpty();
}
document.getElementById('deleteSelectedBtn').addEventListener('click', deleteMultiSelected);

function canvasPoint(e){
  // inner is CSS-scaled by canvasZoom, but node coordinates (and the selection
  // box drawn in the same logical space) are not — divide out the zoom so a
  // client-space mouse position lands on the right logical point.
  const rect = inner.getBoundingClientRect();
  return {x: (e.clientX-rect.left)/canvasZoom, y: (e.clientY-rect.top)/canvasZoom};
}
let selectionBoxEl = null;
inner.addEventListener('mousedown', e=>{
  if(e.button!==0) return;
  if(e.target.closest('.node') || e.target.closest('.port-dot')) return;
  const additive = e.shiftKey || e.ctrlKey || e.metaKey;
  if(!additive) clearMultiSelect();
  const before = new Set(multiSelected);
  const start = canvasPoint(e);
  if(!selectionBoxEl){
    selectionBoxEl = document.createElement('div');
    selectionBoxEl.className = 'selection-box';
    inner.appendChild(selectionBoxEl);
  }
  const box = selectionBoxEl;
  box.style.display = 'block';
  box.style.left = start.x+'px'; box.style.top = start.y+'px';
  box.style.width = '0px'; box.style.height = '0px';
  let moved = false;
  function onMove(ev){
    moved = true;
    const p = canvasPoint(ev);
    const x1=Math.min(start.x,p.x), y1=Math.min(start.y,p.y);
    const x2=Math.max(start.x,p.x), y2=Math.max(start.y,p.y);
    box.style.left=x1+'px'; box.style.top=y1+'px';
    box.style.width=(x2-x1)+'px'; box.style.height=(y2-y1)+'px';
    NODES.forEach(n=>{
      const nx2=n.x+NODE_W, ny2=n.y+nodeHeight(n);
      const intersects = n.x<x2 && nx2>x1 && n.y<y2 && ny2>y1;
      if(intersects) multiSelected.add(n.id);
      else if(before.has(n.id)) multiSelected.add(n.id);
      else multiSelected.delete(n.id);
    });
    refreshMultiSelectUI();
  }
  function onUp(){
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    box.style.display = 'none';
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
});
document.addEventListener('keydown', e=>{
  if(!document.getElementById('view-design').classList.contains('active')) return;
  if(document.activeElement && ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
  if((e.key==='Delete' || e.key==='Backspace') && multiSelected.size>0){
    e.preventDefault();
    deleteMultiSelected();
  } else if(e.key==='Escape'){
    clearMultiSelect();
  }
});

