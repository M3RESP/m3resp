// ---- canvas zoom: buttons + fit-to-view, plus Ctrl/Cmd-scroll to zoom under the
// cursor. inner.x/inner.y (node coordinates) stay in one fixed "logical" unit
// system; only the CSS transform scales how big that looks on screen, so every
// place that turns a mouse position into a node coordinate has to divide the
// on-screen (client) delta by canvasZoom to land back in logical units. ----
let canvasZoom = 1;
const CANVAS_ZOOM_MIN = 0.25, CANVAS_ZOOM_MAX = 2;
const canvasScrollEl = document.getElementById('canvasScroll');
function applyCanvasZoom(){
  inner.style.transform = 'scale('+canvasZoom+')';
  const pill = document.getElementById('canvasZoomPill');
  if(pill) pill.textContent = Math.round(canvasZoom*100)+'%';
  const outBtn = document.getElementById('canvasZoomOutBtn');
  const inBtn = document.getElementById('canvasZoomInBtn');
  if(outBtn) outBtn.disabled = canvasZoom<=CANVAS_ZOOM_MIN+1e-6;
  if(inBtn) inBtn.disabled = canvasZoom>=CANVAS_ZOOM_MAX-1e-6;
}
function canvasZoomBy(factor, anchorClientX, anchorClientY){
  const oldZoom = canvasZoom;
  canvasZoom = Math.max(CANVAS_ZOOM_MIN, Math.min(CANVAS_ZOOM_MAX, canvasZoom*factor));
  if(canvasZoom===oldZoom) return;
  if(anchorClientX!=null){
    // keep the point under the cursor visually fixed while the scale changes
    const rect = canvasScrollEl.getBoundingClientRect();
    const localX = anchorClientX-rect.left+canvasScrollEl.scrollLeft;
    const localY = anchorClientY-rect.top+canvasScrollEl.scrollTop;
    const ratio = canvasZoom/oldZoom;
    canvasScrollEl.scrollLeft += localX*(ratio-1);
    canvasScrollEl.scrollTop += localY*(ratio-1);
  }
  applyCanvasZoom();
}
function canvasContentBounds(){
  if(NODES.length===0) return {minX:0, minY:0, maxX:900, maxY:480};
  return {
    minX: Math.min(...NODES.map(n=>n.x)),
    minY: Math.min(...NODES.map(n=>n.y)),
    maxX: Math.max(...NODES.map(n=>n.x+NODE_W)),
    maxY: Math.max(...NODES.map(n=>n.y+nodeHeight(n))),
  };
}
function fitCanvasToView(){
  const {minX,minY,maxX,maxY} = canvasContentBounds();
  const contentW = Math.max(1, maxX-minX), contentH = Math.max(1, maxY-minY);
  const pad = 50;
  const availW = Math.max(1, canvasScrollEl.clientWidth-pad*2);
  const availH = Math.max(1, canvasScrollEl.clientHeight-pad*2);
  canvasZoom = Math.max(CANVAS_ZOOM_MIN, Math.min(CANVAS_ZOOM_MAX, Math.min(availW/contentW, availH/contentH)));
  applyCanvasZoom();
  canvasScrollEl.scrollLeft = Math.max(0, minX*canvasZoom-(availW-contentW*canvasZoom)/2-pad);
  canvasScrollEl.scrollTop = Math.max(0, minY*canvasZoom-(availH-contentH*canvasZoom)/2-pad);
}
document.getElementById('canvasZoomInBtn').addEventListener('click', ()=> canvasZoomBy(1.25));
document.getElementById('canvasZoomOutBtn').addEventListener('click', ()=> canvasZoomBy(0.8));
document.getElementById('canvasZoomFitBtn').addEventListener('click', fitCanvasToView);
canvasScrollEl.addEventListener('wheel', e=>{
  if(!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  canvasZoomBy(e.deltaY>0 ? 0.9 : 1.1, e.clientX, e.clientY);
}, {passive:false});
applyCanvasZoom();

function nodeHtml(n){
  let portsHtml = '<div class="node-ports">';
  const rows = Math.max(n.ins.length, n.outs.length);
  for(let i=0;i<rows;i++){
    portsHtml += '<div style="display:flex;justify-content:space-between;">';
    if(n.ins[i]) portsHtml += `<div class="port-row" data-side="in" data-port="${n.ins[i][0]}"><span class="port-dot" data-side="in" data-port="${n.ins[i][0]}" style="background:${tc(n.ins[i][1])}"></span><span title="${n.ins[i][0]} · ${n.ins[i][1]}">${n.ins[i][0]}</span></div>`; else portsHtml+='<div></div>';
    if(n.outs[i]) portsHtml += `<div class="port-row out" data-side="out" data-port="${n.outs[i][0]}"><span title="${n.outs[i][0]} · ${n.outs[i][1]}">${n.outs[i][0]}</span><span class="port-dot" data-side="out" data-port="${n.outs[i][0]}" style="background:${tc(n.outs[i][1])}"></span></div>`; else portsHtml+='<div></div>';
    portsHtml += '</div>';
  }
  portsHtml += '</div>';
  return `
    <div class="status-dot ${n.status}"></div>
    <div class="node-head">
      <div class="node-head-row">
        <span class="id" title="${n.id}">${n.id}</span>
        <button class="node-remove" title="Remove this node">×</button>
      </div>
      <span class="op" title="${n.op}">${n.op}</span>
    </div>
    ${portsHtml}
  `;
}

let suppressNodeClick = false;
function renderNode(n){
  const el = document.createElement('div');
  el.className = 'node mod-'+n.mod;
  el.style.left = n.x+'px'; el.style.top = n.y+'px';
  el.dataset.id = n.id;
  el.style.height = nodeHeight(n)+'px';
  el.innerHTML = nodeHtml(n);
  el.classList.toggle('multi-picked', multiSelected.has(n.id));
  el.addEventListener('click', e=>{
    if(e.target.closest('.node-remove') || e.target.closest('.port-dot')) return;
    if(suppressNodeClick){ suppressNodeClick=false; return; }
    if(e.shiftKey || e.ctrlKey || e.metaKey){
      toggleMultiSelected(n.id);
      return;
    }
    clearMultiSelect();
    selectNode(n.id);
  });
  el.querySelector('.node-remove').addEventListener('click', e=>{
    e.stopPropagation();
    removeNode(n.id);
  });
  el.querySelectorAll('.port-dot').forEach(dot=>{
    dot.addEventListener('mousedown', e=>{ e.stopPropagation(); startConnectionDrag(n.id, dot.dataset.side, dot.dataset.port, e); });
  });
  el.addEventListener('mousedown', e=>{
    if(e.button!==0) return;
    if(e.target.closest('.node-remove') || e.target.closest('.port-dot')) return;
    if(e.shiftKey || e.ctrlKey || e.metaKey) return; // let the click handler toggle selection instead of dragging
    startNodeDrag(n, el, e);
  });
  inner.appendChild(el);
  return el;
}

