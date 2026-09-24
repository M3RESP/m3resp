// ---- canvas zoom: buttons + fit-to-view, plus Ctrl/Cmd-scroll to zoom under the
// cursor. inner.x/inner.y (node coordinates) stay in one fixed "logical" unit
// system; only the CSS transform scales how big that looks on screen, so every
// place that turns a mouse position into a node coordinate has to divide the
// on-screen (client) delta by canvasZoom to land back in logical units. ----
let canvasZoom = 1;
const CANVAS_ZOOM_MIN = 0.25, CANVAS_ZOOM_MAX = 2;
const canvasScrollEl = document.getElementById('canvasScroll');
// A CSS transform does not shrink the element's layout box, so a zoomed-out
// canvas would still reserve its full unscaled width/height and leave a band of
// empty scrollable space to the right and below. Negative margins pull the
// footprint back to the size the content actually occupies on screen.
// Whatever space is then left over on an axis (the graph is wider than it is
// tall, so fitting by width usually leaves vertical slack) is split evenly above
// and below / left and right, so the workflow sits centred instead of hugging
// the top-left corner.
function syncCanvasFootprint(recentre){
  const logicalW = parseFloat(inner.style.width) || inner.offsetWidth;
  const logicalH = parseFloat(inner.style.height) || inner.offsetHeight;
  const z = canvasZoom;
  inner.style.marginRight = -(logicalW*(1-z))+'px';
  inner.style.marginBottom = -(logicalH*(1-z))+'px';

  // only on zoom/fit — recentring while nodes are being moved would make the
  // whole graph jump sideways every time one is dropped
  if(!recentre) return;
  const {minX,minY,maxX,maxY} = canvasContentBounds();
  const centreOffset = (avail, cMin, cMax, logical)=>{
    const wanted = (avail-(cMax-cMin)*z)/2 - cMin*z;   // centres the content bounds
    const room   = avail - logical*z;                  // don't create a scrollbar
    return Math.max(0, Math.min(wanted, room));
  };
  inner.style.marginLeft = centreOffset(canvasScrollEl.clientWidth, minX, maxX, logicalW)+'px';
  inner.style.marginTop  = centreOffset(canvasScrollEl.clientHeight, minY, maxY, logicalH)+'px';
}
function applyCanvasZoom(){
  inner.style.transform = 'scale('+canvasZoom+')';
  syncCanvasFootprint(true);
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
  const pad = 28;
  const availW = Math.max(1, canvasScrollEl.clientWidth-pad*2);
  const availH = Math.max(1, canvasScrollEl.clientHeight-pad*2);
  canvasZoom = Math.max(CANVAS_ZOOM_MIN, Math.min(CANVAS_ZOOM_MAX, Math.min(availW/contentW, availH/contentH)));
  applyCanvasZoom();
  // applyCanvasZoom() already centred whatever slack is left; scroll to the
  // content's top-left corner for the axis (if any) that still overflows.
  const ml = parseFloat(inner.style.marginLeft) || 0;
  const mt = parseFloat(inner.style.marginTop) || 0;
  canvasScrollEl.scrollLeft = Math.max(0, minX*canvasZoom+ml-pad);
  canvasScrollEl.scrollTop = Math.max(0, minY*canvasZoom+mt-pad);
}
document.getElementById('canvasZoomInBtn').addEventListener('click', ()=> canvasZoomBy(1.25));
document.getElementById('canvasZoomOutBtn').addEventListener('click', ()=> canvasZoomBy(0.8));
document.getElementById('canvasZoomFitBtn').addEventListener('click', fitCanvasToView);
window.addEventListener('resize', ()=> syncCanvasFootprint(true));
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
    portsHtml += '<div class="port-line">';
    // entry points (left, hollow) and exit points (right, filled) — the colour
    // is the artifact type, carried as --port-color so CSS decides fill vs ring
    if(n.ins[i]) portsHtml += `<div class="port-row" data-side="in" data-port="${n.ins[i][0]}"><span class="port-dot" data-side="in" data-port="${n.ins[i][0]}" data-type="${n.ins[i][1]}" title="input · ${n.ins[i][0]} · ${n.ins[i][1]}" style="--port-color:${tc(n.ins[i][1])}"></span><span title="${n.ins[i][0]} · ${n.ins[i][1]}">${n.ins[i][0]}</span></div>`; else portsHtml+='<div></div>';
    if(n.outs[i]) portsHtml += `<div class="port-row out" data-side="out" data-port="${n.outs[i][0]}"><span title="${n.outs[i][0]} · ${n.outs[i][1]}">${n.outs[i][0]}</span><span class="port-dot" data-side="out" data-port="${n.outs[i][0]}" data-type="${n.outs[i][1]}" title="output · ${n.outs[i][0]} · ${n.outs[i][1]}" style="--port-color:${tc(n.outs[i][1])}"></span></div>`; else portsHtml+='<div></div>';
    portsHtml += '</div>';
  }
  portsHtml += '</div>';
  return `
    <div class="status-dot ${n.status}"></div>
    <div class="node-head">
      <div class="node-head-row">
        <span class="id" title="${n.label || n.id}">${n.isGroup?'▸ ':''}${n.label || n.id}</span>
        <button class="node-remove" title="${n.isGroup?'Remove this whole workflow':'Remove this node'}">×</button>
      </div>
      <span class="op" title="${n.op}">${n.op}</span>
    </div>
    ${portsHtml}
  `;
}

let suppressNodeClick = false;
function renderNode(n){
  const el = document.createElement('div');
  el.className = 'node mod-'+n.mod+(n.isGroup?' group-node':'');
  el.style.left = n.x+'px'; el.style.top = n.y+'px';
  el.dataset.id = n.id;
  el.style.height = nodeHeight(n)+'px';
  el.innerHTML = nodeHtml(n);
  el.classList.toggle('multi-picked', multiSelected.has(n.id));
  el.addEventListener('click', e=>{
    if(e.target.closest('.node-remove') || e.target.closest('.port-dot')) return;
    if(suppressNodeClick){ suppressNodeClick=false; return; }
    if(n.isGroup){ setGroupCollapsed(n.group, false); return; }  // open the box
    if(e.shiftKey || e.ctrlKey || e.metaKey){
      toggleMultiSelected(n.id);
      return;
    }
    clearMultiSelect();
    selectNode(n.id);
  });
  el.querySelector('.node-remove').addEventListener('click', e=>{
    e.stopPropagation();
    if(n.isGroup) removeGroup(n.group); else removeNode(n.id);
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

