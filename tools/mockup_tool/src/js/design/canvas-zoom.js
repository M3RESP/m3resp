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
  // link labels get too small to read below 70 %. The detailed cards show
  // port names in the same font size, so there the labels stay down to 50 %.
  inner.classList.toggle('labels-off', canvasZoom < (NODE_VIEW==='compact' ? 0.7 : 0.5));
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
  const rects = [
    ...NODES.map(n=>({x:n.x, y:n.y, w:nodeWidth(n), h:nodeHeight(n)})),
    ...annotationRects(),
  ];
  if(rects.length===0) return {minX:0, minY:0, maxX:900, maxY:480};
  return {
    minX: Math.min(...rects.map(r=>r.x)),
    minY: Math.min(...rects.map(r=>r.y)),
    maxX: Math.max(...rects.map(r=>r.x+r.w)),
    maxY: Math.max(...rects.map(r=>r.y+r.h)),
  };
}
// maxZoom (optional) keeps a small graph from being blown up past that scale
function fitCanvasToView(maxZoom){
  const {minX,minY,maxX,maxY} = canvasContentBounds();
  const contentW = Math.max(1, maxX-minX), contentH = Math.max(1, maxY-minY);
  const pad = 28;
  const availW = Math.max(1, canvasScrollEl.clientWidth-pad*2);
  const availH = Math.max(1, canvasScrollEl.clientHeight-pad*2);
  const ceiling = Math.min(CANVAS_ZOOM_MAX, maxZoom || CANVAS_ZOOM_MAX);
  canvasZoom = Math.max(CANVAS_ZOOM_MIN, Math.min(ceiling, Math.min(availW/contentW, availH/contentH)));
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
document.getElementById('canvasZoomFitBtn').addEventListener('click', ()=> fitCanvasToView());
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
    ${runBadgeHtml(n)}
    <div class="run-bar"></div>
    <div class="node-head">
      <div class="node-head-row">
        ${nodeIconSvg(n, 14)}
        <span class="id" title="${n.label || n.id}">${n.isGroup?'▸ ':''}${n.label || n.id}</span>
        <button class="node-remove" title="${n.isGroup?'Remove this whole workflow':'Remove this node'}">×</button>
      </div>
      <span class="op" title="${n.op}">${n.op}</span>
    </div>
    ${portsHtml}
  `;
}

// ---- run state of a node: badge and progress (logic in js/design/run-state.js) ----
// node.status: 'pending' (not run), 'running', 'ok', 'warn', 'error',
// 'stale' (out of date: its settings or inputs changed after its last run)
const RUN_BADGE = {
  ok:    {d:'M5.5 10.5l3 3 6-7', title:'Done'},
  warn:  {d:'M10 5v6M10 14.5v.5', title:'Done, with a warning'},
  error: {d:'M6.5 6.5l7 7M13.5 6.5l-7 7', title:'Failed'},
  stale: {d:'M15 9a5 5 0 1 0-1.5 4.5M15 5v4h-4', title:'Out of date: run it again'},
};
function nodeStatus(n){ return (n.isGroup ? groupStatus(n.group) : n.status) || 'pending'; }
function runBadgeHtml(n){
  const b = RUN_BADGE[nodeStatus(n)];
  if(!b) return '<span class="run-badge"></span>';
  return `<span class="run-badge on" title="${b.title}"><svg width="12" height="12" viewBox="0 0 20 20" fill="none" aria-hidden="true">
    <path d="${b.d}" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span>`;
}

// one small line icon per step category, drawn in the modality colour
const NODE_ICONS = {
  loading:        'M10 3v9M6 8l4 4 4-4M4 13v4h12v-4',
  preprocessing:  'M3 4h14l-5.5 6.5V16l-3 1.5v-7z',
  detection:      'M2 15l3.5-6 3 4 3.5-9 3 8 3-3',
  roi:            'M10 3a7 7 0 1 0 0 14a7 7 0 1 0 0-14M10 3v14M3 10h14',
  baseline:       'M2 15h16M2 11c2.5-6 4.5-6 7 0s4.5 6 7 0',
  parameters:     'M15 4H5l5 6-5 6h10',
  features:       'M15 4H5l5 6-5 6h10',
  quality:        'M10 2l6 2.5v5c0 4-3 6.5-6 8-3-1.5-6-4-6-8v-5zM7 10l2 2 4-4',
  synchronization:'M3 7h12M12 4l3 3-3 3M17 13H5M8 10l-3 3 3 3',
  export:         'M10 12V3M6 7l4-4 4 4M4 12v5h12v-5',
  reducer:        'M3 4l7 7 7-7M10 11v6',
  sequence:       'M3 5h14v10H3zM7 5v10M13 5v10',
  workflow:       'M10 3l7 4-7 4-7-4zM3 11l7 4 7-4',
  other:          'M10 6a4 4 0 1 0 0 8a4 4 0 1 0 0-8',
};
function categoryIconSvg(cat, size){
  const d = NODE_ICONS[cat] || NODE_ICONS.other;
  return `<svg class="cnode-icon" width="${size}" height="${size}" viewBox="0 0 20 20" fill="none" aria-hidden="true">
    <path d="${d}" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
function nodeIconSvg(n, size){ return categoryIconSvg(nodeCategory(n), size || 26); }
// the light arc behind the dots of one side, so the input and output zones of
// the circle are visible before the mouse is over them
function compactArcPath(side, count){
  const step = count>1 ? Math.min(CPORT_MAX_STEP, CPORT_MAX_SPREAD/(count-1)) : 0;
  const half = ((count-1)*step/2 + 16)*Math.PI/180;
  const cx = CNODE_W/2, cy = CNODE_D/2, r = CNODE_PORT_R, sgn = side==='out' ? 1 : -1;
  const x1 = cx + sgn*Math.cos(half)*r, y1 = cy - Math.sin(half)*r;
  const x2 = cx + sgn*Math.cos(half)*r, y2 = cy + Math.sin(half)*r;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 0 ${side==='out'?1:0} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}
function nodeCompactHtml(n){
  let dots = '', arcs = '';
  [['in', n.ins], ['out', n.outs]].forEach(([side, list])=>{
    if(!list.length) return;
    arcs += `<path d="${compactArcPath(side, list.length)}"/>`;
    list.forEach(([name, type], i)=>{
      const p = compactPortOffset(side, i, list.length);
      dots += `<span class="port-dot cport" data-side="${side}" data-port="${name}" data-type="${type}" title="${side==='in'?'input':'output'} · ${name} · ${type}" style="left:${p.x.toFixed(1)}px;top:${p.y.toFixed(1)}px;--port-color:${tc(type)}"></span>`;
    });
  });
  // let long snake_case names wrap after an underscore
  const label = (n.label || n.id).replace(/_/g, '_<wbr>');
  return `
    <svg class="cnode-arcs" width="${CNODE_W}" height="${CNODE_D}" aria-hidden="true">${arcs}</svg>
    <div class="cnode-ring" title="${n.label || n.id} · ${n.op}">
      ${nodeIconSvg(n)}
      ${n.isGroup ? `<span class="cnode-count">${n.group.members.length}</span>` : ''}
      <svg class="run-progress" width="72" height="72" viewBox="0 0 72 72" aria-hidden="true"><circle cx="36" cy="36" r="34"/></svg>
      ${runBadgeHtml(n)}
      <button class="node-remove" title="${n.isGroup?'Remove this whole workflow':'Remove this node'}">×</button>
    </div>
    ${dots}
    <div class="cnode-label">${label}</div>
  `;
}

let suppressNodeClick = false;
function renderNode(n){
  const el = document.createElement('div');
  const compact = NODE_VIEW==='compact';
  el.className = 'node mod-'+n.mod+(n.isGroup?' group-node':'')+(compact?' compact':'');
  el.style.left = n.x+'px'; el.style.top = n.y+'px';
  el.dataset.id = n.id;
  el.dataset.status = nodeStatus(n);
  el.style.height = nodeHeight(n)+'px';
  el.innerHTML = compact ? nodeCompactHtml(n) : nodeHtml(n);
  // keep the selection mark when nodes are drawn again (the Inspector panel
  // records which node is selected)
  const inspEl = document.getElementById('inspectorBody');
  if(inspEl && inspEl.dataset.selected===n.id) el.classList.add('selected');
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
  // a double-click opens the node window (a closed workflow opens on one click)
  el.addEventListener('dblclick', e=>{
    if(n.isGroup || e.target.closest('.node-remove') || e.target.closest('.port-dot')) return;
    openNodeWindow(n.id);
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


// ---- "Compact | Detailed" switch in the canvas toolbar ----
function setNodeView(view){
  if(view===NODE_VIEW) return;
  NODE_VIEW = view;
  scaleLayout(view==='compact' ? COMPACT_SCALE : 1/COMPACT_SCALE);
  // cards and circles have other sizes, so check for overlaps again
  resolveAllOverlaps();
  document.querySelectorAll('#nodeViewSeg button').forEach(b=> b.classList.toggle('on', b.dataset.nodeview===view));
  renderAllNodes();
  drawEdges();
  refreshMultiSelectUI();
  fitCanvasToView(1);
}
document.getElementById('nodeViewSeg').addEventListener('click', e=>{
  const btn = e.target.closest('button[data-nodeview]');
  if(btn) setNodeView(btn.dataset.nodeview);
});
