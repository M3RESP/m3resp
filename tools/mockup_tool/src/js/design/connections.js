// ---- drag from a port dot to another node's opposite-side port dot to connect ----
let connDrag = null;
function portCenter(nodeId, side, portName){
  const n = nodeById(nodeId); if(!n) return null;
  const idx = side==='in' ? n.ins.findIndex(p=>p[0]===portName) : n.outs.findIndex(p=>p[0]===portName);
  return portPoint(nodeId, side, portName, idx);
}
// resolve a boundary port on a collapsed group back to the member step it
// actually belongs to (a no-op for ordinary nodes)
function realEndpoint(nodeId, side, portName){
  const g = groupById(nodeId);
  if(!g) return {node:nodeId, port:portName};
  computeGroupPorts(g);
  const m = (side==='in' ? g.inMap : g.outMap)[portName];
  return m ? {node:m.node, port:m.port} : {node:nodeId, port:portName};
}
// While dragging: dim every port the connection cannot land on (same node, or
// the same side — a binding always runs exit → entry), and ring the ports whose
// artifact type matches the one being dragged.
function markPortTargets(nodeId, side, type){
  inner.classList.add('connecting');
  inner.querySelectorAll('.port-dot').forEach(dot=>{
    const host = dot.closest('.node');
    const valid = host && host.dataset.id!==nodeId && dot.dataset.side!==side;
    dot.classList.toggle('port-invalid', !valid);
    dot.classList.toggle('port-match', valid && dot.dataset.type===type);
  });
  // compact view: a link can also land on the circle itself, so ring every
  // node that has a port of the right type on the other side
  inner.querySelectorAll('.node').forEach(el=>{
    const host = nodeById(el.dataset.id);
    el.classList.toggle('drop-match', !!host && host.id!==nodeId && matchingPorts(host, side, type).length>0);
  });
}
function clearPortTargets(){
  inner.classList.remove('connecting');
  inner.querySelectorAll('.port-dot').forEach(dot=> dot.classList.remove('port-invalid','port-match'));
  inner.querySelectorAll('.node.drop-match').forEach(el=> el.classList.remove('drop-match'));
}
// ports on `host` that can take the other end of a link dragged from `side`
// with artifact type `type` (type 'any' fits everything)
function matchingPorts(host, side, type){
  const list = side==='out' ? host.ins : host.outs;
  return list.filter(p=> p[1]===type || p[1]==='any' || type==='any');
}
function portHasLink(nodeId, side, portName){
  const end = realEndpoint(nodeId, side, portName);
  return side==='in'
    ? EDGES.some(e=> e.dst===end.node && e.key===end.port)
    : EDGES.some(e=> e.src===end.node && (e.srcKey||e.key)===end.port);
}
// record one binding from an exit point to an entry point
function connectPorts(nodeId, side, portName, targetNodeId, targetPort){
  // a port on a collapsed group belongs to one of its member steps — the
  // binding is recorded against that step, not against the box
  const outSide = realEndpoint(...(side==='out' ? [nodeId, 'out', portName] : [targetNodeId, 'out', targetPort]));
  const inSide  = realEndpoint(...(side==='in'  ? [nodeId, 'in',  portName] : [targetNodeId, 'in',  targetPort]));
  // an input reads one piece of data, so a new link replaces the old one
  for(let i=EDGES.length-1;i>=0;i--){
    const e = EDGES[i];
    if(!e.session && e.dst===inSide.node && e.key===inSide.port) EDGES.splice(i,1);
  }
  EDGES.push({src:outSide.node, dst:inSide.node, key:inSide.port, srcKey:outSide.port});
  drawEdges();
  markStale(inSide.node);
}
function startConnectionDrag(nodeId, side, portName, e){
  connDrag = {nodeId, side, portName};
  const start = portCenter(nodeId, side, portName);
  const srcNode = nodeById(nodeId);
  const srcPort = srcNode && (side==='in' ? srcNode.ins : srcNode.outs).find(p=>p[0]===portName);
  markPortTargets(nodeId, side, srcPort && srcPort[1]);
  const dragPath = document.createElementNS('http://www.w3.org/2000/svg','path');
  dragPath.setAttribute('class','edge-drag');
  dragPath.setAttribute('stroke','var(--accent)');
  svg.appendChild(dragPath);
  const rect = inner.getBoundingClientRect();
  function update(mx,my){
    const x2=(mx-rect.left)/canvasZoom, y2=(my-rect.top)/canvasZoom;
    // same curve the finished edge will get, so the preview doesn't jump on drop
    dragPath.setAttribute('d', edgePathD(start.x, start.y, x2, y2));
  }
  update(e.clientX, e.clientY);
  function onMove(ev){ update(ev.clientX, ev.clientY); }
  function onUp(ev){
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    clearPortTargets();
    connDrag = null;
    const target = document.elementFromPoint(ev.clientX, ev.clientY);
    const dot = target && target.closest('.port-dot');
    const hostEl = target && target.closest('.node');
    const type = srcPort && srcPort[1];
    const onEmptyCanvas = !dot && !hostEl && target && target.closest('#canvasScroll') && !target.closest('.canvas-hud');
    if(onEmptyCanvas){
      // dropped on empty canvas: choose a step that can take this data; the
      // dashed line stays until the menu closes
      openQuickMenu({clientX:ev.clientX, clientY:ev.clientY, at:canvasPoint(ev),
        from:{nodeId, side, portName, type}, dragPath});
      return;
    }
    dragPath.remove();
    if(dot){
      const targetNodeId = dot.closest('.node').dataset.id;
      const targetSide = dot.dataset.side, targetPort = dot.dataset.port;
      if(targetNodeId!==nodeId && targetSide!==side){
        connectPorts(nodeId, side, portName, targetNodeId, targetPort);
      }
    } else if(hostEl && hostEl.dataset.id!==nodeId){
      // dropped on the node itself: one matching port is used at once; with
      // several, the link chooser asks which one
      const host = nodeById(hostEl.dataset.id);
      const fits = host ? matchingPorts(host, side, type) : [];
      const otherSide = side==='out' ? 'in' : 'out';
      if(fits.length===1){
        connectPorts(nodeId, side, portName, host.id, fits[0][0]);
      } else if(fits.length>1){
        openLinkChooser({clientX:ev.clientX, clientY:ev.clientY, from:{nodeId, side, portName, type}, host, fits});
      } else if(host){
        showEdgeNote(hostEl, `${host.label || host.id} has no ${otherSide==='in'?'input':'output'} of type ${type}.`);
      }
    }
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

renderAllNodes();
drawEdges();
renderBlocksTab();

// ---- toolbar: clear the board / load the default starting workflow ----
// (both are also choices on the welcome card, js/design/welcome.js)
function resetCanvasState(){
  addSlotIndex = 0;
  clearMultiSelect();
  renderAllNodes();
  drawEdges();
  renderInspectorEmpty();
  PROBLEMS = NODES.filter(n=>KNOWN_WARNINGS[n.id]).map(n=>({node:n.id, sev:'warn', ...KNOWN_WARNINGS[n.id]}));
  renderProblems();
}
function clearBoard(){
  NODES.length = 0;
  EDGES.length = 0;
  GROUPS.length = 0;
  ANNOTATIONS.length = 0;
  resetCanvasState();
  showEdgeNote(document.getElementById('clearBoardBtn'), 'Canvas cleared. Add steps from the palette, or open an example with Examples….');
}
function resetToDefaultWorkflow(){
  NODES.length = 0;
  NODES.push(...JSON.parse(JSON.stringify(DEFAULT_NODES)));
  EDGES.length = 0;
  EDGES.push(...JSON.parse(JSON.stringify(DEFAULT_EDGES)));
  GROUPS.length = 0;
  ANNOTATIONS.length = 0;
  if(NODE_VIEW==='compact') scaleLayout(COMPACT_SCALE);
  resetCanvasState();
  fitCanvasToView(1);
}
document.getElementById('clearBoardBtn').addEventListener('click', clearBoard);

// ---- inspector ----
const insp = document.getElementById('inspectorBody');

// ---- collapsible side panels (palette / inspector) — free up canvas width ----
// chevron pointing left ('l') or right ('r') — a lighter, more standard
// collapse affordance than the solid ▶/◀ triangles
function chevronSvg(dir){
  const d = dir==='l' ? 'M10 3 L5 8 L10 13' : 'M6 3 L11 8 L6 13';
  return `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <path d="${d}" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
function wireSidePanelCollapse(panelId, btnId, expandDir, collapseDir){
  const panel = document.getElementById(panelId);
  const btn = document.getElementById(btnId);
  btn.innerHTML = chevronSvg(collapseDir);
  btn.addEventListener('click', ()=>{
    const collapsed = panel.classList.toggle('collapsed');
    btn.innerHTML = chevronSvg(collapsed ? expandDir : collapseDir);
    btn.title = collapsed ? 'Expand this panel' : 'Collapse this panel';
  });
}
wireSidePanelCollapse('palettePanel', 'paletteCollapseBtn', 'r', 'l');
wireSidePanelCollapse('inspector', 'inspectorCollapseBtn', 'l', 'r');
function renderInspectorEmpty(){
  delete insp.dataset.selected;
  insp.innerHTML = `
    <div class="insp-head"><h3>Inspector</h3><p>Select a node on the canvas to view its parameters, ports, and description — generated entirely from the step registry (<code>StepParameter</code> / <code>StepArtifact</code>), no per-operation UI code.</p></div>
  `;
}
function selectNode(id){
  insp.dataset.selected = id;
  document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected', el.dataset.id===id));
  const n = nodeById(id);
  let portsHtml = '<div class="insp-section-title">Inputs</div>';
  n.ins.forEach(([name,type])=> portsHtml += `<div class="port-line"><span>${name}</span><span class="type-chip">${type}</span></div>`);
  if(n.ins.length===0) portsHtml += `<div class="port-line" style="color:var(--text-faint);">none</div>`;
  portsHtml += '<div class="insp-section-title">Outputs</div>';
  n.outs.forEach(([name,type])=> portsHtml += `<div class="port-line"><span>${name}</span><span class="type-chip">${type}</span></div>`);
  // the parts of the shared session data (M3Session) the step reads and writes
  const shared = n.isGroup ? {reads:[], writes:[]} : sharedData(n.op);
  if(shared.reads.length || shared.writes.length){
    portsHtml += '<div class="insp-section-title">Shared session data</div>';
    shared.reads.forEach(name=> portsHtml += `<div class="port-line" title="${escHtml(sharedDataText(name))}"><span>${name}</span><span class="type-chip">reads</span></div>`);
    shared.writes.forEach(name=> portsHtml += `<div class="port-line" title="${escHtml(sharedDataText(name))}"><span>${name}</span><span class="type-chip">writes</span></div>`);
  }

  insp.innerHTML = `
    <div class="insp-head">
      <span class="op">${n.op}</span>
      <h3>${n.id}</h3>
      <p>${escHtml(nodeSummary(n))}</p>
      ${n.isGroup ? '' : '<button class="insp-open-btn" type="button" title="Settings and a preview of the output (or double-click the step)">Open window ↗</button>'}
    </div>
    <div class="insp-scroll">
      <div class="insp-section-title">Parameters</div>
      ${paramFormHtml(n)}
      ${portsHtml}
    </div>
  `;
  const openBtn = insp.querySelector('.insp-open-btn');
  if(openBtn) openBtn.addEventListener('click', ()=> openNodeWindow(id));
}
renderInspectorEmpty();
selectNode('emg_ecg_gating');

