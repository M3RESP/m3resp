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
}
function clearPortTargets(){
  inner.classList.remove('connecting');
  inner.querySelectorAll('.port-dot').forEach(dot=> dot.classList.remove('port-invalid','port-match'));
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
    dragPath.remove();
    clearPortTargets();
    const target = document.elementFromPoint(ev.clientX, ev.clientY);
    const dot = target && target.closest('.port-dot');
    if(dot){
      const targetNodeId = dot.closest('.node').dataset.id;
      const targetSide = dot.dataset.side, targetPort = dot.dataset.port;
      if(targetNodeId!==nodeId && targetSide!==side){
        // a port on a collapsed group belongs to one of its member steps — the
        // binding is recorded against that step, not against the box
        const outSide  = realEndpoint(...(side==='out' ? [nodeId, 'out', portName] : [targetNodeId, 'out', targetPort]));
        const inSide   = realEndpoint(...(side==='in'  ? [nodeId, 'in',  portName] : [targetNodeId, 'in',  targetPort]));
        EDGES.push({src:outSide.node, dst:inSide.node, key:inSide.port});
        drawEdges();
      }
    }
    connDrag = null;
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

renderAllNodes();
drawEdges();
renderBlocksTab();

// ---- toolbar: clear the board / reset to the default starting workflow ----
function clearBoard(){
  NODES.length = 0;
  EDGES.length = 0;
  GROUPS.length = 0;
  addSlotIndex = 0;
  clearMultiSelect();
  renderAllNodes();
  drawEdges();
  renderInspectorEmpty();
  showEdgeNote(document.getElementById('clearBoardBtn'), 'Canvas cleared — add operations from the palette or click Auto layout to bring the default workflow back.');
}
function resetToDefaultWorkflow(){
  NODES.length = 0;
  NODES.push(...JSON.parse(JSON.stringify(DEFAULT_NODES)));
  EDGES.length = 0;
  EDGES.push(...JSON.parse(JSON.stringify(DEFAULT_EDGES)));
  GROUPS.length = 0;
  addSlotIndex = 0;
  clearMultiSelect();
  renderAllNodes();
  drawEdges();
  renderInspectorEmpty();
  showEdgeNote(document.getElementById('autoLayoutBtn'), 'Reset to the default multimodal-full starting graph (15 steps).');
}
document.getElementById('clearBoardBtn').addEventListener('click', clearBoard);
document.getElementById('autoLayoutBtn').addEventListener('click', resetToDefaultWorkflow);

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
const NODE_META = {
  emg_ecg_gating: {
    summary:'Remove ECG peaks from EMG by gating (zero/interpolate/replace).',
    params:[
      {name:'source', type:'text', value:'filtered', unit:null, desc:"Which processed EMG variant to gate."},
      {name:'gate_width_seconds', type:'number', value:'0.05', unit:'s', desc:'Width of the window zeroed/replaced around each detected ECG peak.'},
      {name:'fill_method', type:'seg', options:['zero','interpolate','replace'], value:1, desc:'How the gated window is filled back in.'},
    ],
  },
  mdn_filter: {
    summary:'Apply an MDN heart-rate-removal filter to EIT data.',
    params:[
      {name:'label', type:'text', value:'filtered', unit:null, desc:'Label assigned to the filtered signal.'},
    ],
  },
  vent_detect_breaths: {
    summary:'Detect ventilator breaths from the ventilator volume channel.',
    params:[
      {name:'breath_width_seconds', type:'number', value:'0.5', unit:'s', desc:'Minimum breath width accepted by the detector.'},
    ],
  },
};
function selectNode(id){
  insp.dataset.selected = id;
  document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected', el.dataset.id===id));
  const n = nodeById(id);
  const isSequenceNode = n && n.op === 'prepare.sequence';
  const meta = NODE_META[id] || {
    summary: isSequenceNode ? 'Not a registry step — a sequence saved in 1 · Prepare\'s working window, concatenated from its selected time ranges. Any load step\'s "sequence" input can pull its own channel from it.' : '',
    params: [],
  };
  let paramsHtml = '';
  if(meta.params.length===0){
    paramsHtml = `<p style="font-size:11.5px;color:var(--text-faint);">No tunable parameters — audited and confirmed parameter-free.</p>`;
  } else {
    meta.params.forEach(p=>{
      if(p.type==='seg'){
        paramsHtml += `<div class="param-row"><label>${p.name}</label><div class="seg">${p.options.map((o,i)=>`<button class="${i===p.value?'on':''}">${o}</button>`).join('')}</div><div class="desc">${p.desc}</div></div>`;
      } else {
        paramsHtml += `<div class="param-row"><label>${p.name}${p.unit?`<span class="unit">${p.unit}</span>`:''}</label><input type="text" value="${p.value}"><div class="desc">${p.desc}</div></div>`;
      }
    });
  }
  let portsHtml = '<div class="insp-section-title">Inputs</div>';
  n.ins.forEach(([name,type])=> portsHtml += `<div class="port-line"><span>${name}</span><span class="type-chip">${type}</span></div>`);
  if(n.ins.length===0) portsHtml += `<div class="port-line" style="color:var(--text-faint);">none</div>`;
  portsHtml += '<div class="insp-section-title">Outputs</div>';
  n.outs.forEach(([name,type])=> portsHtml += `<div class="port-line"><span>${name}</span><span class="type-chip">${type}</span></div>`);

  insp.innerHTML = `
    <div class="insp-head">
      <span class="op">${n.op}</span>
      <h3>${n.id}</h3>
      <p>${meta.summary || 'Registered step — see docs/stage3.md for full description.'}</p>
    </div>
    <div class="insp-scroll">
      <div class="insp-section-title">Parameters</div>
      ${paramsHtml}
      ${portsHtml}
    </div>
  `;
}
renderInspectorEmpty();
selectNode('emg_ecg_gating');

