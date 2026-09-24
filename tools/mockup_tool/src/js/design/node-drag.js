// ---- drag a node to reposition it; edges follow live ----
// Dragging a node that is part of a multi-selection moves the whole selection
// as one rigid group (relative offsets preserved); otherwise just that node.
function startNodeDrag(n, el, e){
  const startX=e.clientX, startY=e.clientY;
  let group;
  if(multiSelected.size>1 && multiSelected.has(n.id)){
    group = NODES.filter(node=>multiSelected.has(node.id)).map(node=>({
      node,
      el: inner.querySelector(`.node[data-id="${node.id}"]`),
      origX: node.x, origY: node.y,
    })).filter(g=>g.el);
  } else {
    group = [{node:n, el, origX:n.x, origY:n.y}];
  }
  // the group can only travel left/up until its topmost-leftmost member hits 0
  const minX = Math.min(...group.map(g=>g.origX));
  const minY = Math.min(...group.map(g=>g.origY));
  // dragging a collapsed group carries its hidden members along, so expanding
  // it later puts the steps back where the box now sits
  const carried = n.isGroup
    ? n.group.members.map(id=>NODES.find(x=>x.id===id)).filter(Boolean).map(m=>({node:m, origX:m.x, origY:m.y}))
    : [];
  let dragged=false;
  e.preventDefault();
  function onMove(ev){
    // client-space deltas need to be divided by canvasZoom to land back in
    // the same logical node-coordinate units n.x/n.y are stored in.
    let dx=(ev.clientX-startX)/canvasZoom, dy=(ev.clientY-startY)/canvasZoom;
    if(!dragged && Math.hypot(dx,dy)<3) return;
    if(!dragged){ dragged=true; group.forEach(g=>g.el.classList.add('dragging')); }
    dx = Math.max(dx, -minX);
    dy = Math.max(dy, -minY);
    group.forEach(g=>{
      g.node.x = g.origX+dx;
      g.node.y = g.origY+dy;
      g.el.style.left = g.node.x+'px';
      g.el.style.top = g.node.y+'px';
      if(g.node.isGroup){ g.node.group.x = g.node.x; g.node.group.y = g.node.y; }
    });
    carried.forEach(m=>{ m.node.x = m.origX+dx; m.node.y = m.origY+dy; });
    drawEdges();
  }
  function onUp(){
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    if(dragged){
      group.forEach(g=>g.el.classList.remove('dragging'));
      suppressNodeClick = true;
      updateCanvasSize();
      // click (if any) consumes this synchronously; this is just a safety net
      // in case the drag ends without a click ever following (e.g. released off-node)
      setTimeout(()=>{ suppressNodeClick=false; }, 0);
    }
  }
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

// The transparent frame drawn around an expanded group: it shows the boundary
// the collapsed box stands for, and lists what crosses it.
function renderGroupFrame(g){
  const members = g.members.map(id=>NODES.find(x=>x.id===id)).filter(Boolean);
  if(members.length===0) return;
  const PAD = 26, TOP = 34;
  const x1 = Math.min(...members.map(m=>m.x)) - PAD;
  const y1 = Math.min(...members.map(m=>m.y)) - PAD - TOP;
  const x2 = Math.max(...members.map(m=>m.x+NODE_W)) + PAD;
  const y2 = Math.max(...members.map(m=>m.y+nodeHeight(m))) + PAD;
  const {ins, outs} = computeGroupPorts(g);
  const el = document.createElement('div');
  el.className = 'group-frame mod-'+g.mod;
  el.dataset.group = g.id;
  el.style.left=x1+'px'; el.style.top=y1+'px';
  el.style.width=(x2-x1)+'px'; el.style.height=(y2-y1)+'px';
  el.innerHTML = `
    <div class="group-frame-head">
      <button class="group-collapse" title="Collapse this workflow back into one box">▾</button>
      <span class="group-name">${g.name}</span>
      <span class="group-meta">${members.length} steps · ${ins.length} in · ${outs.length} out</span>
    </div>`;
  el.querySelector('.group-collapse').addEventListener('click', e=>{
    e.stopPropagation();
    setGroupCollapsed(g, true);
  });
  inner.appendChild(el);
}
function setGroupCollapsed(g, collapsed){
  if(collapsed){
    const members = g.members.map(id=>NODES.find(x=>x.id===id)).filter(Boolean);
    if(members.length){ g.x = Math.min(...members.map(m=>m.x)); g.y = Math.min(...members.map(m=>m.y)); }
  }
  g.collapsed = collapsed;
  multiSelected.clear();
  renderAllNodes();
  drawEdges();
  refreshMultiSelectUI();
}
function removeGroup(g){
  [...g.members].forEach(id=>{
    for(let i=EDGES.length-1;i>=0;i--){ if(EDGES[i].src===id || EDGES[i].dst===id) EDGES.splice(i,1); }
    const idx = NODES.findIndex(n=>n.id===id);
    if(idx>=0) NODES.splice(idx,1);
  });
  const gi = GROUPS.indexOf(g);
  if(gi>=0) GROUPS.splice(gi,1);
  renderAllNodes();
  drawEdges();
}

function renderAllNodes(){
  // members deleted individually (× on a node, or bulk delete) leave the group
  // behind — drop them from it, and drop the group once it is empty
  for(let i=GROUPS.length-1;i>=0;i--){
    GROUPS[i].members = GROUPS[i].members.filter(id=>NODES.some(n=>n.id===id));
    if(GROUPS[i].members.length===0) GROUPS.splice(i,1);
  }
  inner.querySelectorAll('.node, .group-frame').forEach(el=>el.remove());
  GROUPS.filter(g=>!g.collapsed).forEach(renderGroupFrame);   // behind the nodes
  NODES.forEach(n=>{ if(!collapsedGroupOf(n.id)) renderNode(n); });
  GROUPS.filter(g=>g.collapsed).forEach(g=> renderNode(groupVirtualNode(g)));
  updateCanvasSize();
  updateStepCountPill();
  if(typeof renderDataTab==='function') renderDataTab();
}

// Auto-wire a newly-added node's inputs to whichever existing node already
// produces a context key of the same name — this is exactly the positional,
// name-based binding the compiler itself uses (see the session-rail note on
// 2 · Design), just applied the moment a node lands on the canvas instead of
// left for the author to draw by hand. 'session' is skipped: it matches
// almost every node and would just clutter the canvas with edges that add
// no information beyond what the dashed session rail already shows.
function autoWireNode(newNode){
  let added = 0;
  newNode.ins.forEach(([inName])=>{
    if(inName==='session') return;
    const already = EDGES.some(e=>e.dst===newNode.id && e.key===inName);
    if(already) return;
    const source = NODES.find(n=> n.id!==newNode.id && n.outs.some(o=>o[0]===inName));
    if(source){
      EDGES.push({src:source.id, dst:newNode.id, key:inName});
      added++;
    }
  });
  return added;
}

// Which dot an edge end actually hangs off. `key` is the destination input name
// and `srcKey` the source output name, but the two ends of a binding are often
// named differently and a step can be swapped for one with other port names, so
// don't give up when the name is unknown: fall back to the only port carrying
// the same artifact type, then to the first port. Anything is better than -1,
// which drops the end in the middle of the node header.
function resolvePortIdx(node, side, preferred, type){
  const list = side==='out' ? node.outs : node.ins;
  if(!list.length) return -1;
  let i = list.findIndex(p=>p[0]===preferred);
  if(i<0 && type) i = list.findIndex(p=>p[1]===type);
  return i<0 ? 0 : i;
}
// The S-curve every edge (and the live connection drag) is drawn with: control
// points pushed out horizontally so a wire always leaves an exit point going
// right and arrives at an entry point going right, whatever the nodes' relative
// positions. The offset scales with the gap and is clamped, so short hops don't
// balloon and long ones don't flatten into a hard step.
function edgePathD(sx, sy, ex, ey){
  const span = ex - sx;
  if(span > 30){
    const c = Math.max(24, Math.min(160, span*0.5));
    return `M ${sx} ${sy} C ${sx+c} ${sy}, ${ex-c} ${ey}, ${ex} ${ey}`;
  }
  // target sits left of (or on top of) the source — bulge the control points
  // outward so the edge still loops around instead of collapsing into a flat
  // line running back through both nodes
  const off = Math.max(70, Math.min(180, Math.abs(ey-sy)/2 + 80));
  return `M ${sx} ${sy} C ${sx+off} ${sy}, ${ex-off} ${ey}, ${ex} ${ey}`;
}

function drawEdges(){
  svg.innerHTML = '';
  EDGES.forEach(edge=>{
    const {src,dst,key,srcKey,session}=edge;
    const sn = nodeById(src), dn = nodeById(dst);
    if(!sn || !dn) return;
    // an edge whose ends are both inside one collapsed group is hidden with it
    const from = edgeEndpoint(src, 'out', srcKey||key);
    const to   = edgeEndpoint(dst, 'in',  key);
    if(from.group && from.group===to.group) return;
    const fromNode = nodeById(from.id), toNode = nodeById(to.id);
    if(!fromNode || !toNode) return;
    const oidx = resolvePortIdx(fromNode, 'out', from.port);
    const srcType = (fromNode.outs[oidx]||[])[1];
    const iidx = resolvePortIdx(toNode, 'in', to.port, srcType);
    const sp = portPoint(from.id, 'out', (fromNode.outs[oidx]||[])[0], oidx);
    const dp = portPoint(to.id,  'in',  (toNode.ins[iidx]||[])[0],   iidx);
    const sx = sp.x, sy = sp.y, dx = dp.x, dy = dp.y;
    const color = key==='session' ? 'var(--generic)' : tc(srcType || 'm3session');
    // the arrow points at the input circle and stops just outside it, so the
    // ring stays readable; the tail starts under the filled output dot
    const tipX = dx - PORT_R, endX = tipX - ARROW_LEN;
    const d = edgePathD(sx, sy, endX, dy);

    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    const hit = document.createElementNS('http://www.w3.org/2000/svg','path');
    hit.setAttribute('d', d);
    hit.setAttribute('class', 'edge-hit'+(session?' session':''));
    if(session){
      // Session edges are derived facts about the engine (session_reads/writes),
      // not a binding the author drew — there is nothing here to "undo", so
      // deleting it would just make the dependency invisible, not gone. Explain
      // instead of allowing a delete that can't be reconstructed from the graph.
      const title = document.createElementNS('http://www.w3.org/2000/svg','title');
      title.textContent = 'Implicit session dependency — cannot be removed from the canvas. Click for why.';
      hit.appendChild(title);
      hit.addEventListener('click', ()=> showEdgeNote(hit,
        'Implicit session dependency — derived from session_reads/session_writes, not a drawn binding. Removing it here would hide the dependency without removing it, so this edge can\'t be deleted from the canvas.'));
    } else {
      hit.addEventListener('click', ()=> removeEdge(edge));
    }
    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d', d);
    path.setAttribute('class', 'edge-path'+(session?' session':''));
    path.setAttribute('stroke', color);
    // arrowhead at the input port — the only direction cue that survives when a
    // node sits left of the one feeding it
    const head = document.createElementNS('http://www.w3.org/2000/svg','path');
    head.setAttribute('class', 'edge-arrow'+(session?' session':''));
    head.setAttribute('d', `M ${endX} ${dy-ARROW_W} L ${tipX} ${dy} L ${endX} ${dy+ARROW_W} Z`);
    head.setAttribute('fill', color);
    g.appendChild(hit); g.appendChild(path); g.appendChild(head);
    svg.appendChild(g);
  });
}

function removeEdge(edge){
  const idx = EDGES.indexOf(edge);
  if(idx>=0) EDGES.splice(idx,1);
  drawEdges();
}

let edgeNoteTimer = null;
function showEdgeNote(nearEl, text){
  let note = document.getElementById('edgeNote');
  if(!note){
    note = document.createElement('div');
    note.id = 'edgeNote';
    note.className = 'edge-note';
    document.body.appendChild(note);
  }
  const bbox = nearEl.getBoundingClientRect();
  note.style.left = Math.round(bbox.left + bbox.width/2) + 'px';
  note.style.top = Math.round(bbox.top) + 'px';
  note.textContent = text;
  note.classList.add('on');
  clearTimeout(edgeNoteTimer);
  edgeNoteTimer = setTimeout(()=> note.classList.remove('on'), 4200);
}

function removeNode(id){
  for(let i=EDGES.length-1;i>=0;i--){
    if(EDGES[i].src===id || EDGES[i].dst===id) EDGES.splice(i,1);
  }
  const idx = NODES.findIndex(n=>n.id===id);
  if(idx>=0) NODES.splice(idx,1);
  const wasSelected = insp.dataset.selected===id;
  multiSelected.delete(id);
  renderAllNodes();
  drawEdges();
  refreshMultiSelectUI();
  if(wasSelected) renderInspectorEmpty();
}

