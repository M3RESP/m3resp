// ---- drag a node to reposition it; edges follow live ----
function startNodeDrag(n, el, e){
  const startX=e.clientX, startY=e.clientY;
  const origX=n.x, origY=n.y;
  let dragged=false;
  e.preventDefault();
  function onMove(ev){
    // client-space deltas need to be divided by canvasZoom to land back in
    // the same logical node-coordinate units n.x/n.y are stored in.
    const dx=(ev.clientX-startX)/canvasZoom, dy=(ev.clientY-startY)/canvasZoom;
    if(!dragged && Math.hypot(dx,dy)<3) return;
    if(!dragged){ dragged=true; el.classList.add('dragging'); }
    n.x = Math.max(0, origX+dx);
    n.y = Math.max(0, origY+dy);
    el.style.left = n.x+'px';
    el.style.top = n.y+'px';
    drawEdges();
  }
  function onUp(){
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    if(dragged){
      el.classList.remove('dragging');
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

function renderAllNodes(){
  inner.querySelectorAll('.node').forEach(el=>el.remove());
  NODES.forEach(renderNode);
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

function drawEdges(){
  svg.innerHTML = '';
  EDGES.forEach(edge=>{
    const {src,dst,key,srcKey,session}=edge;
    const sn = nodeById(src), dn = nodeById(dst);
    if(!sn || !dn) return;
    const oidx = sn.outs.findIndex(o=>o[0]===(srcKey||key));
    const iidx = dn.ins.findIndex(i=>i[0]===key);
    const sy = oidx>=0 ? portY(sn, oidx) : sn.y+HEAD_H/2;
    const dy = iidx>=0 ? portY(dn, iidx) : dn.y+HEAD_H/2;
    const sx = sn.x + NODE_W, dx = dn.x;
    const mx = (sx+dx)/2;
    const color = key==='session' ? 'var(--generic)' : tc((sn.outs[oidx]||[null,'m3session'])[1]);
    const d = `M ${sx} ${sy} C ${mx} ${sy}, ${mx} ${dy}, ${dx} ${dy}`;

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
    g.appendChild(hit); g.appendChild(path);
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

