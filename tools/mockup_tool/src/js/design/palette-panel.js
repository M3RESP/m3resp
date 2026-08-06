// ---- add a node from the palette, placed in an open staging area below the diagram ----
let addSlotIndex = 0;
function nextSlugId(base){
  if(!nodeById(base)) return base;
  let i=2;
  while(nodeById(base+'_'+i)) i++;
  return base+'_'+i;
}
function addNodeFromPalette(op, shortLabel, mod){
  const def = STEP_DEFS[op] || {ins:[], outs:[]};
  const baseY = Math.max(...NODES.map(n=>n.y+nodeHeight(n)), 480) + 40;
  const col = addSlotIndex % 5, row = Math.floor(addSlotIndex/5);
  addSlotIndex++;
  const n = {
    id: nextSlugId(shortLabel),
    op, mod,
    x: 30 + col*(NODE_W+30),
    y: baseY + row*140,
    ins: def.ins.map(([name,type])=>[name,type]),
    outs: def.outs.map(([name,type])=>[name,type]),
    status: 'pending',
  };
  NODES.push(n);
  const wired = autoWireNode(n);
  renderAllNodes();
  drawEdges();
  selectNode(n.id);
  const el = inner.querySelector(`.node[data-id="${n.id}"]`);
  if(el){
    el.scrollIntoView({behavior:'smooth', block:'center', inline:'center'});
    if(wired>0) showEdgeNote(el, `Auto-wired ${wired} input${wired===1?'':'s'} to existing data on the canvas.`);
  }
}

// ---- palette tab switching (Available data / Operations / Workflows) ----
document.getElementById('palTabs').addEventListener('click', e=>{
  const btn = e.target.closest('.pal-tab'); if(!btn) return;
  const tab = btn.dataset.paltab;
  document.querySelectorAll('.pal-tab').forEach(b=>b.classList.toggle('active', b===btn));
  document.querySelectorAll('.pal-tab-panel').forEach(p=>p.classList.toggle('active', p.dataset.paltab===tab));
  // refresh on entry: sequences saved on 1 · Prepare since the last render, or
  // nodes added/removed while this tab wasn't visible, should always be current
  if(tab==='data') renderDataTab();
});

// ---- "Available data" tab: the context keys the CURRENT graph has actually produced ----
function renderDataTab(){
  const el = document.getElementById('dataTab');
  if(!el) return;
  const sequences = (typeof SAVED_SEQUENCES!=='undefined') ? SAVED_SEQUENCES : [];
  const producing = NODES.filter(n=>n.outs.length>0);
  let html='';

  if(sequences.length){
    html += `<div class="data-node-group">
      <div class="data-node-head"><span class="dot" style="background:var(--accent)"></span>From 1 · Prepare<span class="op">saved sequences</span></div>`;
    sequences.forEach(seq=>{
      const onCanvas = !!nodeById(seq.name);
      html += `<div class="data-row seq-data-row" data-seq="${seq.id}" title="${onCanvas?'Already on canvas — click to jump to it':'Click to load \''+seq.name+'\' as a source node'}">
        <span class="type-dot" style="background:var(--accent)"></span>
        <span class="key">${seq.name}${onCanvas?' ✓':''}</span>
        <span class="type">${seq.windows.length}w · ${fmtDuration(seq.total)}</span>
        <button class="seq-preview-btn" data-preview-seq="${seq.id}" title="Visualize this sequence">👁</button>
      </div>`;
    });
    html += `</div>`;
  }

  if(producing.length===0 && sequences.length===0){
    el.innerHTML = '<div class="pal-empty">No data yet — add a loading step, or save a sequence in 1 · Prepare.</div>';
    return;
  }
  producing.forEach(n=>{
    html += `<div class="data-node-group">
      <div class="data-node-head"><span class="dot" style="background:var(--${n.mod})"></span>${n.id}<span class="op">${n.op}</span></div>`;
    n.outs.forEach(([name,type])=>{
      html += `<div class="data-row" data-key="${name}" data-type="${type}" data-node="${n.id}" data-op="${n.op}" title="Click to visualize ${name} (${type})"><span class="type-dot" style="background:${tc(type)}"></span><span class="key">${name}</span><span class="type">${type}</span><span class="viz-hint">👁</span></div>`;
    });
    html += `</div>`;
  });
  el.innerHTML = html;
  el.querySelectorAll('.seq-data-row').forEach(row=>{
    row.addEventListener('click', e=>{
      if(e.target.closest('.seq-preview-btn')) return;
      loadSequenceAsNode(row.dataset.seq);
    });
  });
  el.querySelectorAll('.seq-preview-btn').forEach(btn=>{
    btn.addEventListener('click', e=>{
      e.stopPropagation();
      const seq = sequences.find(s=>s.id===btn.dataset.previewSeq);
      if(seq) openDataViz({kind:'sequence', seq});
    });
  });
  el.querySelectorAll('.data-row[data-key]').forEach(row=>{
    row.addEventListener('click', ()=> openDataViz({
      kind:'key', key:row.dataset.key, type:row.dataset.type, node:row.dataset.node, op:row.dataset.op,
    }));
  });
}

