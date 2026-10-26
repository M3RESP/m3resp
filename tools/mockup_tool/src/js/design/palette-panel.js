// ---- add a node from the palette, placed in an open staging area below the diagram ----
let addSlotIndex = 0;
function nextSlugId(base){
  if(!nodeById(base)) return base;
  let i=2;
  while(nodeById(base+'_'+i)) i++;
  return base+'_'+i;
}
// `at` (optional) is a canvas point, for a step dropped on the canvas; the
// node is centred on it. Without it, the node goes into a row of up to five
// under the graph.
let stagingRowY = 0;
function addNodeFromPalette(op, at){
  if(MOCK_UNAVAILABLE.has(op)) return null;
  const def = stepPorts(op);
  // where the step should go: at the drop point, else to the right of the
  // selected step (as in Orange), else in a row under the graph; then the
  // nearest free place to that point
  const w = nodeWidth({}), h = NODE_VIEW==='compact' ? nodeHeight({}) : nodeHeight({ins:def.ins, outs:def.outs});
  const gapX = NODE_VIEW==='compact' ? 60 : 40;
  const selId = document.getElementById('inspectorBody').dataset.selected;
  const sel = selId && !collapsedGroupOf(selId) ? NODES.find(n=>n.id===selId) : null;
  let want;
  if(at){
    want = {x: at.x - w/2, y: at.y - (NODE_VIEW==='compact' ? CNODE_D/2 : 30)};
  } else if(sel){
    want = {x: sel.x + nodeWidth(sel) + gapX, y: sel.y};
  } else {
    if(addSlotIndex % 5 === 0) stagingRowY = stagingTop();
    want = {x: 30 + (addSlotIndex % 5)*(w+gapX), y: stagingRowY};
    addSlotIndex++;
  }
  const {x, y} = findFreeSpot(want.x, want.y, w, h);
  const n = {
    id: nextSlugId(stepShortName(op)),
    op, mod: stepMod(op),
    x, y,
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
    el.classList.add('node-new');
    if(!at) el.scrollIntoView({behavior:'smooth', block:'center', inline:'center'});
    if(wired>0) showEdgeNote(el, `Auto-wired ${wired} input${wired===1?'':'s'} to existing data on the canvas.`);
  }
  return n;
}

// ---- tabs of the Steps panel (Operations / Workflows) ----
document.getElementById('palTabs').addEventListener('click', e=>{
  const btn = e.target.closest('.pal-tab'); if(!btn) return;
  const tab = btn.dataset.paltab;
  document.querySelectorAll('.pal-tab').forEach(b=>b.classList.toggle('active', b===btn));
  document.querySelectorAll('.pal-tab-panel').forEach(p=>p.classList.toggle('active', p.dataset.paltab===tab));
});

// ---- "Available data" panel, the lower half of the palette: the data the
// steps on the canvas make, and the sequences saved in 1 · Prepare. One section for the saved sequences and one
// per modality; inside a modality, one sub-title per step that makes data. ----
let dataView = 'list';
let dataOpenSec = null;      // null until the first draw: then the first section
const dataSearchEl = document.getElementById('dataSearchInput');

function dataOutputItemHtml(n, name, type){
  const attrs = `data-key="${escHtml(name)}" data-type="${escHtml(type)}" data-node="${escHtml(n.id)}" data-op="${escHtml(n.op)}" title="Click to see ${escHtml(name)} (${escHtml(type)})"`;
  if(dataView==='grid'){
    return `<div class="tb-tile" ${attrs}>
      <span class="tb-icon type-icon" style="--mod:${tc(type)}">${typeIconSvg(type, 17)}</span>
      <span class="tb-name">${escHtml(name).replace(/_/g, '_<wbr>')}</span>
    </div>`;
  }
  return `<div class="data-row" ${attrs}><span class="type-dot" style="background:${tc(type)}"></span><span class="key">${escHtml(name)}</span><span class="type">${escHtml(type)}</span><span class="viz-hint">👁</span></div>`;
}
function dataSequenceItemHtml(seq){
  const onCanvas = !!nodeById(seq.name);
  const title = onCanvas ? 'Already on the canvas: click to go to it' : `Click to put '${escHtml(seq.name)}' on the canvas`;
  if(dataView==='grid'){
    return `<div class="tb-tile seq-tile" data-seq="${seq.id}" title="${title}">
      <span class="tb-icon type-icon" style="--mod:var(--accent)">${categoryIconSvg('sequence', 17)}</span>
      <span class="tb-name">${escHtml(seq.name)}${onCanvas?' ✓':''}</span>
      <button class="seq-preview-btn" data-preview-seq="${seq.id}" title="See this sequence">👁</button>
    </div>`;
  }
  return `<div class="data-row seq-data-row" data-seq="${seq.id}" title="${title}">
    <span class="type-dot" style="background:var(--accent)"></span>
    <span class="key">${escHtml(seq.name)}${onCanvas?' ✓':''}</span>
    <span class="type">${seq.windows.length}w · ${fmtDuration(seq.total)}</span>
    <button class="seq-preview-btn" data-preview-seq="${seq.id}" title="See this sequence">👁</button>
  </div>`;
}

function renderDataTab(){
  const el = document.getElementById('dataTab');
  if(!el) return;
  const sequences = (typeof SAVED_SEQUENCES!=='undefined') ? SAVED_SEQUENCES : [];
  const q = dataSearchEl.value.trim().toLowerCase();
  const hit = (...texts)=> !q || texts.some(t=>String(t).toLowerCase().includes(q));
  const wrap = items=> `<div class="${dataView==='grid'?'tb-grid':'tb-list'}">${items}</div>`;

  // the head of the panel shows how many items there are, also when it is closed
  const total = sequences.length + NODES.reduce((sum, n)=> sum + n.outs.length, 0);
  document.getElementById('dataPaneCount').textContent = total;

  const sections = [];
  const seqs = sequences.filter(seq=>hit(seq.name));
  if(seqs.length) sections.push({key:'prepare', label:'From 1 · Prepare', dot:'var(--accent)', count:seqs.length,
    body: wrap(seqs.map(dataSequenceItemHtml).join(''))});
  MODALITIES.forEach(m=>{
    let count = 0, body = '';
    NODES.filter(n=>n.mod===m.mod && n.outs.length).forEach(n=>{
      const outs = n.outs.filter(([name, type])=>hit(name, type, n.id));
      if(!outs.length) return;
      count += outs.length;
      body += `<div class="tb-cat-title">${escHtml(n.id)}</div>` + wrap(outs.map(([name, type])=>dataOutputItemHtml(n, name, type)).join(''));
    });
    if(count) sections.push({key:m.mod, label:m.label, dot:`var(--${m.mod})`, count, body});
  });

  if(!sections.length){
    el.innerHTML = q
      ? `<div class="pal-empty">No data matches “${escHtml(q)}”.</div>`
      : '<div class="pal-empty">No data yet. Add a loading step, or save a sequence in 1 · Prepare.</div>';
    return;
  }
  if(dataOpenSec===null) dataOpenSec = sections[0].key;
  el.innerHTML = sections.map(sec=> paletteSectionHtml({...sec, open: !!q || sec.key===dataOpenSec})).join('');
}

const dataTabEl = document.getElementById('dataTab');
dataTabEl.addEventListener('click', e=>{
  const head = e.target.closest('.pal-group-head[data-sec]');
  if(head){
    dataOpenSec = nextOpenSection(dataOpenSec, head.dataset.sec) || '';
    dataSearchEl.value = '';
    renderDataTab();
    return;
  }
  const sequences = (typeof SAVED_SEQUENCES!=='undefined') ? SAVED_SEQUENCES : [];
  const peek = e.target.closest('[data-preview-seq]');
  if(peek){
    const seq = sequences.find(s=>s.id===peek.dataset.previewSeq);
    if(seq) openDataViz({kind:'sequence', seq});
    return;
  }
  const seqItem = e.target.closest('[data-seq]');
  if(seqItem){ loadSequenceAsNode(seqItem.dataset.seq); return; }
  const item = e.target.closest('[data-key]');
  if(item) openDataViz({kind:'key', key:item.dataset.key, type:item.dataset.type, node:item.dataset.node, op:item.dataset.op});
});
dataSearchEl.addEventListener('input', renderDataTab);
wireViewSeg('dataViewSeg', view=>{ dataView = view; renderDataTab(); });
