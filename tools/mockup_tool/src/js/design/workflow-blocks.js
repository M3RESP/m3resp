// ---- "Workflows" tab: predefined multi-step blocks built from real step ports ----
function opModality(op){
  const prefix = op.split('.')[0];
  if(prefix==='ventilator') return 'vent';
  if(prefix==='eit' || prefix==='emg') return prefix;
  return 'generic';
}
const PREDEFINED_BLOCKS = [
  {
    id:'emg-quality', name:'EMG quality checks', mod:'emg',
    desc:'Peaks → on/offset indices → SNR, bell-curve fit, and local-AUB quality flags.',
    ops:['emg.peak_indices','emg.onoffpeak_baseline_crossing','emg.snr_pseudo','emg.evaluate_bell_curve_error','emg.detect_local_high_aub'],
  },
  {
    id:'vent-pocc', name:'Ventilator Pocc analysis', mod:'vent',
    desc:'Occluded-breath detection feeding the Pocc time-product calculation.',
    ops:['ventilator.find_occluded_breaths','ventilator.pocc_time_product'],
  },
  {
    id:'eit-roi', name:'EIT ROI & lung masks', mod:'eit',
    desc:'Watershed lung segmentation, size-filtered, plus end-expiratory lung impedance (EELI).',
    ops:['eit.roi_watershed','eit.roi_filter_by_size','eit.eeli'],
  },
];
const REFERENCE_WORKFLOWS = [
  {name:'eit_full_preprocessing', desc:'Full single-modality EIT chain — examples/eit_full_preprocessing'},
  {name:'emg_full_preprocessing', desc:'Full single-modality EMG chain — examples/emg_full_preprocessing'},
  {name:'annemijn_multimodal', desc:'Real EIT + diaphragm sEMG + airway pressure recording — examples/annemijn_multimodal'},
];

// ---- "Workflows" tab: sections "Ready to insert" and "Reference only" ----
let blocksView = 'grid';
let blocksOpenSec = 'ready';
const blocksSearchEl = document.getElementById('blocksSearchInput');

function blockItemHtml(b){
  if(blocksView==='grid'){
    return `<div class="tb-tile" data-block="${b.id}" title="Click to add this workflow as one box">
      <span class="tb-icon mod-${b.mod}">${categoryIconSvg('workflow', 18)}</span>
      <span class="tb-name">${escHtml(b.name)}</span>
      <span class="tb-sub">${b.ops.length} steps</span>
    </div>`;
  }
  return `<div class="block-item" data-block="${b.id}" title="Click to add this workflow as one box">
    <div class="bname"><span class="block-swatches"><span style="background:var(--${b.mod})"></span></span>${escHtml(b.name)}<span class="bcount">${b.ops.length} steps</span></div>
    <div class="bdesc">${escHtml(b.desc)}</div>
    <div class="bsteps">${b.ops.map(o=>stepShortName(o)).join(' → ')}</div>
  </div>`;
}
function referenceItemHtml(w, i){
  if(blocksView==='grid'){
    return `<div class="tb-tile unavailable" data-ref="${i}" title="Full example pipeline, not in this mockup yet">
      <span class="tb-icon mod-generic">${categoryIconSvg('workflow', 18)}</span>
      <span class="tb-name">${escHtml(w.name).replace(/_/g, '_<wbr>')}</span>
      <span class="tb-sub">yaml</span>
    </div>`;
  }
  return `<div class="block-item unavailable" data-ref="${i}" title="Full example pipeline, not in this mockup yet">
    <div class="bname">${escHtml(w.name)}<span class="bcount">yaml</span></div>
    <div class="bdesc">${escHtml(w.desc)}</div>
  </div>`;
}

function renderBlocksTab(){
  const el = document.getElementById('blocksTab');
  if(!el) return;
  const q = blocksSearchEl.value.trim().toLowerCase();
  const hit = (...texts)=> !q || texts.some(t=>String(t).toLowerCase().includes(q));
  const wrap = items=> `<div class="${blocksView==='grid'?'tb-grid':'tb-list'}">${items}</div>`;
  const ready = PREDEFINED_BLOCKS.filter(b=>hit(b.name, b.desc, ...b.ops));
  const refs = REFERENCE_WORKFLOWS.map((w, i)=>[w, i]).filter(([w])=>hit(w.name, w.desc));
  const sections = [];
  if(ready.length) sections.push({key:'ready', label:'Ready to insert', dot:'var(--accent)', count:ready.length,
    body: wrap(ready.map(blockItemHtml).join(''))});
  if(refs.length) sections.push({key:'reference', label:'Reference only', dot:'var(--generic)', count:refs.length,
    body: wrap(refs.map(([w, i])=>referenceItemHtml(w, i)).join(''))});
  el.innerHTML = sections.length
    ? sections.map(sec=> paletteSectionHtml({...sec, open: !!q || sec.key===blocksOpenSec})).join('')
    : `<div class="pal-empty">No workflow matches “${escHtml(q)}”.</div>`;
}

const blocksTabEl = document.getElementById('blocksTab');
blocksTabEl.addEventListener('click', e=>{
  const head = e.target.closest('.pal-group-head[data-sec]');
  if(head){
    blocksOpenSec = nextOpenSection(blocksOpenSec, head.dataset.sec);
    blocksSearchEl.value = '';
    renderBlocksTab();
    return;
  }
  const item = e.target.closest('[data-block]');
  if(item) insertBlock(item.dataset.block);
});
blocksSearchEl.addEventListener('input', renderBlocksTab);
wireViewSeg('blocksViewSeg', view=>{ blocksView = view; renderBlocksTab(); });

function insertBlock(blockId){
  const block = PREDEFINED_BLOCKS.find(b=>b.id===blockId);
  if(!block) return;
  // the box goes to a free place under the graph; its steps lie left to
  // right from there, in the order they run
  const stepX = nodeWidth({}) + (NODE_VIEW==='compact' ? 60 : 40);
  const box = findFreeSpot(30, stagingTop(), nodeWidth({}), nodeHeight({ins:[[]], outs:[[]]}));
  const x = box.x, baseY = box.y;
  const added = [];
  block.ops.forEach((op, i)=>{
    const def = stepPorts(op);
    const n = {
      id: nextSlugId(op.split('.')[1]), op, mod: opModality(op),
      x: x + i*stepX, y: baseY,
      ins: def.ins.map(([nm,ty])=>[nm,ty]),
      outs: def.outs.map(([nm,ty])=>[nm,ty]),
      status:'pending',
    };
    NODES.push(n);
    added.push(n);
  });
  let wired = 0;
  added.forEach(n=> wired += autoWireNode(n)); // in order, so later steps see earlier ones too
  // the inserted steps stay one addressable unit — collapsed to a single box
  // that exposes only the ports crossing its boundary; click it to open it up
  const group = {
    id: nextSlugId(block.id.replace(/-/g,'_')),
    name: block.name, mod: block.mod,
    members: added.map(n=>n.id),
    collapsed: true, x, y: baseY, status:'pending',
  };
  GROUPS.push(group);
  renderAllNodes();
  drawEdges();
  const el = inner.querySelector(`.node[data-id="${group.id}"]`);
  if(el){
    el.scrollIntoView({behavior:'smooth', block:'center', inline:'center'});
    const {ins, outs} = computeGroupPorts(group);
    showEdgeNote(el, `Added "${block.name}" as one box — ${added.length} steps, ${ins.length} inputs / ${outs.length} outputs on its boundary, ${wired} auto-wired. Click the box to see inside.`);
  }
}

