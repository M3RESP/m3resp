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

function renderBlocksTab(){
  const el = document.getElementById('blocksTab');
  if(!el) return;
  let html = `<div class="pal-group-head open" style="cursor:default;"><span class="chip-dot" style="background:var(--accent)"></span>Ready to insert<span class="count">${PREDEFINED_BLOCKS.length}</span></div>`;
  PREDEFINED_BLOCKS.forEach(b=>{
    html += `<div class="block-item" data-block="${b.id}" title="Click to add this block to the canvas">
      <div class="bname"><span class="block-swatches"><span style="background:var(--${b.mod})"></span></span>${b.name}<span class="bcount">${b.ops.length} steps</span></div>
      <div class="bdesc">${b.desc}</div>
      <div class="bsteps">${b.ops.map(o=>o.split('.')[1]).join(' → ')}</div>
    </div>`;
  });
  html += `<div class="pal-group-head open" style="cursor:default;margin-top:8px;"><span class="chip-dot" style="background:var(--generic)"></span>Reference only<span class="count">${REFERENCE_WORKFLOWS.length}</span></div>`;
  REFERENCE_WORKFLOWS.forEach(w=>{
    html += `<div class="block-item unavailable" title="Full example pipeline — not yet insertable in this mockup">
      <div class="bname">${w.name}<span class="bcount">yaml</span></div>
      <div class="bdesc">${w.desc}</div>
    </div>`;
  });
  el.innerHTML = html;
  el.querySelectorAll('.block-item[data-block]').forEach(row=>{
    row.addEventListener('click', ()=> insertBlock(row.dataset.block));
  });
}

function insertBlock(blockId){
  const block = PREDEFINED_BLOCKS.find(b=>b.id===blockId);
  if(!block) return;
  const baseY = Math.max(...NODES.map(n=>n.y+nodeHeight(n)), 480) + 40;
  const col = addSlotIndex % 5; addSlotIndex++;
  const x = 30 + col*(NODE_W+30);
  const added = [];
  block.ops.forEach((op, i)=>{
    const def = STEP_DEFS[op] || {ins:[], outs:[]};
    const n = {
      id: nextSlugId(op.split('.')[1]), op, mod: opModality(op),
      x, y: baseY + i*140,
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

