// ---- Operations tab: every registry step, grouped by modality and then by
// category, as a grid of icons or as a list. ----
const paletteEl = document.getElementById('palette');
const paletteSearchEl = document.getElementById('paletteSearchInput');
let paletteView = 'grid';
let paletteOpenMod = 'eit';

paletteSearchEl.placeholder = `Search ${Object.keys(STEP_INFO).length} operations…`;

// steps that match the search text, as [{mod, cats:[{cat, ops:[...]}]}]
function paletteGroups(query){
  const q = query.trim().toLowerCase();
  return MODALITIES.map(m=>{
    const cats = CATEGORY_ORDER.map(cat=>{
      const ops = Object.keys(STEP_INFO).filter(op=>{
        const info = STEP_INFO[op];
        if(stepMod(op)!==m.mod || (info.category||'other')!==cat) return false;
        return !q || op.toLowerCase().includes(q) || info.summary.toLowerCase().includes(q) || cat.includes(q);
      });
      return {cat, ops};
    }).filter(c=>c.ops.length);
    return {...m, cats, count: cats.reduce((a, c)=>a+c.ops.length, 0)};
  }).filter(m=>m.count);
}

function paletteItemHtml(op, cat){
  const off = MOCK_UNAVAILABLE.has(op);
  const name = stepShortName(op);
  const attrs = `data-op="${op}" draggable="${off?'false':'true'}" title="${off?'Not available: an optional package is missing':'Click to add, or drag onto the canvas'}"`;
  if(paletteView==='grid'){
    return `<div class="tb-tile${off?' unavailable':''}" ${attrs}>
      <span class="tb-icon mod-${stepMod(op)}">${categoryIconSvg(cat, 18)}</span>
      <span class="tb-name">${name.replace(/_/g, '_<wbr>')}</span>
    </div>`;
  }
  return `<div class="pal-item${off?' unavailable':''}" ${attrs}>
    <span class="pal-name">${name}</span><span class="op">${op}</span><span class="add-hint">+ add</span>
  </div>`;
}

function renderPalette(){
  const query = paletteSearchEl.value;
  const groups = paletteGroups(query);
  if(!groups.length){
    paletteEl.innerHTML = `<div class="pal-empty">No step matches “${escHtml(query.trim())}”.</div>`;
    return;
  }
  const searching = query.trim().length>0;
  paletteEl.innerHTML = groups.map(g=> paletteSectionHtml({
    key: g.mod, label: g.label, dot: `var(--${g.mod})`, count: g.count,
    open: searching || g.mod===paletteOpenMod,
    body: g.cats.map(c=>`
      <div class="tb-cat-title">${c.cat}</div>
      <div class="${paletteView==='grid'?'tb-grid':'tb-list'}">${c.ops.map(op=>paletteItemHtml(op, c.cat)).join('')}</div>`).join(''),
  })).join('');
}

paletteEl.addEventListener('click', e=>{
  const head = e.target.closest('.pal-group-head[data-sec]');
  if(head){
    // one section open at a time; a click on the open one closes it
    paletteOpenMod = nextOpenSection(paletteOpenMod, head.dataset.sec);
    paletteSearchEl.value = '';
    renderPalette();
    return;
  }
  const item = e.target.closest('[data-op]');
  if(item && !MOCK_UNAVAILABLE.has(item.dataset.op)) addNodeFromPalette(item.dataset.op);
});
paletteEl.addEventListener('mouseover', e=>{
  const item = e.target.closest('[data-op]');
  if(item) showStepHelp(item.dataset.op);
});
paletteEl.addEventListener('dragstart', e=>{
  const item = e.target.closest('[data-op]');
  if(!item) return;
  e.dataTransfer.setData('text/m3resp-op', item.dataset.op);
  e.dataTransfer.effectAllowed = 'copy';
});
paletteSearchEl.addEventListener('input', renderPalette);
wireViewSeg('palViewSeg', view=>{ paletteView = view; renderPalette(); });

// a step dragged from the palette lands where it is dropped
canvasScrollEl.addEventListener('dragover', e=>{
  if(!e.dataTransfer.types.includes('text/m3resp-op')) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});
canvasScrollEl.addEventListener('drop', e=>{
  const op = e.dataTransfer.getData('text/m3resp-op');
  if(!op) return;
  e.preventDefault();
  addNodeFromPalette(op, canvasPoint(e));
});

renderPalette();
