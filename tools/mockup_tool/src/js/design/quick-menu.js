// ---- quick menu and link chooser: small lists that open at the mouse ----
// The quick menu lists steps to add. It opens on a double-click or a
// right-click on empty canvas, and when a link is dropped on empty canvas;
// from a link it lists only the steps that can take the link's data type.
// The link chooser opens when a link is dropped on a step with more than one
// port of the right type, and asks which port to use.
// Both lists work with the mouse and with the arrow keys, Enter and Escape.
let pickListEl = null;
let pickList = null;   // {items, active, onPick, onClose, getItems, query}

function ensurePickListEl(){
  if(pickListEl) return pickListEl;
  pickListEl = document.createElement('div');
  pickListEl.className = 'pick-list';
  pickListEl.tabIndex = -1;
  document.body.appendChild(pickListEl);
  pickListEl.addEventListener('mousemove', e=>{
    const row = e.target.closest('.pl-row');
    if(row && Number(row.dataset.i)!==pickList.active) setPickActive(Number(row.dataset.i));
  });
  pickListEl.addEventListener('click', e=>{
    const row = e.target.closest('.pl-row');
    if(row) choosePickItem(Number(row.dataset.i));
  });
  pickListEl.addEventListener('keydown', onPickListKey);
  // a press anywhere outside the list closes it
  document.addEventListener('mousedown', e=>{
    if(pickList && !pickListEl.contains(e.target)) closePickList();
  }, true);
  return pickListEl;
}

// opts: clientX, clientY, headHtml, search (placeholder text, or null for no
// search box), getItems(query) -> [{group, html, title, value}],
// emptyText(query), onPick(value), onClose()
function openPickList(opts){
  closePickList();
  const el = ensurePickListEl();
  pickList = {...opts, items:[], active:0, query:''};
  el.innerHTML = `
    ${opts.headHtml ? `<div class="pl-head">${opts.headHtml}</div>` : ''}
    ${opts.search!=null ? `<input class="pl-search" type="search" placeholder="${opts.search}">` : ''}
    <div class="pl-rows"></div>`;
  const input = el.querySelector('.pl-search');
  if(input) input.addEventListener('input', ()=>{ pickList.query = input.value; renderPickRows(); });
  renderPickRows();
  el.classList.add('on');
  // open at the mouse, but keep the whole list inside the window
  const w = el.offsetWidth, h = el.offsetHeight;
  el.style.left = Math.max(8, Math.min(opts.clientX+4, window.innerWidth-w-8))+'px';
  el.style.top  = Math.max(8, Math.min(opts.clientY+4, window.innerHeight-h-8))+'px';
  (input || el).focus();
}
function renderPickRows(){
  const rowsEl = pickListEl.querySelector('.pl-rows');
  pickList.items = pickList.getItems(pickList.query);
  pickList.active = 0;
  if(!pickList.items.length){
    rowsEl.innerHTML = `<div class="pl-empty">${pickList.emptyText(pickList.query)}</div>`;
    return;
  }
  let html = '', group = null;
  pickList.items.forEach((it, i)=>{
    if(it.group && it.group!==group){ group = it.group; html += `<div class="pl-group">${group}</div>`; }
    html += `<div class="pl-row${i===0?' active':''}" data-i="${i}" title="${escHtml(it.title||'')}">${it.html}</div>`;
  });
  rowsEl.innerHTML = html;
}
function setPickActive(i){
  if(!pickList || !pickList.items.length) return;
  pickList.active = (i + pickList.items.length) % pickList.items.length;
  pickListEl.querySelectorAll('.pl-row').forEach(r=> r.classList.toggle('active', Number(r.dataset.i)===pickList.active));
  const row = pickListEl.querySelector('.pl-row.active');
  if(row) row.scrollIntoView({block:'nearest'});
}
function onPickListKey(e){
  if(!pickList) return;
  if(e.key==='ArrowDown'){ e.preventDefault(); setPickActive(pickList.active+1); }
  else if(e.key==='ArrowUp'){ e.preventDefault(); setPickActive(pickList.active-1); }
  else if(e.key==='Enter'){ e.preventDefault(); choosePickItem(pickList.active); }
  else if(e.key==='Escape'){ e.preventDefault(); e.stopPropagation(); closePickList(); }
}
function choosePickItem(i){
  const it = pickList && pickList.items[i];
  if(!it) return;
  const onPick = pickList.onPick;
  closePickList();
  onPick(it.value);
}
function closePickList(){
  if(!pickList) return;
  const onClose = pickList.onClose;
  pickList = null;
  pickListEl.classList.remove('on');
  if(onClose) onClose();
}

// ---- quick menu ----
// does `op` have a port that can take the other end of a link from `from`?
function stepFitsLink(op, from){
  const ports = stepPorts(op);
  const list = from.side==='out' ? ports.ins : ports.outs;
  return list.some(p=> p[1]===from.type || p[1]==='any' || from.type==='any');
}
function quickMenuItems(from, query){
  const q = query.trim().toLowerCase();
  const fromMod = from ? (nodeById(from.nodeId)||{}).mod : null;
  const modIdx = op=> MODALITIES.findIndex(m=>m.mod===stepMod(op));
  const catIdx = op=> CATEGORY_ORDER.indexOf(STEP_INFO[op].category||'other');
  return Object.keys(STEP_INFO)
    .filter(op=> !MOCK_UNAVAILABLE.has(op))
    .filter(op=> !from || stepFitsLink(op, from))
    .filter(op=> !q || op.toLowerCase().includes(q) || STEP_INFO[op].summary.toLowerCase().includes(q))
    // from a link, steps of the same modality come first
    .sort((a, b)=>{
      const sa = from && stepMod(a)===fromMod ? 0 : 1, sb = from && stepMod(b)===fromMod ? 0 : 1;
      return sa-sb || modIdx(a)-modIdx(b) || catIdx(a)-catIdx(b) || a.localeCompare(b);
    })
    .map(op=>{
      const cat = STEP_INFO[op].category || 'other';
      return {
        group: (MODALITIES.find(m=>m.mod===stepMod(op))||{}).label,
        value: op,
        title: STEP_INFO[op].summary,
        html: `<span class="tb-icon pl-icon mod-${stepMod(op)}">${categoryIconSvg(cat, 13)}</span>
          <span class="pl-name">${stepShortName(op)}</span><span class="pl-meta">${cat}</span>`,
      };
    });
}

// opts: clientX, clientY, at (canvas point for the new step),
// from (optional {nodeId, side, portName, type}), dragPath (optional)
function openQuickMenu(opts){
  const {from} = opts;
  const typeHtml = from ? `<span class="type-dot" style="background:${tc(from.type)}"></span><code>${escHtml(from.type)}</code>` : '';
  openPickList({
    clientX: opts.clientX, clientY: opts.clientY,
    headHtml: from
      ? `Steps that ${from.side==='out'?'take':'make'} ${typeHtml}`
      : 'Add a step',
    search: 'Search steps…',
    getItems: q=> quickMenuItems(from, q),
    emptyText: q=> q.trim()
      ? `No step matches “${escHtml(q.trim())}”.`
      : `No step ${from && from.side==='out'?'takes':'makes'} ${typeHtml}.`,
    onPick: op=> addStepFromQuickMenu(op, opts),
    onClose: ()=>{ if(opts.dragPath) opts.dragPath.remove(); },
  });
}
function addStepFromQuickMenu(op, {at, from}){
  const n = addNodeFromPalette(op, at);
  if(!n || !from) return;
  // connect the link to a matching port of the new step, preferably one that
  // the automatic wiring left free
  const otherSide = from.side==='out' ? 'in' : 'out';
  const fits = matchingPorts(n, from.side, from.type);
  const pick = fits.find(p=> !portHasLink(n.id, otherSide, p[0])) || fits[0];
  if(pick) connectPorts(from.nodeId, from.side, from.portName, n.id, pick[0]);
}

// ---- link chooser ----
// opts: clientX, clientY, from {nodeId, side, portName, type}, host, fits
function openLinkChooser({clientX, clientY, from, host, fits}){
  const otherSide = from.side==='out' ? 'in' : 'out';
  openPickList({
    clientX, clientY,
    headHtml: `Connect <code>${escHtml(from.portName)}</code> to which ${otherSide==='in'?'input':'output'} of <b>${escHtml(host.label||host.id)}</b>?`,
    search: null,
    getItems: ()=> fits.map(([name, type])=>({
      value: name,
      title: `${name} · ${type}`,
      html: `<span class="type-dot" style="background:${tc(type)}"></span><span class="pl-name">${escHtml(name)}</span>
        ${portHasLink(host.id, otherSide, name) ? '<span class="pl-meta">has a link</span>' : '<span class="pl-meta">free</span>'}`,
    })),
    emptyText: ()=> '',
    onPick: port=> connectPorts(from.nodeId, from.side, from.portName, host.id, port),
  });
}

// ---- open the quick menu from the canvas ----
function quickMenuFromCanvasEvent(e){
  if(e.target.closest('.node, .canvas-hud, .edge-hit, .edge-label, .group-frame-head, .annot-note, .annot-arrow')) return false;
  openQuickMenu({clientX:e.clientX, clientY:e.clientY, at:canvasPoint(e)});
  return true;
}
canvasScrollEl.addEventListener('dblclick', e=>{ quickMenuFromCanvasEvent(e); });
canvasScrollEl.addEventListener('contextmenu', e=>{ if(quickMenuFromCanvasEvent(e)) e.preventDefault(); });
