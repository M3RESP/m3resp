// ---- node window: the settings of a step next to a preview of its output ----
// A double-click on a step, or "Open window ↗" in the Inspector, opens it, as
// a widget window in Orange. Several windows can be open at once; each one
// moves by its title bar. With "Apply automatically" on, a change goes to the
// step at once; with it off, changes wait for "Apply".
const NODE_WINDOWS = new Map();   // node id -> {el, draft, auto, out}
let nodeWindowZ = 400;

const STATUS_TEXT = {pending:'Not run', running:'Running…', ok:'Done', warn:'Done, with a warning', error:'Failed', stale:'Out of date'};

// outputs worth a preview (the session is shared by all steps)
function previewOuts(n){ return n.outs.filter(([, type])=> type!=='m3session'); }
function settingsSeed(n){ return JSON.stringify(n.params || {}); }

function nodeWindowInputsHtml(n){
  const ins = n.ins.filter(([name])=>name!=='session');
  if(!ins.length) return '<div class="nw-in none">This step reads only the session.</div>';
  return ins.map(([name, type])=>{
    const e = EDGES.find(x=>x.dst===n.id && x.key===name);
    const from = e ? `<b>${escHtml(e.src)}</b> · ${escHtml(e.srcKey || name)}` : '<span class="nw-missing">not linked</span>';
    return `<div class="nw-in"><span class="type-dot" style="background:${tc(type)}"></span><code>${escHtml(name)}</code><span class="nw-arrow">←</span>${from}</div>`;
  }).join('');
}
function nodeWindowPreviewHtml(n, st){
  const outs = previewOuts(n);
  if(!outs.length) return {html:'<div class="nw-empty">This step has no output to preview.</div>', summary:''};
  const [key, type] = outs.find(o=>o[0]===st.out) || outs[0];
  const pv = keyPreview(key, type, n.id, n.op, settingsSeed(n));
  const picker = outs.length>1
    ? `<div class="nw-outs">${outs.map(([k, t])=>`<button type="button" data-out="${escHtml(k)}" class="${k===key?'on':''}"><span class="type-dot" style="background:${tc(t)}"></span>${escHtml(k)}</button>`).join('')}</div>`
    : `<div class="nw-outs single"><span class="type-dot" style="background:${tc(type)}"></span><code>${escHtml(key)}</code><span class="nw-type">${escHtml(type)}</span></div>`;
  return {html: picker + `<div class="nw-plot">${pv.html}</div>`, summary: `${key}: ${pv.summary}`};
}

function renderNodeWindow(id){
  const st = NODE_WINDOWS.get(id), n = NODES.find(x=>x.id===id);
  if(!st || !n) return;
  const values = {...(n.params||{}), ...st.draft};
  st.el.innerHTML = `
    <div class="nw-head">
      <span class="tb-icon nw-icon mod-${n.mod}">${categoryIconSvg(nodeCategory(n), 15)}</span>
      <div class="nw-title"><b>${escHtml(n.id)}</b><code>${escHtml(n.op)}</code></div>
      <span class="nw-status"></span>
      <button class="nw-close" type="button" title="Close (Esc)">×</button>
    </div>
    <div class="nw-body">
      <div class="nw-side">
        <p class="nw-summary">${escHtml(nodeSummary(n))}</p>
        <div class="nw-section">Settings</div>
        <div class="nw-form">${paramFormHtml(n, values)}</div>
      </div>
      <div class="nw-main">
        <div class="nw-preview"></div>
        <div class="nw-section">Inputs</div>
        <div class="nw-inputs">${nodeWindowInputsHtml(n)}</div>
      </div>
    </div>
    <div class="nw-foot">
      <label class="pf-check"><input type="checkbox" class="nw-auto"${st.auto?' checked':''}> Apply automatically</label>
      <button class="tb-btn nw-apply" type="button">Apply</button>
      <span class="nw-info"></span>
    </div>`;
  renderNodeWindowPreview(id);
  updateNodeWindowStatus(id);
}
// the parts that change after an Apply or a run: preview, footer
function renderNodeWindowPreview(id){
  const st = NODE_WINDOWS.get(id), n = NODES.find(x=>x.id===id);
  if(!st || !n) return;
  const pv = nodeWindowPreviewHtml(n, st);
  st.el.querySelector('.nw-preview').innerHTML = pv.html;
  const pending = Object.keys(st.draft).length;
  const apply = st.el.querySelector('.nw-apply');
  apply.disabled = st.auto || !pending;
  apply.classList.toggle('primary', !st.auto && pending>0);
  st.el.querySelector('.nw-info').textContent = pending && !st.auto
    ? `${pending} change${pending===1?'':'s'} not applied`
    : pv.summary;
}
function updateNodeWindowStatus(id){
  const st = NODE_WINDOWS.get(id), n = NODES.find(x=>x.id===id);
  if(!st || !n) return;
  const el = st.el.querySelector('.nw-status');
  el.textContent = STATUS_TEXT[n.status||'pending'];
  el.dataset.status = n.status || 'pending';
}
// another view changed the node: draw the form and the preview again
function refreshNodeWindow(id){
  if(NODE_WINDOWS.has(id)) renderNodeWindow(id);
}

function applyNodeWindow(id){
  const st = NODE_WINDOWS.get(id);
  if(!st || !Object.keys(st.draft).length) return;
  const values = st.draft;
  st.draft = {};
  commitParams(id, values, 'win');
  renderNodeWindowPreview(id);
}
function focusNodeWindow(id){
  const st = NODE_WINDOWS.get(id);
  if(!st) return;
  st.el.style.zIndex = ++nodeWindowZ;
}
function closeNodeWindow(id){
  const st = NODE_WINDOWS.get(id);
  if(!st) return;
  st.el.remove();
  NODE_WINDOWS.delete(id);
}
// windows of steps that are no longer on the canvas close
function pruneNodeWindows(){
  [...NODE_WINDOWS.keys()].forEach(id=>{ if(!NODES.some(n=>n.id===id)) closeNodeWindow(id); });
}

function openNodeWindow(id){
  const n = NODES.find(x=>x.id===id);
  if(!n) return;
  if(NODE_WINDOWS.has(id)){ focusNodeWindow(id); return; }
  const el = document.createElement('div');
  el.className = 'node-window';
  el.tabIndex = -1;
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', `Settings of ${id}`);
  const st = {el, draft:{}, auto:true, out:(previewOuts(n)[0]||[])[0]};
  NODE_WINDOWS.set(id, st);
  document.body.appendChild(el);
  renderNodeWindow(id);
  // over the canvas; every next window a little lower and to the right
  const wrap = document.getElementById('canvasWrap').getBoundingClientRect();
  const k = (NODE_WINDOWS.size-1) % 6;
  const left = wrap.left + (wrap.width-el.offsetWidth)/2 + k*28;
  const top = wrap.top + 30 + k*28;
  el.style.left = Math.max(8, Math.min(left, window.innerWidth-el.offsetWidth-8))+'px';
  el.style.top = Math.max(8, Math.min(top, window.innerHeight-el.offsetHeight-8))+'px';
  focusNodeWindow(id);
  el.focus();

  el.addEventListener('mousedown', e=>{
    focusNodeWindow(id);
    const head = e.target.closest('.nw-head');
    if(!head || e.target.closest('button') || e.button!==0) return;
    // move the window by its title bar
    e.preventDefault();
    const sx = e.clientX - el.offsetLeft, sy = e.clientY - el.offsetTop;
    const move = ev=>{
      el.style.left = Math.max(0, Math.min(ev.clientX-sx, window.innerWidth-80))+'px';
      el.style.top = Math.max(0, Math.min(ev.clientY-sy, window.innerHeight-40))+'px';
    };
    const up = ()=>{ document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  });
  el.addEventListener('keydown', e=>{
    if(e.key==='Escape'){ e.stopPropagation(); closeNodeWindow(id); }
  });
  el.addEventListener('click', e=>{
    if(e.target.closest('.nw-close')){ closeNodeWindow(id); return; }
    if(e.target.closest('.nw-apply')){ applyNodeWindow(id); return; }
    const outBtn = e.target.closest('.nw-outs button[data-out]');
    if(outBtn){ st.out = outBtn.dataset.out; renderNodeWindowPreview(id); return; }
    onNodeWindowField(id, e);
  });
  el.addEventListener('change', e=>{
    if(e.target.classList.contains('nw-auto')){
      st.auto = e.target.checked;
      if(st.auto) applyNodeWindow(id); else renderNodeWindowPreview(id);
      return;
    }
    onNodeWindowField(id, e);
  });
}
function onNodeWindowField(id, e){
  const st = NODE_WINDOWS.get(id), n = NODES.find(x=>x.id===id);
  const r = st && n && e.target.closest('.nw-form') && readParamEvent(n, e);
  if(!r) return;
  st.draft[r[0]] = r[1];
  if(st.auto) applyNodeWindow(id); else renderNodeWindowPreview(id);
}
