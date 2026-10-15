// ---- parameter form of a step, used by the Inspector and the node window ----
// The fields come from the step registry (STEP_INFO params): name, kind,
// default, unit, limits, choices, help text. The values a user sets are kept
// in node.params; a field left empty uses the registry default.

// parameters of a node, from the registry (a saved sequence has none)
function nodeParams(n){
  return (STEP_INFO[n.op] && STEP_INFO[n.op].params) || [];
}
function nodeSummary(n){
  if(n.op==='prepare.sequence') return 'A sequence saved in 1 · Prepare: the selected time windows of all modalities, put one after the other. A load step reads its own modality from it.';
  return (STEP_INFO[n.op] && STEP_INFO[n.op].summary) || '';
}
function paramValue(n, p){
  return n.params && Object.prototype.hasOwnProperty.call(n.params, p.name) ? n.params[p.name] : p.default;
}
function choiceLabel(c){ return c===null ? 'auto' : String(c); }

// one field; `values` holds the values to show (a draft in the node window)
function paramFieldHtml(p, value){
  const name = escHtml(p.name);
  const unit = p.unit ? `<span class="unit">${escHtml(p.unit)}</span>` : '';
  const def = p.default===null || p.default===undefined ? 'auto' : String(p.default);
  let control;
  if(p.choices && p.choices.length){
    const short = p.choices.length<=4 && p.choices.every(c=>choiceLabel(c).length<=12);
    control = short
      ? `<div class="seg" data-param="${name}">${p.choices.map((c, i)=>
          `<button type="button" data-choice="${i}" class="${c===value?'on':''}">${escHtml(choiceLabel(c))}</button>`).join('')}</div>`
      : `<select data-param="${name}">${p.choices.map((c, i)=>
          `<option value="${i}"${c===value?' selected':''}>${escHtml(choiceLabel(c))}</option>`).join('')}</select>`;
  } else if(p.type==='boolean'){
    control = `<label class="pf-check"><input type="checkbox" data-param="${name}"${value?' checked':''}> ${value?'on':'off'}</label>`;
  } else if(p.type==='number' || p.type==='integer'){
    control = `<input type="number" data-param="${name}" value="${value===null||value===undefined?'':value}"
      placeholder="${escHtml(def)}" step="${p.type==='integer'?1:'any'}"${p.min!=null?` min="${p.min}"`:''}${p.max!=null?` max="${p.max}"`:''}>`;
  } else {
    const shown = value===null || value===undefined ? '' : (typeof value==='object' ? JSON.stringify(value) : value);
    control = `<input type="text" data-param="${name}" value="${escHtml(shown)}" placeholder="${escHtml(def)}">`;
  }
  return `<div class="param-row" data-name="${name}">
    <label>${name}${unit}</label>${control}
    ${p.desc ? `<div class="desc">${escHtml(p.desc)}</div>` : ''}
  </div>`;
}
// the whole form; advanced parameters go into a closed section
function paramFormHtml(n, values){
  const params = nodeParams(n);
  if(!params.length) return '<p class="pf-none">This step has no settings.</p>';
  const val = p=> values && Object.prototype.hasOwnProperty.call(values, p.name) ? values[p.name] : paramValue(n, p);
  const basic = params.filter(p=>!p.advanced), adv = params.filter(p=>p.advanced);
  return basic.map(p=>paramFieldHtml(p, val(p))).join('') +
    (adv.length ? `<details class="pf-advanced"><summary>Advanced (${adv.length})</summary>${adv.map(p=>paramFieldHtml(p, val(p))).join('')}</details>` : '');
}

// Read the value of the field that changed. Returns [name, value], or null
// when the event did not come from a field. An empty field gives null (the
// registry default).
function readParamEvent(n, e){
  const params = nodeParams(n);
  const segBtn = e.type==='click' && e.target.closest('.seg[data-param] button');
  if(segBtn){
    const p = params.find(x=>x.name===segBtn.parentElement.dataset.param);
    if(!p || segBtn.classList.contains('on')) return null;
    segBtn.parentElement.querySelectorAll('button').forEach(b=>b.classList.toggle('on', b===segBtn));
    return [p.name, p.choices[Number(segBtn.dataset.choice)]];
  }
  const el = e.type==='change' && e.target.closest('[data-param]');
  if(!el || el.classList.contains('seg')) return null;
  const p = params.find(x=>x.name===el.dataset.param);
  if(!p) return null;
  if(el.tagName==='SELECT') return [p.name, p.choices[Number(el.value)]];
  if(el.type==='checkbox'){
    el.parentElement.lastChild.textContent = el.checked ? ' on' : ' off';
    return [p.name, el.checked];
  }
  const raw = el.value.trim();
  if(raw==='') return [p.name, null];
  if(p.type==='integer') return [p.name, parseInt(raw, 10)];
  if(p.type==='number') return [p.name, parseFloat(raw)];
  return [p.name, raw];
}

// store new values on the node and make it out of date; the other open view
// of the node ('insp' = Inspector, 'win' = node window) is drawn again
function commitParams(id, values, from){
  const n = NODES.find(x=>x.id===id);
  if(!n) return;
  n.params = {...(n.params||{}), ...values};
  const insp = document.getElementById('inspectorBody');
  if(from!=='insp' && insp.dataset.selected===id) selectNode(id);
  if(from!=='win') refreshNodeWindow(id);
  markStale(id);
}

// a change in the Inspector form
['change', 'click'].forEach(type=> document.getElementById('inspectorBody').addEventListener(type, e=>{
  const id = document.getElementById('inspectorBody').dataset.selected;
  const n = id && NODES.find(x=>x.id===id);
  const r = n && readParamEvent(n, e);
  if(r) commitParams(id, {[r[0]]: r[1]}, 'insp');
}));
