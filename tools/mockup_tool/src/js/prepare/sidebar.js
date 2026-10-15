// ================= Sidebar: modality overview, load/remove, add-custom =================
const BUILTIN_CARD_ID = {eit:'cardEit', emg:'cardEmg', vent:'cardVent'};
let customSeq = 0;
const customModalities = {}; // id -> {name, vendor}

// Real native file picker per modality card — lets the team see browsing/selecting
// a file per modality, even though this mockup has no backend to actually parse it.
function wireLoadFileButton(card, mod){
  const btn = card.querySelector(`.load-file-btn[data-mod="${mod}"]`);
  const input = card.querySelector(`.load-file-input[data-mod="${mod}"]`);
  if(!btn || !input) return;
  btn.addEventListener('click', e=>{ e.stopPropagation(); input.click(); });
  input.addEventListener('click', e=> e.stopPropagation());
  input.addEventListener('change', ()=>{
    const file = input.files[0];
    if(!file) return;
    const head = card.querySelector(':scope > .card-head');
    const pill = head.querySelector('.pill');
    const meta = head.querySelector('.meta');
    const sizeMb = (file.size/1_000_000).toFixed(file.size<1_000_000 ? 3 : 1);
    pill.textContent = 'Loading…';
    pill.classList.remove('ok'); pill.classList.add('busy');
    setTimeout(()=>{
      pill.textContent = 'Loaded';
      pill.classList.remove('busy'); pill.classList.add('ok');
      if(meta) meta.textContent = `${file.name} · ${sizeMb} MB · browsed just now`;
      showEdgeNote(btn, `Selected "${file.name}" — this mockup shows the file picker, not a real parse. Add a backend endpoint to actually load it into the session.`);
    }, 600);
  });
}
wireLoadFileButton(document.getElementById('cardEit'), 'eit');
wireLoadFileButton(document.getElementById('cardEmg'), 'emg');
wireLoadFileButton(document.getElementById('cardVent'), 'vent');

function refreshModCount(){
  const rows = document.querySelectorAll('#modList .mod-list-row');
  const loaded = document.querySelectorAll('#modList .mod-list-row:not(.unloaded)').length;
  document.getElementById('modCount').textContent = `${loaded} of ${rows.length}`;
}

// a removed built-in modality keeps its row, with a button to load it again
function setBuiltinLoaded(mod, loaded){
  const row = document.getElementById('modRow-'+mod);
  const card = document.getElementById(BUILTIN_CARD_ID[mod]);
  const name = row.querySelector('.nm').textContent;
  row.classList.toggle('unloaded', !loaded);
  const st = row.querySelector('.st');
  st.classList.toggle('on', loaded);
  st.title = loaded ? 'Loaded' : 'Not loaded';
  if(!row.dataset.file) row.dataset.file = row.querySelector('.file').textContent;
  row.querySelector('.file').textContent = loaded ? row.dataset.file : 'not loaded';
  const btn = row.querySelector('button[data-action]');
  btn.classList.toggle('remove', loaded);
  btn.dataset.action = loaded ? 'remove' : 'load';
  btn.dataset.icon = loaded ? 'close' : 'plus';
  btn.title = loaded ? `Remove ${name}` : `Load ${name} again`;
  btn.setAttribute('aria-label', btn.title);
  btn.innerHTML = '';
  placeIcons(row);
  card.classList.toggle('hidden', !loaded);
  refreshModCount();
  // a loaded modality opens; a removed one makes way for the next section
  if(loaded) showSection(card.id);
  else if(currentSectionId===card.id) showSection(sectionOrder()[0]);
  else showSection(currentSectionId);
}

document.getElementById('modList').addEventListener('click', e=>{
  const btn = e.target.closest('button[data-action]');
  if(!btn){
    const row = e.target.closest('.mod-list-row');
    if(row && !row.classList.contains('unloaded')) showSection(row.dataset.target);
    return;
  }
  const mod = btn.dataset.mod;
  if(customModalities[mod]){
    // custom modality: remove deletes it entirely
    const wasShown = currentSectionId==='card-'+mod;
    document.getElementById('modRow-'+mod)?.remove();
    document.getElementById('card-'+mod)?.remove();
    delete customModalities[mod];
    delete MASK_SVGS['mask-'+mod];
    refreshModCount();
    showSection(wasShown ? sectionOrder()[0] : currentSectionId);
    return;
  }
  setBuiltinLoaded(mod, btn.dataset.action==='load');
});

function nameToSlug(name){
  return 'custom_'+(name.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'') || (++customSeq));
}
function seedFromString(str){
  let s=0; for(let i=0;i<str.length;i++) s = (s*31+str.charCodeAt(i))>>>0;
  return (s%97)+3;
}

document.getElementById('addCustomModBtn').addEventListener('click', ()=>{
  const nameInput = document.getElementById('customModName');
  const vendorInput = document.getElementById('customModVendor');
  const name = nameInput.value.trim();
  if(!name) return;
  const vendor = vendorInput.value.trim() || 'custom file';
  addCustomModality(nameToSlug(name) + '_' + (++customSeq), name, vendor);
  nameInput.value=''; vendorInput.value='';
  closeAddModality();
});

// a custom modality: a row in the side panel and a card in the main column
function addCustomModality(id, name, vendor){
  customModalities[id] = {name, vendor};

  // sidebar row
  const row = document.createElement('div');
  row.className = 'mod-list-row';
  row.id = 'modRow-'+id;
  row.dataset.mod = id;
  row.dataset.target = 'card-'+id;
  row.title = `Show the ${name} section`;
  row.innerHTML = `
    <span class="swatch custom"></span>
    <div class="info"><div class="nm">${name}</div><div class="file">${vendor}</div></div>
    <span class="st on" title="Loaded"></span>
    <button class="tb-btn small icon quiet remove" data-action="remove" data-mod="${id}" data-icon="close" title="Remove ${name}" aria-label="Remove ${name}"></button>
  `;
  placeIcons(row);
  document.getElementById('modList').appendChild(row);

  // full preview card
  const maskId = 'mask-'+id;
  const card = document.createElement('div');
  card.className = 'card mod-card mod-custom';
  card.id = 'card-'+id;
  card.innerHTML = `
    <div class="card-head">
      <div class="mod-head-left">
        <div class="modal-info">
          <div class="name"><span class="mod-name">${name}</span><span class="mod-device">custom modality</span></div>
          <div class="meta">${vendor} · 60.0 s · rate n/a</div>
        </div>
      </div>
      <div class="mod-head-right">
        <span class="pill ok">Loaded</span>
        <button class="tb-btn small load-file-btn" data-mod="${id}" data-icon="folder" title="Browse for a file to load into this modality">Load file…</button>
        <input type="file" class="load-file-input" data-mod="${id}" style="display:none;">
      </div>
    </div>
    <div class="card-body">
      <div class="signal-row">
        <div class="signal-plot-wrap">
          <div class="signal-plot" id="plot-${id}">
            <svg viewBox="0 0 700 172" preserveAspectRatio="none"></svg>
            <span class="axis-label">${name.toLowerCase()} · raw</span>
            <span class="axis-label right">custom</span>
          </div>
        </div>
        <div class="aux-panel">
          <div class="card-head" style="padding:0 0 8px;border:none;"><h3 style="font-size:11.5px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.04em;">Channel setup<span class="help-tip" tabindex="0" data-icon="help" aria-label="Help: channel setup"><span class="help-pop end" role="tooltip"><span class="hp-p">No step in the registry reads this modality yet. Add a <code>StepDefinition</code> for it to use it in <b>2 · Design</b>.</span></span></span></h3><span class="hint">unregistered step</span></div>
          <div class="field-grid">
            <div class="field"><label>Channels</label><input type="text" placeholder="e.g. 1"></div>
            <div class="field"><label>Sample rate</label><input type="text" placeholder="e.g. 100 Hz"></div>
          </div>
        </div>
      </div>
    </div>
  `;
  document.getElementById('customModalityCards').appendChild(card);
  placeIcons(card);
  wireLoadFileButton(card, id);

  const svgEl = card.querySelector(`#plot-${id} svg`);
  const seed = seedFromString(name);
  buildSignalSvg(svgEl, {stroke:'var(--custom)', freqHz:(14+seed%6)/60, amp:38+((seed*7)%26), baseline:93, noise:4, wobble:0.2, rectify:(seed%2===0), seed, h:172});
  MASK_SVGS[maskId] = svgEl;
  updateMasks();
  refreshModCount();
  showSection(card.id);
}

// "+ Add modality" opens the small form under the list
const addModalityForm = document.getElementById('addModalityForm');
const addModalityOpenBtn = document.getElementById('addModalityOpenBtn');
function closeAddModality(){ addModalityForm.hidden = true; addModalityOpenBtn.hidden = false; }
addModalityOpenBtn.addEventListener('click', ()=>{
  addModalityForm.hidden = false;
  addModalityOpenBtn.hidden = true;
  document.getElementById('customModName').focus();
});
document.getElementById('addCustomModCancel').addEventListener('click', closeAddModality);
addModalityForm.addEventListener('keydown', e=>{
  if(e.key==='Escape') closeAddModality();
  if(e.key==='Enter') document.getElementById('addCustomModBtn').click();
});

refreshModCount();

// ================= Side panel: collapse, steps =================
// The main column shows one section at a time: a modality card, the
// synchronization or the working window. The side panel and the Back / Next
// buttons choose it.
wireSidePanelCollapse('prepSidebar', 'prepSidebarCollapseBtn', 'r', 'l');

const prepMainEl = document.getElementById('prepMain');
const stepsEl = document.getElementById('prepSteps');
const prepCrumb = document.getElementById('prepCrumb');
const prepBackBtn = document.getElementById('prepBackBtn');
const prepNextBtn = document.getElementById('prepNextBtn');
const prepNextHint = document.getElementById('prepNextHint');
let currentSectionId = null;

// the sections in step order: the loaded modalities, then steps 2 and 3
function sectionOrder(){
  const mods = [...prepMainEl.querySelectorAll('.mod-card')].filter(c=>!c.classList.contains('hidden')).map(c=>c.id);
  return [...mods, 'cardSync', 'cardWindow'];
}
function sectionLabel(id){
  if(id==='cardSync') return {num:2, step:'Synchronization'};
  if(id==='cardWindow') return {num:3, step:'Working window'};
  const card = document.getElementById(id);
  return {num:1, step:'Signals', sub: card ? card.querySelector('.mod-name').textContent : ''};
}
function sectionText(id){ const l = sectionLabel(id); return l.sub || `${l.num} ${l.step}`; }

function showSection(id){
  const order = sectionOrder();
  if(!order.includes(id)) id = order[0];
  currentSectionId = id;
  prepMainEl.querySelectorAll('.card').forEach(c=> c.classList.toggle('is-shown', c.id===id));
  const isMod = !['cardSync', 'cardWindow'].includes(id);
  stepsEl.querySelectorAll('.step-item').forEach(b=>
    b.classList.toggle('current', b.dataset.section===id || (isMod && b.dataset.section==='signals')));
  stepsEl.querySelectorAll('.mod-list-row').forEach(r=> r.classList.toggle('current', r.dataset.target===id));
  const l = sectionLabel(id), i = order.indexOf(id);
  prepCrumb.innerHTML = `<span class="step-num">${l.num}</span><span class="crumb-step">${l.step}</span>`
    + (l.sub ? `<span class="crumb-sep">›</span><span class="crumb-sub">${l.sub}</span>` : '')
    + `<span class="crumb-pos">${i+1} of ${order.length}</span>`;
  prepBackBtn.disabled = i===0;
  const next = order[i+1];
  prepNextBtn.textContent = next ? 'Next' : 'Continue to 2 · Design';
  prepNextHint.textContent = next ? 'Next: '+sectionText(next) : 'Saved sequences show on 2 · Design under Available data.';
  prepMainEl.scrollTop = 0;
}
// kept for callers that pass a card element
function showPrepSection(card){ if(card) showSection(card.id); }

stepsEl.addEventListener('click', e=>{
  const item = e.target.closest('.step-item');
  if(!item) return;
  showSection(item.dataset.section==='signals' ? sectionOrder()[0] : item.dataset.section);
});
prepBackBtn.addEventListener('click', ()=>{
  const order = sectionOrder(), i = order.indexOf(currentSectionId);
  if(i>0) showSection(order[i-1]);
});
prepNextBtn.addEventListener('click', ()=>{
  const order = sectionOrder(), i = order.indexOf(currentSectionId);
  if(i < order.length-1) showSection(order[i+1]);
  else goToView('design');
});
showSection('cardEit');

// ================= Collapsible sections =================
const CHEVRON_SVG = '<svg viewBox="0 0 16 16" width="14" height="14"><path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
function wireCollapsible(card){
  const head = card.querySelector(':scope > .card-head');
  if(!head || head.querySelector('.collapse-toggle')) return;
  head.classList.add('collapsible-head');
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'collapse-toggle';
  btn.setAttribute('aria-label', 'Collapse section');
  btn.innerHTML = CHEVRON_SVG;
  head.appendChild(btn);
  head.addEventListener('click', ()=> card.classList.toggle('collapsed'));
}
document.querySelectorAll('#view-check > .card, #view-results > .card').forEach(wireCollapsible);

