// ================= Sidebar: modality overview, load/remove, add-custom =================
const BUILTIN_CARD_ID = {eit:'cardEit', emg:'cardEmg', vent:'cardVent'};
let customSeq = 0;
const customModalities = {}; // id -> {name}

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
    pill.textContent = 'loading…';
    pill.classList.remove('ok'); pill.classList.add('busy');
    setTimeout(()=>{
      pill.textContent = 'loaded';
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
  const rows = document.querySelectorAll('.mod-list-row');
  const loaded = document.querySelectorAll('.mod-list-row:not(.unloaded)').length;
  document.getElementById('modCount').textContent = `${loaded} loaded · ${rows.length} available`;
}

function setBuiltinLoaded(mod, loaded){
  const row = document.getElementById('modRow-'+mod);
  const card = document.getElementById(BUILTIN_CARD_ID[mod]);
  row.classList.toggle('unloaded', !loaded);
  row.querySelector('.st').textContent = loaded ? 'loaded' : 'not loaded';
  row.querySelector('.st').classList.toggle('on', loaded);
  const btn = row.querySelector('button');
  btn.textContent = loaded ? 'Remove' : 'Load';
  btn.className = loaded ? 'remove' : 'load';
  btn.dataset.action = loaded ? 'remove' : 'load';
  card.classList.toggle('hidden', !loaded);
  refreshModCount();
}

document.getElementById('modList').addEventListener('click', e=>{
  const btn = e.target.closest('button[data-action]');
  if(!btn) return;
  const mod = btn.dataset.mod;
  if(customModalities[mod]){
    // custom modality: remove deletes it entirely
    document.getElementById('modRow-'+mod)?.remove();
    document.getElementById('card-'+mod)?.remove();
    delete customModalities[mod];
    delete MASK_SVGS['mask-'+mod];
    refreshModCount();
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
  const id = nameToSlug(name) + '_' + (++customSeq);
  customModalities[id] = {name};

  // sidebar row
  const row = document.createElement('div');
  row.className = 'mod-list-row';
  row.id = 'modRow-'+id;
  row.dataset.mod = id;
  row.innerHTML = `
    <div class="swatch custom"></div>
    <div class="info"><div class="nm">${name}</div><div class="st on">loaded</div></div>
    <button class="remove" data-action="remove" data-mod="${id}">Remove</button>
  `;
  document.getElementById('modList').appendChild(row);

  // full preview card
  const maskId = 'mask-'+id;
  const card = document.createElement('div');
  card.className = 'card mod-card';
  card.id = 'card-'+id;
  card.innerHTML = `
    <div class="card-head">
      <div class="mod-head-left">
        <div class="mod-swatch" style="background:var(--custom);"></div>
        <div class="modal-info">
          <div class="name">${name} — custom modality</div>
          <div class="meta">${vendor} · 60.0 s · rate n/a</div>
        </div>
      </div>
      <div class="mod-head-right">
        <button class="load-file-btn" data-mod="${id}" title="Browse for a file to load into this modality">📁 Load file…</button>
        <input type="file" class="load-file-input" data-mod="${id}" style="display:none;">
        <span class="pill ok">loaded</span>
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
          <div class="card-head" style="padding:0 0 8px;border:none;"><h3 style="font-size:11.5px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.04em;">Channel setup</h3><span class="hint">unregistered step</span></div>
          <div class="field-grid">
            <div class="field"><label>Channels</label><input type="text" placeholder="e.g. 1"></div>
            <div class="field"><label>Sample rate</label><input type="text" placeholder="e.g. 100 Hz"></div>
          </div>
          <p class="slice-note" style="margin-top:10px;">No step in the registry reads this modality yet — add a <code>StepDefinition</code> for it before it can be wired into <b>2 · Design</b>.</p>
        </div>
      </div>
    </div>
  `;
  document.getElementById('customModalityCards').appendChild(card);
  wireCollapsible(card);
  wireLoadFileButton(card, id);

  const svgEl = card.querySelector(`#plot-${id} svg`);
  const seed = seedFromString(name);
  buildSignalSvg(svgEl, {stroke:'var(--custom)', freqHz:(14+seed%6)/60, amp:38+((seed*7)%26), baseline:93, noise:4, wobble:0.2, rectify:(seed%2===0), seed, h:172});
  MASK_SVGS[maskId] = svgEl;
  updateMasks();

  nameInput.value=''; vendorInput.value='';
  refreshModCount();
});

refreshModCount();

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
document.querySelectorAll('#prepMain > .card').forEach(wireCollapsible);
document.querySelectorAll('#view-check > .card, #view-results > .card').forEach(wireCollapsible);

