// ---- welcome card: ways to start on 2 · Design ----
// As the welcome screen of Orange, it opens on the first visit to 2 · Design
// (the user can switch that off) and with "Examples…" in the toolbar. The
// choices: an empty canvas, a pipeline file, an example, a recent file.
const WELCOME_KEY = 'm3resp-mockup-welcome';
// example entries for the Recent list of this mockup
const RECENT_FILES = [
  {name:'multimodal-full.pipeline.yaml', when:'2026-10-12', example:true},
  {name:'emg_quality_test.pipeline.yaml', when:'2026-10-09'},
];

// the "show at start" choice is kept in this browser only
function welcomeAtStart(){
  try{ return localStorage.getItem(WELCOME_KEY)!=='off'; }catch(e){ return true; }
}
function setWelcomeAtStart(on){
  try{ localStorage.setItem(WELCOME_KEY, on ? 'on' : 'off'); }catch(e){ /* not kept */ }
}

const welcomeEl = document.createElement('div');
welcomeEl.className = 'welcome';
welcomeEl.hidden = true;
document.getElementById('canvasWrap').appendChild(welcomeEl);

function welcomeItemHtml(act, title, text, icon, extra){
  return `<button type="button" class="wc-item" data-act="${act}"${extra||''}>
    <span class="wc-icon">${categoryIconSvg(icon, 16)}</span>
    <span class="wc-text"><b>${title}</b><span>${text}</span></span>
  </button>`;
}
function renderWelcome(){
  welcomeEl.innerHTML = `
    <div class="wc-card" role="dialog" aria-label="Start a workflow" tabindex="-1">
      <div class="wc-head">
        <h3>Start a workflow</h3>
        <button type="button" class="wc-close" title="Close (Esc)">×</button>
      </div>
      <div class="wc-cols">
        <div class="wc-col">
          <div class="wc-title">New</div>
          ${welcomeItemHtml('empty', 'Empty canvas', 'No steps; add them from the palette.', 'other')}
          ${welcomeItemHtml('open', 'Open YAML…', 'A pipeline file (<code>.pipeline.yaml</code>).', 'loading')}
          <input type="file" class="wc-file" accept=".yaml,.yml" hidden>
        </div>
        <div class="wc-col">
          <div class="wc-title">Examples</div>
          ${welcomeItemHtml('example', 'multimodal-full', 'EIT, EMG and ventilator together; 15 of its 45 steps.', 'workflow')}
          ${REFERENCE_WORKFLOWS.map(w=>welcomeItemHtml('none', escHtml(w.name), 'Not in this mockup yet.', 'workflow', ' disabled')).join('')}
        </div>
        <div class="wc-col">
          <div class="wc-title">Recent</div>
          ${RECENT_FILES.map((f, i)=>welcomeItemHtml('recent', escHtml(f.name), `Opened ${f.when}`, 'sequence', ` data-i="${i}"`)).join('')}
        </div>
      </div>
      <div class="wc-msg" hidden></div>
      <div class="wc-foot">
        <label class="pf-check"><input type="checkbox" class="wc-start"${welcomeAtStart()?' checked':''}> Show this card when 2 · Design opens</label>
      </div>
    </div>`;
}
function welcomeMessage(html){
  const msg = welcomeEl.querySelector('.wc-msg');
  msg.innerHTML = html;
  msg.hidden = false;
}
function openWelcome(){
  renderWelcome();
  welcomeEl.hidden = false;
  welcomeEl.querySelector('.wc-card').focus();
}
function closeWelcome(){ welcomeEl.hidden = true; }

welcomeEl.addEventListener('click', e=>{
  // a click next to the card closes it
  if(e.target===welcomeEl || e.target.closest('.wc-close')){ closeWelcome(); return; }
  const item = e.target.closest('.wc-item');
  if(!item || item.disabled) return;
  const act = item.dataset.act;
  if(act==='empty'){ closeWelcome(); clearBoard(); }
  else if(act==='example'){ closeWelcome(); resetToDefaultWorkflow(); }
  else if(act==='open'){ welcomeEl.querySelector('.wc-file').click(); }
  else if(act==='recent'){
    const f = RECENT_FILES[Number(item.dataset.i)];
    if(f.example){ closeWelcome(); resetToDefaultWorkflow(); }
    else welcomeMessage(`This mockup keeps no files, so <code>${escHtml(f.name)}</code> cannot be opened.`);
  }
});
welcomeEl.addEventListener('change', e=>{
  if(e.target.classList.contains('wc-start')){ setWelcomeAtStart(e.target.checked); return; }
  if(e.target.classList.contains('wc-file') && e.target.files.length){
    welcomeMessage(`This mockup cannot read pipeline files, so <code>${escHtml(e.target.files[0].name)}</code> is not loaded. ` +
      'The real GUI reads it with <code>load_spec()</code> from <code>m3resp.workflows.spec</code>.');
  }
});
welcomeEl.addEventListener('keydown', e=>{
  if(e.key==='Escape'){ e.stopPropagation(); closeWelcome(); }
});
document.getElementById('examplesBtn').addEventListener('click', openWelcome);
