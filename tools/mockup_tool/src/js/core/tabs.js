// ---- saved sequences: each is a named, concatenated snapshot of the Prepare
// working-window selection, kept around so several distinct sequences can
// exist at once and show up as loadable data on 2 · Design's "Available data"
// tab. Declared here because this file loads first (see build.js): the
// Design-tab code renders once on load, before js/prepare/saved-sequences.js
// — which owns the rest of this feature — has run.
let SAVED_SEQUENCES = [];
let seqSaveSeq = 0;

// ---- tab switching ----
function goToView(name){
  const btn = document.querySelector(`.tab-btn[data-view="${name}"]`);
  if(!btn || btn.classList.contains('locked')) return;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active', b===btn));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.getElementById('view-'+name).classList.add('active');
  // draw "Available data" again: sequences saved in 1 · Prepare since the
  // last visit go into it
  if(name==='design' && typeof renderDataTab==='function') renderDataTab();
  // the first visit shows the whole graph; after that the user's zoom stays
  if(name==='design' && !designViewFitted && typeof fitCanvasToView==='function'){
    designViewFitted = true;
    fitCanvasToView(1);
    // the welcome card opens once per session (js/core/session.js keeps this)
    if(!designWelcomeSeen && typeof openWelcome==='function' && welcomeAtStart()) openWelcome();
    designWelcomeSeen = true;
  }
}
let designViewFitted = false;
let designWelcomeSeen = false;
// a locked tab opens after its step: 3 · Check after a run of the pipeline on
// 2 · Design (js/design/run-state.js), 4 · Results after the confirm on 3 · Check
function unlockTab(btn){
  btn.classList.remove('locked');
  btn.title = '';
  const lock = btn.querySelector('.tab-lock');
  if(lock) lock.remove();
}
document.getElementById('tabs').addEventListener('click', e=>{
  const btn = e.target.closest('.tab-btn'); if(!btn) return;
  goToView(btn.dataset.view);
});

// ---- "3 · Check" -> confirm gate -> unlocks "4 · Results" ----
document.getElementById('confirmResultsBtn').addEventListener('click', e=>{
  const btn = e.currentTarget;
  if(btn.dataset.confirmed) { goToView('results'); return; }
  btn.disabled = true;
  btn.textContent = 'Making parameters and plots…';
  setTimeout(()=>{
    markResultsConfirmed();
    goToView('results');
  }, 650);
});
// the confirm on 3 · Check is done: 4 · Results opens
function markResultsConfirmed(){
  const btn = document.getElementById('confirmResultsBtn');
  unlockTab(document.getElementById('resultsTab'));
  btn.disabled = false;
  btn.dataset.confirmed = '1';
  btn.textContent = 'Open 4 · Results';
  document.getElementById('checkConfirmBar').classList.add('confirmed');
  if(typeof checkConfirmed==='function') checkConfirmed();
}

// generic dock-tab wiring: works for any number of .dock instances on the page
document.querySelectorAll('.dock').forEach(dock=>{
  dock.addEventListener('click', e=>{
    const tab=e.target.closest('.dock-tab');
    if(!tab || !dock.contains(tab)) return;
    const key=tab.dataset.dock;
    dock.querySelectorAll(':scope > .dock-tabs > .dock-tab').forEach(t=>t.classList.toggle('active', t===tab));
    dock.querySelectorAll(':scope > .dock-body').forEach(b=>b.classList.toggle('hidden', b.dataset.dock!==key));
  });
});

