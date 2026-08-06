// ---- saved sequences: each is a named, concatenated snapshot of the Prepare
// working-window selection, kept around so several distinct sequences can
// exist at once and show up as loadable data on 2 · Design's "Available data"
// tab. Declared first, before anything else in this script, because the
// Design-tab code below runs its first render before the Prepare-tab code
// (further down this same script) would otherwise define these.
let SAVED_SEQUENCES = [];
let seqSaveSeq = 0;

// ---- tab switching ----
function goToView(name){
  const btn = document.querySelector(`.tab-btn[data-view="${name}"]`);
  if(!btn || btn.classList.contains('locked')) return;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active', b===btn));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.getElementById('view-'+name).classList.add('active');
  // "Available data" may be stale if it was already the active palette tab
  // (the default) when sequences were saved or nodes changed elsewhere
  if(name==='design' && typeof renderDataTab==='function') renderDataTab();
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
  btn.textContent = 'Generating parameters & plots…';
  setTimeout(()=>{
    const resultsTab = document.getElementById('resultsTab');
    resultsTab.classList.remove('locked');
    resultsTab.textContent = '4 · Results';
    resultsTab.title = '';
    btn.disabled = false;
    btn.dataset.confirmed = '1';
    btn.textContent = '✓ Confirmed — view results →';
    document.getElementById('checkConfirmBar').classList.add('confirmed');
    goToView('results');
  }, 650);
});

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

