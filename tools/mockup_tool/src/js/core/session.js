// ================= the session: keep, save, load, start new =================
// The session is everything the user has set up on the four tabs: the files,
// sections and signals of 1 · Prepare, the offsets, the working windows and
// saved sequences, the graph of 2 · Design with its settings and run state,
// and how far the user got on 3 · Check and 4 · Results.
//
// The browser keeps the session (localStorage), so a reload of the page
// opens the same session again. The actions, in the ☰ menu left of the logo:
//   New    start a new session: the page goes back to its start state
//   Save   write the session to a file, m3resp_session_<name>.json
//   Load…  read a session file and open it
// This file loads last (see build.js): it reads the state that all the other
// files have set up, and on a reload it puts the kept state back.

const SESSION_KEY = 'm3resp-mockup-session';
const SESSION_FORMAT = 'm3resp-mockup-session';
const SESSION_VERSION = 1;
const SESSION_KEEP_MS = 3000;    // the browser copy is written again this often

let sessionId = null;
let sessionCreated = null;       // ISO time the session started
let sessionReloading = false;    // true while New or Load reloads the page

// session name from the clock: 2026-10-15_142233
function newSessionId(d=new Date()){
  const p = n=> String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function showSessionId(){
  document.getElementById('resSessionId').textContent = sessionId;
  document.getElementById('appMenuSessionId').textContent = sessionId;
  document.getElementById('appMenuStarted').textContent = `started ${new Date(sessionCreated).toLocaleString()}`;
  document.getElementById('appMenuBtn').title = `Session ${sessionId}: new, save, load`;
}

// ---------------- read the state of each tab ----------------
function prepareSnapshot(){
  const builtins = {};
  Object.entries(BUILTIN_CARD_ID).forEach(([mod, cardId])=>{
    const row = document.getElementById('modRow-'+mod), head = document.getElementById(cardId).querySelector(':scope > .card-head');
    const loaded = !row.classList.contains('unloaded');
    builtins[mod] = {
      loaded,
      rowFile: loaded ? row.querySelector('.file').textContent : (row.dataset.file || ''),
      meta: head.querySelector('.meta')?.textContent ?? null,
      pill: head.querySelector('.pill')?.textContent ?? null,
    };
  });
  return {
    subjectType: document.getElementById('recSubjectType').value,
    builtins,
    custom: Object.entries(customModalities).map(([id, m])=>({id, name:m.name, vendor:m.vendor})),
    customSeq,
    files: {eit:EIT_MERGE.snapshot(), emg:EMG_MERGE.snapshot(), vent:VENT_MERGE.snapshot()},
    emgChannels: EMG_CHANNELS.map(c=>({seed:Number(c.id.replace('emg_', '')), muscleKey:c.muscleKey, side:c.side, ch:c.ch, on:c.on})),
    emgSeq,
    ventOn: Object.fromEntries(VENT_CHANNELS.map(c=>[c.id, c.on])),
    sync: JSON.parse(JSON.stringify(SYNC_STATE)),
    syncMethod: document.getElementById('syncMethod').value,
    syncReference: document.getElementById('syncReference').value,
    windows: JSON.parse(JSON.stringify(windows)),
    winSeq, activeWindowId, sliceMode,
    sequences: JSON.parse(JSON.stringify(SAVED_SEQUENCES)),
    seqSaveSeq,
    section: currentSectionId,
  };
}
function designSnapshot(){
  const insp = document.getElementById('inspectorBody');
  return {
    nodeView: NODE_VIEW,
    // a step that runs while the page closes is out of date when it opens again
    nodes: NODES.map(n=> n.status==='running' ? {...n, status:'stale'} : n),
    edges: EDGES, groups: GROUPS, annotations: ANNOTATIONS,
    sessionRailBox,
    problems: PROBLEMS,
    runLog: [...dockLogEl.querySelectorAll('.log-line')].map(el=>el.outerHTML),
    autoRun,
    selected: insp.dataset.selected || null,
    welcomeSeen: designWelcomeSeen,
  };
}
function stagesSnapshot(){
  return {
    view: document.querySelector('.tab-btn.active')?.dataset.view || 'prepare',
    checkOpen: !document.getElementById('checkTab').classList.contains('locked'),
    confirmed: !!document.getElementById('confirmResultsBtn').dataset.confirmed,
    checkedAt: document.getElementById('resCheckedAt').textContent,
    checkPart: currentCheckPart,
    resultsPart: currentResPart,
  };
}
function sessionSnapshot(){
  return JSON.parse(JSON.stringify({
    format: SESSION_FORMAT, version: SESSION_VERSION,
    id: sessionId, created: sessionCreated, saved: new Date().toISOString(),
    prepare: prepareSnapshot(),
    design: designSnapshot(),
    stages: stagesSnapshot(),
  }));
}

// ---------------- put a kept state back ----------------
function restorePrepare(p){
  document.getElementById('recSubjectType').value = p.subjectType;

  // EMG signals first: the sync row and the file lanes use them. Each signal
  // gets its old number back, so its id and its noise stay the same.
  EMG_CHANNELS.length = 0;
  p.emgChannels.forEach(c=>{
    if(!EMG_MUSCLES[c.muscleKey]) throw new Error(`unknown EMG muscle "${c.muscleKey}"`);
    emgSeq = c.seed - 1;
    const chan = makeEmgChannel(c.muscleKey, c.side, c.ch);
    chan.on = !!c.on;
    EMG_CHANNELS.push(chan);
  });
  emgSeq = p.emgSeq;
  emgChanged();
  pickFreeEmgChannel();

  VENT_CHANNELS.forEach(c=>{
    if(!(c.id in p.ventOn)) return;
    c.on = !!p.ventOn[c.id];
    ventChecks.querySelector(`input[data-ch="${c.id}"]`).checked = c.on;
  });
  syncVentVisibility();

  EIT_MERGE.restore(p.files.eit);
  EMG_MERGE.restore(p.files.emg);
  VENT_MERGE.restore(p.files.vent);

  Object.keys(SYNC_IDS).forEach(mod=>{
    const s = p.sync[mod];
    if(!s) return;
    Object.assign(SYNC_STATE[mod], {est:+s.est || 0, typed:s.typed===null ? null : +s.typed, sig:s.sig, ref:s.ref});
    document.getElementById(SYNC_IDS[mod].input).value = SYNC_STATE[mod].typed===null ? '' : String(SYNC_STATE[mod].typed);
  });
  document.getElementById('syncMethod').value = p.syncMethod;
  document.getElementById('syncReference').value = p.syncReference;
  updateSyncEstimateButton();
  renderSyncSelects();
  Object.keys(SYNC_IDS).forEach(updateSyncRow);
  renderSyncCheck();

  SAVED_SEQUENCES = p.sequences;
  seqSaveSeq = p.seqSaveSeq;
  windows = p.windows;
  winSeq = p.winSeq;
  activeWindowId = p.activeWindowId;
  renderSavedSequences();
  document.getElementById('sequenceNameInput').value = defaultSequenceName();
  if(p.sliceMode!==sliceMode) document.querySelector(`#sliceMode button[data-mode="${p.sliceMode}"]`)?.click();
  renderWindows();
  renderSliceLanes();

  // the file names and the "Merged" marks of the built-in modalities
  Object.entries(p.builtins).forEach(([mod, b])=>{
    const row = document.getElementById('modRow-'+mod), head = document.getElementById(BUILTIN_CARD_ID[mod]).querySelector(':scope > .card-head');
    if(!row) return;
    row.dataset.file = b.rowFile;
    row.querySelector('.file').textContent = b.rowFile;
    if(b.meta!==null && head.querySelector('.meta')) head.querySelector('.meta').textContent = b.meta;
    if(b.pill!==null && head.querySelector('.pill')) head.querySelector('.pill').textContent = b.pill;
    if(!b.loaded) setBuiltinLoaded(mod, false);
  });
  p.custom.forEach(m=> addCustomModality(m.id, m.name, m.vendor));
  customSeq = p.customSeq;
  showSection(p.section);
}

function restoreDesign(d){
  NODE_VIEW = d.nodeView==='detailed' ? 'detailed' : 'compact';
  document.querySelectorAll('#nodeViewSeg button').forEach(b=> b.classList.toggle('on', b.dataset.nodeview===NODE_VIEW));
  NODES.splice(0, NODES.length, ...d.nodes);
  EDGES.splice(0, EDGES.length, ...d.edges);
  GROUPS.splice(0, GROUPS.length, ...d.groups);
  ANNOTATIONS.splice(0, ANNOTATIONS.length, ...d.annotations);
  sessionRailBox = d.sessionRailBox || null;
  // new notes and arrows continue the numbers of the kept ones
  annotSeq = Math.max(0, ...ANNOTATIONS.map(a=> Number(String(a.id).split('_').pop()) || 0));
  PROBLEMS = d.problems;
  clearMultiSelect();
  renderAllNodes();
  drawEdges();
  renderProblems();
  // the run log: the lines of the last run
  dockLogEl.innerHTML = d.runLog.join('');
  document.querySelector('#view-design .dock-tab[data-dock="log"] .n').textContent = dockLogEl.querySelectorAll('.log-line').length;
  autoRun = !!d.autoRun;
  document.getElementById('autoRunToggle').checked = autoRun;
  if(d.selected && nodeById(d.selected)) selectNode(d.selected);
  else renderInspectorEmpty();
  designWelcomeSeen = !!d.welcomeSeen;
}

function restoreStages(s){
  if(s.checkOpen) unlockTab(document.getElementById('checkTab'));
  if(s.confirmed){
    markResultsConfirmed();
    document.getElementById('resCheckedAt').textContent = s.checkedAt;
  }
  if(CHECK_ORDER.includes(s.checkPart)) showCheckPart(s.checkPart);
  if(RES_ORDER.includes(s.resultsPart)) showResPart(s.resultsPart);
  goToView(s.view);
}

// a session file or a browser copy: the right format, and all parts there
function checkSessionData(data){
  if(!data || data.format!==SESSION_FORMAT) throw new Error('this is not a session file of the m3resp mockup');
  if(data.version!==SESSION_VERSION) throw new Error(`session version ${data.version}; this mockup reads version ${SESSION_VERSION}`);
  if(typeof data.id!=='string' || !data.id) throw new Error('the session has no name');
  ['prepare', 'design', 'stages'].forEach(k=>{ if(!data[k]) throw new Error(`the session has no "${k}" part`); });
}

// Each part goes back on its own, so a part that fails leaves the others in
// place. Returns the names of the parts that failed.
function restoreSession(data){
  sessionId = data.id;
  sessionCreated = data.created || new Date().toISOString();
  showSessionId();
  const failed = [];
  [['1 · Prepare', ()=> restorePrepare(data.prepare)],
   ['2 · Design', ()=> restoreDesign(data.design)],
   ['3 · Check and 4 · Results', ()=> restoreStages(data.stages)]].forEach(([name, fn])=>{
    try{ fn(); }catch(err){ failed.push(name); console.error(`Session ${data.id}: ${name} was not restored.`, err); }
  });
  return failed;
}

// ---------------- the browser copy ----------------
function keepSession(){
  if(sessionReloading) return;
  try{ localStorage.setItem(SESSION_KEY, JSON.stringify(sessionSnapshot())); }
  catch(err){ /* this browser keeps no data for the page: a reload starts again */ }
}
function readKeptSession(){
  try{ return JSON.parse(localStorage.getItem(SESSION_KEY)); }catch(err){ return null; }
}
// New and Load write the next session (or none) and open the page again
function reloadWithSession(data){
  sessionReloading = true;
  try{
    if(data) localStorage.setItem(SESSION_KEY, JSON.stringify(data));
    else localStorage.removeItem(SESSION_KEY);
  }catch(err){
    sessionReloading = false;
    return false;
  }
  location.reload();
  return true;
}

// ---------------- the actions ----------------
// The actions are in the menu under the ☰ button left of the logo. A message
// shows next to that button.
const sessionLoadInput = document.getElementById('sessionLoadInput');
let sessionLoadFrom = null;      // the button or item that opened the file picker

function startNewSession(from){
  const ok = confirm(`Start a new session?\n\nThe page goes back to its start state, and session ${sessionId} closes. Use Save first to keep it in a file.`);
  if(!ok) return;
  if(!reloadWithSession(null)) showEdgeNote(from, 'This browser keeps no data for this page, so a new session cannot start. Reload the page instead.');
}

function saveSessionFile(from){
  const data = sessionSnapshot();
  const name = `m3resp_session_${sessionId}.json`;
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:'application/json'}));
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=> URL.revokeObjectURL(url), 1000);
  keepSession();
  showEdgeNote(from, `Saved session ${sessionId} as ${name}. "Load…" opens it again.`);
}

function pickSessionFile(from){
  sessionLoadFrom = from;
  sessionLoadInput.click();
}
sessionLoadInput.addEventListener('change', ()=>{
  const file = sessionLoadInput.files[0], from = sessionLoadFrom || document.getElementById('appMenuBtn');
  sessionLoadInput.value = '';
  if(!file) return;
  const reader = new FileReader();
  reader.onload = ()=>{
    let data;
    try{
      data = JSON.parse(reader.result);
      checkSessionData(data);
    }catch(err){
      showEdgeNote(from, `"${file.name}" was not loaded: ${err.message}.`);
      return;
    }
    if(!confirm(`Load session ${data.id} from "${file.name}"?\n\nSession ${sessionId} closes. Use Save first to keep it in a file.`)) return;
    if(!reloadWithSession(data)) showEdgeNote(from, 'This browser keeps no data for this page, so the session cannot be loaded.');
  };
  reader.onerror = ()=> showEdgeNote(from, `"${file.name}" could not be read.`);
  reader.readAsText(file);
});

const SESSION_ACTIONS = {new:startNewSession, save:saveSessionFile, load:pickSessionFile};

// the menu under the ☰ button: the session name and the same three actions
(function wireAppMenu(){
  const btn = document.getElementById('appMenuBtn');
  const menu = document.getElementById('appMenu');
  placeIcons(menu);
  function setOpen(open){ menu.hidden = !open; btn.setAttribute('aria-expanded', String(open)); }
  btn.addEventListener('click', ()=> setOpen(menu.hidden));
  menu.addEventListener('click', e=>{
    const item = e.target.closest('[data-session-act]'); if(!item) return;
    setOpen(false);
    SESSION_ACTIONS[item.dataset.sessionAct](btn);
  });
  document.addEventListener('mousedown', e=>{ if(!menu.hidden && !e.target.closest('.app-menu')) setOpen(false); });
  document.addEventListener('keydown', e=>{ if(e.key==='Escape' && !menu.hidden){ setOpen(false); btn.focus(); } });
})();

// ---------------- on page load ----------------
(function openSession(){
  const kept = readKeptSession();
  let failed = [];
  try{
    checkSessionData(kept);
    failed = restoreSession(kept);
  }catch(err){
    // no kept session, or one this mockup cannot read: a new session starts
    sessionId = newSessionId();
    sessionCreated = new Date().toISOString();
    showSessionId();
  }
  if(failed.length) showEdgeNote(document.getElementById('appMenuBtn'),
    `Session ${sessionId}: ${failed.join(', ')} could not be restored and show the start state. The browser console has the details.`);
  keepSession();
  setInterval(keepSession, SESSION_KEEP_MS);
  window.addEventListener('pagehide', keepSession);
  document.addEventListener('visibilitychange', ()=>{ if(document.visibilityState==='hidden') keepSession(); });
})();
