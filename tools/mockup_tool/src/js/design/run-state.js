// ---- run state on the canvas ----
// "▶ Run workflow" plays a fake run: the steps run one at a time in graph
// order, each with a progress ring, and the Run log fills line by line. A
// step with no linked input fails, and the steps after it are skipped. A
// change to a step's settings or links marks it and every step after it
// "out of date"; with Auto-run on, those steps run again at once. The first
// run of the whole pipeline to its end opens the 3 · Check tab.
let runStepScale = 1;      // multiplies the time each fake step takes
let autoRun = false;
let currentRun = null;     // {stop} while a run is going on

// problems found by the last run or by Validate
let PROBLEMS = [];
// warnings the fake run always reports for these steps (taken from a real run)
const KNOWN_WARNINGS = {
  emg_ecg_gating: {code:'session_dependency_reordered',
    msg:'reads <code>session.processed.emg</code>; confirm this runs after <b>emg_preprocess</b> writes it.'},
};

const runBtn = document.getElementById('runPipelineBtn');
const RUN_BTN_TEXT = runBtn.textContent;

// ---- graph order ----
// every node that gets data from `id`, directly or through other nodes
function downstreamOf(id){
  const seen = new Set(), todo = [id];
  while(todo.length){
    const cur = todo.pop();
    EDGES.forEach(e=>{ if(e.src===cur && !seen.has(e.dst)){ seen.add(e.dst); todo.push(e.dst); } });
  }
  seen.delete(id);
  return seen;
}
// `ids` in run order: a step comes after every step it gets data from. Steps
// that are ready at the same time go left to right, then top to bottom.
function runOrder(ids){
  const set = new Set(ids || NODES.map(n=>n.id));
  const pos = id=>{ const n = NODES.find(x=>x.id===id) || {x:0, y:0}; return [n.x, n.y]; };
  const byPos = (a, b)=>{ const pa = pos(a), pb = pos(b); return pa[0]-pb[0] || pa[1]-pb[1]; };
  const links = EDGES.filter(e=> set.has(e.src) && set.has(e.dst) && e.src!==e.dst);
  const waiting = new Map([...set].map(id=>[id, 0]));
  links.forEach(e=> waiting.set(e.dst, waiting.get(e.dst)+1));
  const order = [];
  let ready = [...set].filter(id=>waiting.get(id)===0).sort(byPos);
  while(ready.length){
    const id = ready.shift();
    order.push(id);
    links.filter(e=>e.src===id).forEach(e=>{
      waiting.set(e.dst, waiting.get(e.dst)-1);
      if(waiting.get(e.dst)===0) ready.push(e.dst);
    });
    ready.sort(byPos);
  }
  // steps in a loop never become ready; they run last
  return order.concat([...set].filter(id=>!order.includes(id)).sort(byPos));
}

// ---- drawing the state ----
function refreshNodeState(id){
  const g = collapsedGroupOf(id);
  const el = inner.querySelector(`.node[data-id="${g ? g.id : id}"]`);
  const n = nodeById(g ? g.id : id);
  if(!el || !n) return;
  el.dataset.status = nodeStatus(n);
  el.querySelector('.run-badge').outerHTML = runBadgeHtml(n);
  updateNodeWindowStatus(id);
}
function setNodeProgress(id, p){
  const g = collapsedGroupOf(id);
  const el = inner.querySelector(`.node[data-id="${g ? g.id : id}"]`);
  if(el) el.style.setProperty('--run-p', p.toFixed(3));
}
// put the canvas on a node and make it blink
function focusNode(id){
  const g = collapsedGroupOf(id);
  const el = inner.querySelector(`.node[data-id="${g ? g.id : id}"]`);
  if(!el) return;
  el.scrollIntoView({behavior:'smooth', block:'center', inline:'center'});
  if(!g) selectNode(id);
  el.classList.remove('blink');
  void el.offsetWidth;   // restart the animation
  el.classList.add('blink');
  setTimeout(()=> el.classList.remove('blink'), 1800);
}

// ---- out of date ----
function markStale(id){
  const ids = [id, ...downstreamOf(id)];
  ids.forEach(nid=>{
    const n = NODES.find(x=>x.id===nid);
    if(n && ['ok','warn','error'].includes(n.status)){ n.status = 'stale'; refreshNodeState(nid); }
  });
  if(autoRun && !currentRun) runPipeline(ids);
}

// ---- checks ----
// problems of one step: a step (other than a loading step, which reads the
// working window of 1 · Prepare) needs at least one linked input besides the
// session; some steps also give a known warning
function checkNode(n){
  const out = [];
  const ins = n.ins.filter(([name])=>name!=='session');
  if(nodeCategory(n)!=='loading' && ins.length && !ins.some(([name])=>portHasLink(n.id, 'in', name))){
    out.push({node:n.id, sev:'error', code:'missing_binding',
      msg:`no input is linked; it needs ${ins.map(([name])=>`<code>${name}</code>`).join(' or ')}.`});
  }
  const known = KNOWN_WARNINGS[n.id];
  if(known) out.push({node:n.id, sev:'warn', ...known});
  return out;
}

// ---- dock: Problems and Run log ----
const dockProblemsEl = document.getElementById('dockProblems');
const dockLogEl = document.getElementById('dockLog');
function renderProblems(){
  dockProblemsEl.innerHTML = PROBLEMS.length
    ? PROBLEMS.map(p=>`
      <div class="diag-row" data-node="${p.node}" title="Click to show this step on the canvas">
        <div class="sev-stripe ${p.sev}"></div>
        <div><b>${p.node}</b><div class="msg">${p.code} (${p.sev==='error'?'error':'warning'}): ${p.msg}</div></div>
      </div>`).join('')
    : '<div class="dock-empty">No problems.</div>';
  const tab = document.querySelector('#view-design .dock-tab[data-dock="problems"]');
  tab.querySelector('.n').textContent = PROBLEMS.length;
  tab.classList.toggle('warn', PROBLEMS.length>0);
}
dockProblemsEl.addEventListener('click', e=>{
  const row = e.target.closest('.diag-row[data-node]');
  if(row) focusNode(row.dataset.node);
});
function showDockTab(key){
  document.querySelector(`#view-design .dock-tab[data-dock="${key}"]`).click();
}
function logLine(text, cls){
  const t = new Date().toTimeString().slice(0, 8);
  dockLogEl.insertAdjacentHTML('beforeend', `<div class="log-line${cls?' '+cls:''}"><span class="t">${t}</span>${text}</div>`);
  dockLogEl.scrollTop = dockLogEl.scrollHeight;
  document.querySelector('#view-design .dock-tab[data-dock="log"] .n').textContent = dockLogEl.querySelectorAll('.log-line').length;
}

// ---- the fake run ----
const sleep = ms=> new Promise(r=> setTimeout(r, ms));
function stepTime(n){
  const cat = nodeCategory(n);
  const ms = cat==='loading' ? 900 : ['detection','roi','quality'].includes(cat) ? 650 : 420;
  return ms*runStepScale;
}
async function playProgress(id, ms){
  const ticks = Math.max(3, Math.round(ms/40));
  for(let i=1;i<=ticks;i++){
    if(currentRun.stop) return false;
    setNodeProgress(id, i/ticks);
    await sleep(ms/ticks);
  }
  return true;
}
// runs `ids` (all steps when omitted); returns when the run is over
async function runPipeline(ids){
  if(currentRun) return;
  const full = !ids;
  const order = runOrder(ids).filter(id=>NODES.some(n=>n.id===id));
  currentRun = {stop:false};
  runBtn.textContent = '■ Stop';
  runBtn.title = 'Stop the run';
  if(full) dockLogEl.innerHTML = '';
  showDockTab('log');
  logLine(`run started · ${order.length} step${order.length===1?'':'s'}`);
  PROBLEMS = PROBLEMS.filter(p=>!order.includes(p.node));
  const failed = new Set();
  let done = 0, warned = 0, stopped = false;
  for(const id of order){
    const n = NODES.find(x=>x.id===id);
    if(!n) continue;
    if(EDGES.some(e=>e.dst===id && failed.has(e.src))){
      failed.add(id);
      n.status = 'pending';
      refreshNodeState(id);
      logLine(`${id} skipped: an earlier step failed`, 'muted');
      continue;
    }
    const before = n.status;
    n.status = 'running';
    setNodeProgress(id, 0);
    refreshNodeState(id);
    if(!await playProgress(id, stepTime(n))){
      n.status = before;
      refreshNodeState(id);
      stopped = true;
      break;
    }
    const problems = checkNode(n);
    PROBLEMS.push(...problems);
    if(problems.some(p=>p.sev==='error')){
      n.status = 'error';
      failed.add(id);
      logLine(`${id} failed: ${problems[0].msg}`, 'crit');
    } else {
      n.status = problems.length ? 'warn' : 'ok';
      if(problems.length) warned++;
      const outs = n.outs.map(([name, type])=>`${name} (${type})`).join(', ') || 'done';
      logLine(`${id} → ${outs}`, problems.length ? 'warnl' : '');
      done++;
    }
    refreshNodeState(id);
  }
  if(stopped) logLine('run stopped', 'muted');
  else logLine(`${done}/${order.length} steps completed` +
    (warned ? ` · ${warned} warning${warned===1?'':'s'}` : '') +
    (failed.size ? ` · ${failed.size} failed or skipped` : ''), failed.size ? 'crit' : 'ok');
  renderProblems();
  currentRun = null;
  runBtn.textContent = RUN_BTN_TEXT;
  runBtn.title = 'Run all steps in graph order';
  const checkTab = document.getElementById('checkTab');
  if(full && !stopped && checkTab.classList.contains('locked')){
    unlockTab(checkTab);
    logLine('3 · Check is open: the quality checks and the report of this run', 'ok');
    showEdgeNote(runBtn, 'Run finished. 3 · Check is now open.');
  }
}

// ---- toolbar ----
runBtn.addEventListener('click', ()=>{
  if(currentRun) currentRun.stop = true;
  else runPipeline();
});
document.getElementById('validateBtn').addEventListener('click', e=>{
  PROBLEMS = NODES.flatMap(checkNode);
  renderProblems();
  showDockTab('problems');
  showEdgeNote(e.currentTarget, PROBLEMS.length ? `${PROBLEMS.length} problem${PROBLEMS.length===1?'':'s'} found.` : 'No problems found.');
});
document.getElementById('autoRunToggle').addEventListener('change', e=>{
  autoRun = e.target.checked;
  // switching it on brings everything that is out of date up to date
  const stale = NODES.filter(n=>n.status==='stale').map(n=>n.id);
  if(autoRun && stale.length && !currentRun) runPipeline(stale);
});

// the problems known before any run in this mockup
PROBLEMS = NODES.filter(n=>KNOWN_WARNINGS[n.id]).map(n=>({node:n.id, sev:'warn', ...KNOWN_WARNINGS[n.id]}));
renderProblems();
