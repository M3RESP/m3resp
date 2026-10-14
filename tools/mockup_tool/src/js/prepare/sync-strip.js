// ---- raw synchronization: strip, match, estimate, alignment check ----
// The ventilator is the reference. Each other recording has an offset in
// seconds: the time of its first sample on the reference time line (a
// negative offset starts before the reference). A recording read at
// reference time t gives its own sample at t − offset.
//
// Each row chooses its signal (EIT: global impedance or one ROI; EMG: one of
// the loaded EMG signals, as its envelope: RMS over 0.3 s, ECG removed) and
// the ventilator signal it is compared with. The estimate is the
// cross-correlation of the two over −5.0 to +5.0 s in steps of 0.1 s.
// "Match" is the correlation at the offset in use (1 = the same shape).
const SYNC_VIEW = {t0:-6, t1:REC_SECONDS+6};   // seconds shown on the strip
const SYNC_SEARCH = {lo:-5, hi:5, step:0.1};
const SYNC_TEST = {t0:6, t1:REC_SECONDS-6, dt:0.1}; // reference times the match uses
function syncFrac(t){ return (t - SYNC_VIEW.t0)/(SYNC_VIEW.t1 - SYNC_VIEW.t0); }

// EMG envelope of one EMG signal after ECG removal: RMS over 0.3 s, on a
// 200 Hz grid of file time from −8 s to REC_SECONDS + 8 s
const EMG_ENV = {};
function emgEnvelopeAt(tau, chanId){
  const chan = EMG_CHANNELS.find(c=>c.id===chanId) || EMG_CHANNELS[0];
  if(!chan) return 0;
  if(!EMG_ENV[chan.id]){
    const fs=200, t0=-8, n=Math.round((REC_SECONDS+16)*fs), win=Math.round(0.3*fs);
    const sq = new Float32Array(n);
    for(let i=0;i<n;i++){ const v = chan.noEcgFn(t0+i/fs, i); sq[i] = v*v; }
    const env = new Float32Array(n);
    let acc = 0;
    for(let i=0;i<n;i++){
      acc += sq[i];
      if(i>=win) acc -= sq[i-win];
      env[Math.max(0, i-(win>>1))] = Math.sqrt(acc/Math.min(i+1, win));
    }
    EMG_ENV[chan.id] = {fs, t0, env};
  }
  const e = EMG_ENV[chan.id], k = Math.round((tau-e.t0)*e.fs);
  return e.env[Math.max(0, Math.min(e.env.length-1, k))];
}

const ventChan = id=> VENT_CHANNELS.find(c=>c.id===id);
// ventilator signals read from a file channel (P_es and P_ga are unmapped here)
const syncRefChoices = ()=> VENT_CHANNELS.filter(c=> /^ch /.test(c.ch)).map(c=>({id:c.id, name:c.name}));
// an ROI curve of the real EIT recording (file 1), at file time tau
function eitRoiAt(k, tau){
  const a = fileDetailOf(EIT_FILES.find(f=>f.real) || EIT_FILES[0]).raw[k];
  return a[Math.max(0, Math.min(a.length-1, Math.round(tau*DETAIL_HZ)))];
}
const SYNC_MODS = {
  eit: {name:'EIT',
        choices:()=> [{id:'global', name:'Global impedance'}, ...ROI_KEYS.map((k,i)=>({id:k, name:`ROI ${i+1}`}))],
        at:(tau, sig)=> sig==='global' ? eitOverviewAt(tau) : eitRoiAt(sig, tau)},
  emg: {name:'EMG',
        choices:()=> EMG_CHANNELS.map(c=>({id:c.id, name:`${c.name} envelope`})),
        at:(tau, sig)=> emgEnvelopeAt(tau, sig)},
};
// sig: the signal of the row; ref: the ventilator signal it is compared with
const SYNC_STATE = {eit:{est:0, typed:null, sig:'global', ref:'volume'},
                    emg:{est:0, typed:null, sig:EMG_CHANNELS[0] && EMG_CHANNELS[0].id, ref:'paw'}};
function syncSigName(mod){
  const c = SYNC_MODS[mod].choices().find(x=>x.id===SYNC_STATE[mod].sig);
  return c ? c.name : '';
}
function syncOffsetOf(mod){
  const s = SYNC_STATE[mod];
  return !s ? 0 : (s.typed!==null ? s.typed : s.est);
}

function correlation(a, b){
  const n=a.length; let ma=0, mb=0;
  for(let i=0;i<n;i++){ ma+=a[i]; mb+=b[i]; }
  ma/=n; mb/=n;
  let sab=0, saa=0, sbb=0;
  for(let i=0;i<n;i++){ const x=a[i]-ma, y=b[i]-mb; sab+=x*y; saa+=x*x; sbb+=y*y; }
  return (saa && sbb) ? sab/Math.sqrt(saa*sbb) : 0;
}
// the reference signal and the modality on the test grid, for one offset
function syncSeries(mod, offset){
  const m = SYNC_MODS[mod], st = SYNC_STATE[mod], ref = ventChan(st.ref), a=[], b=[];
  for(let t=SYNC_TEST.t0; t<=SYNC_TEST.t1; t+=SYNC_TEST.dt){
    a.push(ref.fn(ventPhase(t)));
    b.push(m.at(t - offset, st.sig));
  }
  return [a, b];
}
function syncMatch(mod, offset){ return correlation(...syncSeries(mod, offset)); }
function estimateOffset(mod){
  let best = {o:0, r:-2};
  for(let o=SYNC_SEARCH.lo; o<=SYNC_SEARCH.hi+1e-9; o+=SYNC_SEARCH.step){
    const r = syncMatch(mod, o);
    if(r > best.r) best = {o:Math.round(o*10)/10, r};
  }
  return best.o;
}

function placeSyncBar(bar, offset){
  const shown = Math.max(SYNC_VIEW.t0, Math.min(-SYNC_VIEW.t0, offset));
  bar.style.left = (syncFrac(shown)*100)+'%';
  bar.style.width = (REC_SECONDS/(SYNC_VIEW.t1-SYNC_VIEW.t0)*100)+'%';
}

(function drawSyncRuler(){
  // a thin line in each track marks 0 s of the reference
  document.getElementById('syncStrip').style.setProperty('--sync-zero', (syncFrac(0)*100)+'%');
  const ruler = document.getElementById('syncRuler');
  let out = '';
  for(let t=0; t<=REC_SECONDS; t+=10) out += `<span style="left:${(syncFrac(t)*100).toFixed(2)}%">${fmtSec(t)}</span>`;
  ruler.innerHTML = out;
})();

const SYNC_IDS = {eit:{input:'syncOffsetEit', bar:'syncBarEit', match:'syncMatchEit'},
                  emg:{input:'syncOffsetEmg', bar:'syncBarEmg', match:'syncMatchEmg'}};
function fmtOffset(sec){ return `${sec>0?'+':sec<0?'−':''}${Math.abs(sec).toFixed(1)} s`; }

function updateSyncRow(mod){
  const ids = SYNC_IDS[mod], st = SYNC_STATE[mod];
  const input = document.getElementById(ids.input), bar = document.getElementById(ids.bar);
  const sec = syncOffsetOf(mod), typed = st.typed!==null;
  input.placeholder = st.est.toFixed(1);
  placeSyncBar(bar, sec);
  bar.querySelector('.v').textContent = fmtOffset(sec);
  const tag = bar.querySelector('.tag');
  tag.textContent = typed ? 'typed' : 'estimated';
  tag.classList.toggle('typed', typed);
  const r = syncMatch(mod, sec), m = document.getElementById(ids.match);
  m.textContent = r.toFixed(2);
  m.className = 'sync-match ' + (r>=0.8 ? 'good' : r>=0.5 ? 'fair' : 'poor');
  m.title = `Correlation of ${SYNC_MODS[mod].name} ${syncSigName(mod)} with ventilator ${ventChan(st.ref).name} at ${fmtOffset(sec)}`;
}

// ---- alignment check: reference Flow, EIT and the EMG envelope on one time line ----
let syncViewFrom = 0;
const SYNC_VIEW_LEN = 20;
function syncLaneSvg(fn, t0, t1, color){
  const N=500, W=1000, H=54, PAD=5, vals=[];
  for(let i=0;i<N;i++) vals.push(fn(t0+(i/(N-1))*(t1-t0)));
  const lo=Math.min(...vals), hi=Math.max(...vals), span=(hi-lo)||1;
  let pts='';
  for(let i=0;i<N;i++) pts+=`${((i/(N-1))*W).toFixed(1)},${(PAD+(1-(vals[i]-lo)/span)*(H-2*PAD)).toFixed(1)} `;
  const lines = breathStarts().filter(b=>b>=t0 && b<=t1).map(b=>{
    const x=((b-t0)/(t1-t0)*W).toFixed(1);
    return `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${lines}<polyline points="${pts.trim()}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
}
function renderSyncCheck(){
  const t0 = syncViewFrom, t1 = syncViewFrom + SYNC_VIEW_LEN;
  const oEit = syncOffsetOf('eit'), oEmg = syncOffsetOf('emg');
  const hasEmg = EMG_CHANNELS.length>0;
  // one reference lane for each ventilator signal in use
  const refs = [...new Set(['eit', ...(hasEmg ? ['emg'] : [])].map(m=>SYNC_STATE[m].ref))];
  const lanes = refs.map(id=>{
    const c = ventChan(id);
    const users = ['eit','emg'].filter(m=> SYNC_STATE[m].ref===id && (m!=='emg' || hasEmg)).map(m=>SYNC_MODS[m].name);
    return {mod:'vent', label:'Ventilator', sig:`${c.name} · reference for ${users.join(' and ')}`, fn:t=>c.fn(ventPhase(t))};
  });
  lanes.push({mod:'eit', label:'EIT', sig:`${syncSigName('eit')} · ${fmtOffset(oEit)}`, fn:t=>SYNC_MODS.eit.at(t-oEit, SYNC_STATE.eit.sig)});
  if(hasEmg) lanes.push({mod:'emg', label:'EMG', sig:`${syncSigName('emg')}, ECG removed · ${fmtOffset(oEmg)}`, fn:t=>SYNC_MODS.emg.at(t-oEmg, SYNC_STATE.emg.sig)});
  document.getElementById('syncCheckPlot').innerHTML = lanes.map(l=>`
    <div class="sync-lane mod-${l.mod}">
      <span class="slice-panel-label"><span class="src">${l.label}</span>${l.sig}</span>
      ${syncLaneSvg(l.fn, t0, t1, `var(--${l.mod})`)}
    </div>`).join('');
  let axis = '';
  for(let k=0;k<=4;k++) axis += `<span>${fmtSec(t0 + SYNC_VIEW_LEN*k/4)}</span>`;
  document.getElementById('syncCheckAxis').innerHTML = axis;
}
document.getElementById('syncViewSeg').addEventListener('click', e=>{
  const btn = e.target.closest('button[data-from]'); if(!btn) return;
  syncViewFrom = +btn.dataset.from;
  document.querySelectorAll('#syncViewSeg button').forEach(b=> b.classList.toggle('on', b===btn));
  renderSyncCheck();
});

// one change of an offset updates the bar, the match, the check and the lanes
function syncChanged(mod){
  updateSyncRow(mod);
  renderSyncCheck();
  if(typeof renderSliceLanes==='function') renderSliceLanes();
}
Object.keys(SYNC_IDS).forEach(mod=>{
  const input = document.getElementById(SYNC_IDS[mod].input);
  input.addEventListener('input', ()=>{
    const v = parseFloat(input.value.replace('−','-'));
    SYNC_STATE[mod].typed = (input.value.trim()==='' || isNaN(v)) ? null : v;
    syncChanged(mod);
  });
});
document.getElementById('syncStrip').addEventListener('click', e=>{
  const btn = e.target.closest('[data-estimate]'); if(!btn) return;
  const mod = btn.dataset.estimate;
  SYNC_STATE[mod].est = estimateOffset(mod);
  SYNC_STATE[mod].typed = null;
  document.getElementById(SYNC_IDS[mod].input).value = '';
  syncChanged(mod);
  showEdgeNote(btn, `${SYNC_MODS[mod].name} ${syncSigName(mod)} against ${ventChan(SYNC_STATE[mod].ref).name}: best match at ${fmtOffset(SYNC_STATE[mod].est)} (correlation ${syncMatch(mod, SYNC_STATE[mod].est).toFixed(2)}).`);
});

// ---- the signal of each row and the ventilator signal it is compared with ----
const SYNC_SELECTS = {eit:{sig:'syncSigEit', ref:'syncRefEit'}, emg:{sig:'syncSigEmg', ref:'syncRefEmg'}};
function fillSyncSelect(sel, choices, value){
  sel.innerHTML = choices.map(c=> `<option value="${c.id}">${c.name}</option>`).join('');
  sel.value = value;
  sel.disabled = !choices.length;
}
function renderSyncSelects(){
  Object.keys(SYNC_SELECTS).forEach(mod=>{
    const st = SYNC_STATE[mod], ids = SYNC_SELECTS[mod];
    const sigs = SYNC_MODS[mod].choices(), refs = syncRefChoices();
    if(!sigs.some(c=>c.id===st.sig)) st.sig = sigs[0] ? sigs[0].id : null;
    if(!refs.some(c=>c.id===st.ref)) st.ref = refs[0].id;
    fillSyncSelect(document.getElementById(ids.sig), sigs, st.sig);
    fillSyncSelect(document.getElementById(ids.ref), refs, st.ref);
  });
}
// a new signal gives a new estimate; a typed offset stays as typed
function syncSignalChanged(mod){
  SYNC_STATE[mod].est = estimateOffset(mod);
  syncChanged(mod);
}
Object.keys(SYNC_SELECTS).forEach(mod=>{
  ['sig','ref'].forEach(key=>{
    document.getElementById(SYNC_SELECTS[mod][key]).addEventListener('change', e=>{
      SYNC_STATE[mod][key] = e.target.value;
      syncSignalChanged(mod);
    });
  });
});
// EMG signals added or removed: new choices, and a new estimate if the chosen one is gone
function syncEmgChannelsChanged(){
  const before = SYNC_STATE.emg.sig;
  renderSyncSelects();
  if(SYNC_STATE.emg.sig!==before && SYNC_STATE.emg.sig) SYNC_STATE.emg.est = estimateOffset('emg');
  updateSyncRow('emg');
  renderSyncCheck();
}

// first estimate when the page opens; the EMG signals change when one is added or removed
placeSyncBar(document.getElementById('syncBarVent'), 0);
renderSyncSelects();
Object.keys(SYNC_STATE).forEach(mod=>{ SYNC_STATE[mod].est = estimateOffset(mod); updateSyncRow(mod); });
renderSyncCheck();
