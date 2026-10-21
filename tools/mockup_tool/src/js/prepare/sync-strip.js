// ---- raw synchronization: strip, match, estimate, alignment check ----
// The ventilator is the reference. Each other recording has an offset in
// seconds: the time of its first sample on the reference time line (a
// negative offset starts before the reference). A recording read at
// reference time t gives its own sample at t − offset.
//
// Each row chooses its recording: one loaded file of that card, or the merged
// recording of its sections (the Files blocks, js/prepare/file-merge.js).
// Each row also chooses its signal (EIT: global impedance or one ROI; EMG:
// one of the loaded EMG signals, as its envelope: RMS over 0.3 s, ECG
// removed) and the ventilator signal it is compared with. The estimate is
// the cross-correlation of the two over −5.0 to +5.0 s in steps of 0.1 s, on
// the reference times where both recordings have data.
// "Match" is the correlation at the offset in use (1 = the same shape).
const SYNC_SEARCH = {lo:-5, hi:5, step:0.1};
// edge: seconds left out at each end of the common part; dt: step of the
// comparison; min: shortest common part (s) that gives a match
const SYNC_TEST = {edge:1.0, dt:0.1, min:3.0};
// seconds shown on the strip; the end follows the longest recording
const SYNC_VIEW = {t0:-6, t1:REC_SECONDS+6};
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

// ---- the recording of each row: one loaded file or the merged recording ----
// SYNC_SRC holds 'f1', 'f2', … or 'merged' for each modality.
const SYNC_SRC = {vent:'f1', eit:'f1', emg:'f1'};
// the Files block of a modality; the EMG block exists once
// js/prepare/emg-files.js has loaded
function syncMerger(mod){
  if(mod==='vent') return VENT_MERGE;
  if(mod==='eit') return EIT_MERGE;
  return typeof EMG_MERGE==='undefined' ? null : EMG_MERGE;
}
// the recording of a row: its length dur (s), and at(t), the file and the
// file time of recording time t (held inside the recording)
function syncSource(mod){
  const M = syncMerger(mod);
  if(!M) return {dur:REC_SECONDS, at:t=>({file:null, t:Math.max(0, Math.min(REC_SECONDS, t))})};
  if(SYNC_SRC[mod]==='merged' && M.sections.length){
    const pieces = M.mergedPieces(), last = pieces[pieces.length-1];
    return {
      dur: last.from + (last.end-last.start),
      at: t=>{
        let p = pieces[0];
        for(const q of pieces) if(t >= q.from) p = q;
        return {file:p.file, t:Math.max(p.start, Math.min(p.end, p.start + (t - p.from)))};
      },
    };
  }
  const f = M.file(SYNC_SRC[mod]) || M.files[0];
  return {dur:f.dur, at:t=>({file:f, t:Math.max(0, Math.min(f.dur, t))})};
}
function syncSourceChoices(mod){
  const M = syncMerger(mod);
  if(!M) return [{id:'f1', name:'File 1'}];
  const files = M.files.map((f, i)=>({id:f.id, name:`File ${i+1}`, title:`${f.name} · ${fmtSec(f.dur)}`}));
  if(!M.sections.length) return files;
  const n = M.sections.length;
  return files.concat({id:'merged', name: M.merged ? 'Merged' : 'Merged (preview)',
    title:`${n} section${n===1?'':'s'} joined in merge order · ${fmtSec(M.mergedLength())}`});
}
function syncSourceName(mod){
  const c = syncSourceChoices(mod).find(x=>x.id===SYNC_SRC[mod]);
  return c ? c.name : '';
}

// fn(file, file time, i) of modality mod at reference time t, with the
// recording and the offset of its row; null where the recording has no data.
// The working-window lanes use it (js/prepare/signal-lanes.js).
function syncAtRef(mod, fn){
  const S = syncSource(mod), o = mod==='vent' ? 0 : syncOffsetOf(mod);
  return (t, i)=>{
    const tau = t - o;
    if(tau<0 || tau>S.dur) return null;
    const p = S.at(tau);
    return fn(p.file, p.t, i);
  };
}

const ventChan = id=> VENT_CHANNELS.find(c=>c.id===id);
// ventilator signals read from a file channel (P_es and P_ga are unmapped here)
const syncRefChoices = ()=> VENT_CHANNELS.filter(c=> /^ch /.test(c.ch)).map(c=>({id:c.id, name:c.name}));
// ventilator signal `id` of recording R at recording time t
function syncRefAt(R, id, t){
  const p = R.at(t);
  return ventChan(id).fn(ventPhase(ventFileAt(p.file, p.t).tau));
}
// at(S, t, sig): signal sig of recording S at recording time t
const SYNC_MODS = {
  eit: {name:'EIT',
        choices:()=> [{id:'global', name:'Global impedance'}, ...ROI_KEYS.map((k,i)=>({id:k, name:`ROI ${i+1}`}))],
        at:(S, t, sig)=>{
          const p = S.at(t);
          if(sig==='global') return p.file.fn(p.t);
          const a = fileDetailOf(p.file).raw[sig];
          return a[Math.max(0, Math.min(a.length-1, Math.round(p.t*DETAIL_HZ)))];
        }},
  emg: {name:'EMG',
        choices:()=> EMG_CHANNELS.map(c=>({id:c.id, name:`${c.name} envelope`})),
        at:(S, t, sig)=>{
          const p = S.at(t), q = p.file ? emgFileAt(p.file, p.t) : {tau:p.t, gain:1};
          return q.gain * emgEnvelopeAt(q.tau, sig);
        }},
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
// the reference signal and the modality on the reference times where both
// recordings have data, for one offset
function syncSeries(mod, offset){
  const m = SYNC_MODS[mod], st = SYNC_STATE[mod], R = syncSource('vent'), S = syncSource(mod), a=[], b=[];
  const lo = Math.max(0, offset) + SYNC_TEST.edge, hi = Math.min(R.dur, offset + S.dur) - SYNC_TEST.edge;
  for(let t=lo; t<=hi; t+=SYNC_TEST.dt){
    a.push(syncRefAt(R, st.ref, t));
    b.push(m.at(S, t - offset, st.sig));
  }
  return [a, b];
}
function syncMatch(mod, offset){
  const [a, b] = syncSeries(mod, offset);
  return a.length*SYNC_TEST.dt >= SYNC_TEST.min ? correlation(a, b) : 0;
}
function estimateOffset(mod){
  let best = {o:0, r:-2};
  for(let o=SYNC_SEARCH.lo; o<=SYNC_SEARCH.hi+1e-9; o+=SYNC_SEARCH.step){
    const r = syncMatch(mod, o);
    if(r > best.r) best = {o:Math.round(o*10)/10, r};
  }
  return best.o;
}

// offsets (s) that keep a bar of `dur` seconds on the strip
function syncOffsetRange(dur){
  return {lo:SYNC_VIEW.t0, hi:Math.max(-SYNC_VIEW.t0, SYNC_VIEW.t1 + SYNC_VIEW.t0 - dur)};
}
function placeSyncBar(bar, offset, dur){
  const r = syncOffsetRange(dur), shown = Math.max(r.lo, Math.min(r.hi, offset));
  bar.style.left = (syncFrac(shown)*100)+'%';
  bar.style.width = (dur/(SYNC_VIEW.t1-SYNC_VIEW.t0)*100)+'%';
}

// the strip shows the longest recording; the ruler, the reference bar and a
// thin line in each track at 0 s of the reference follow it
function drawSyncStrip(){
  SYNC_VIEW.t1 = Math.max(...Object.keys(SYNC_SRC).map(mod=> syncSource(mod).dur)) + 6;
  document.getElementById('syncStrip').style.setProperty('--sync-zero', (syncFrac(0)*100)+'%');
  const step = SYNC_VIEW.t1 > 100 ? 20 : 10;
  let out = '';
  for(let t=0; t<=SYNC_VIEW.t1-6; t+=step) out += `<span style="left:${(syncFrac(t)*100).toFixed(2)}%">${fmtSec(t)}</span>`;
  document.getElementById('syncRuler').innerHTML = out;
  placeSyncBar(document.getElementById('syncBarVent'), 0, syncSource('vent').dur);
}

const SYNC_IDS = {eit:{input:'syncOffsetEit', bar:'syncBarEit', match:'syncMatchEit'},
                  emg:{input:'syncOffsetEmg', bar:'syncBarEmg', match:'syncMatchEmg'}};
function fmtOffset(sec){ return `${sec>0?'+':sec<0?'−':''}${Math.abs(sec).toFixed(1)} s`; }

function updateSyncRow(mod){
  const ids = SYNC_IDS[mod], st = SYNC_STATE[mod];
  const input = document.getElementById(ids.input), bar = document.getElementById(ids.bar);
  const sec = syncOffsetOf(mod), typed = st.typed!==null;
  input.placeholder = st.est.toFixed(1);
  placeSyncBar(bar, sec, syncSource(mod).dur);
  bar.querySelector('.v').textContent = fmtOffset(sec);
  const tag = bar.querySelector('.tag');
  tag.textContent = typed ? 'typed' : 'estimated';
  tag.classList.toggle('typed', typed);
  const r = syncMatch(mod, sec), m = document.getElementById(ids.match);
  m.textContent = r.toFixed(2);
  m.className = 'sync-match ' + (r>=0.8 ? 'good' : r>=0.5 ? 'fair' : 'poor');
  m.title = `Correlation of ${SYNC_MODS[mod].name} ${syncSigName(mod)} (${syncSourceName(mod)}) with ventilator ${ventChan(st.ref).name} (${syncSourceName('vent')}) at ${fmtOffset(sec)}`;
}

// ---- alignment check: the ventilator signals, EIT and the EMG envelope on one time line ----
// The lanes show the whole reference recording at first; the zoom view 'sync'
// (js/prepare/zoom-engine.js) sets the time range shown. A lane is empty
// where its recording has no data.
function syncLaneSvg(fn, t0, t1, color, starts){
  const N=1200, W=1000, H=54, PAD=5, vals=[];
  for(let i=0;i<N;i++) vals.push(fn(t0+(i/(N-1))*(t1-t0)));
  const known = vals.filter(v=> v!==null);
  const lo=Math.min(...known), hi=Math.max(...known), span=(hi-lo)||1;
  // one polyline for each stretch with data
  const runs=[]; let pts='';
  for(let i=0;i<N;i++){
    if(vals[i]===null){ if(pts) runs.push(pts); pts=''; continue; }
    pts+=`${((i/(N-1))*W).toFixed(1)},${(PAD+(1-(vals[i]-lo)/span)*(H-2*PAD)).toFixed(1)} `;
  }
  if(pts) runs.push(pts);
  const lines = starts.filter(b=>b>=t0 && b<=t1).map(b=>{
    const x=((b-t0)/(t1-t0)*W).toFixed(1);
    return `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="var(--text-faint)" stroke-width="1" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>`;
  }).join('');
  const polys = runs.map(p=> `<polyline points="${p.trim()}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${lines}${polys}</svg>`;
}
// breath starts of the reference recording in reference time: the places
// where the breath clock of its ventilator signals starts a new breath
function syncBreathStarts(R, t0, t1){
  const out=[]; let prev=null;
  for(let t=Math.max(0, t0); t<=Math.min(R.dur, t1); t+=0.05){
    const p = R.at(t), ph = breathPhaseAt(ventFileAt(p.file, p.t).tau).p;
    if(prev!==null && ph < prev-0.5) out.push(t);
    prev = ph;
  }
  return out;
}
function renderSyncWindow(t0, t1){
  const oEit = syncOffsetOf('eit'), oEmg = syncOffsetOf('emg');
  const R = syncSource('vent'), sEit = syncSource('eit'), sEmg = syncSource('emg');
  const hasEmg = EMG_CHANNELS.length>0;
  // the value fn() of recording S at its time tau, or null outside the recording
  const within = (S, tau, fn)=> (tau>=0 && tau<=S.dur) ? fn() : null;
  // one lane for each ticked ventilator signal and each ventilator signal a
  // row is compared with, in the order of the ventilator card
  const refs = ['eit', ...(hasEmg ? ['emg'] : [])].map(m=>SYNC_STATE[m].ref);
  const lanes = VENT_CHANNELS.filter(c=> c.on || refs.includes(c.id)).map(c=>{
    const users = ['eit','emg'].filter(m=> SYNC_STATE[m].ref===c.id && (m!=='emg' || hasEmg)).map(m=>SYNC_MODS[m].name);
    return {mod:'vent', color:c.color, label:'Ventilator',
            sig:`${syncSourceName('vent')} · ${c.name}${users.length ? ` · reference for ${users.join(' and ')}` : ''}`,
            fn:t=> within(R, t, ()=> syncRefAt(R, c.id, t))};
  });
  lanes.push({mod:'eit', label:'EIT', sig:`${syncSourceName('eit')} · ${syncSigName('eit')} · ${fmtOffset(oEit)}`,
              fn:t=> within(sEit, t-oEit, ()=> SYNC_MODS.eit.at(sEit, t-oEit, SYNC_STATE.eit.sig))});
  if(hasEmg) lanes.push({mod:'emg', label:'EMG', sig:`${syncSourceName('emg')} · ${syncSigName('emg')}, ECG removed · ${fmtOffset(oEmg)}`,
              fn:t=> within(sEmg, t-oEmg, ()=> SYNC_MODS.emg.at(sEmg, t-oEmg, SYNC_STATE.emg.sig))});
  const starts = syncBreathStarts(R, t0, t1);
  const plot = document.getElementById('syncCheckPlot');
  const band = plot.querySelector('.zoom-band');
  // each lane: its name above, then the plot
  plot.innerHTML = lanes.map(l=>`
    <div class="sync-lane-wrap">
      <span class="slice-panel-label"><span class="src">${l.label}</span>${l.sig}</span>
      <div class="sync-lane mod-${l.mod}">${syncLaneSvg(l.fn, t0, t1, l.color || `var(--${l.mod})`, starts)}</div>
    </div>`).join('');
  if(band) plot.appendChild(band);
}
// draws the lanes again for the time range shown; the zoom view draws them
// the first time when js/prepare/zoom-engine.js loads
function renderSyncCheck(){
  if(ZOOM.sync) ZOOM.sync.refresh();
}

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

// ---- drag a bar: the recording slides along the reference time line ----
// A drag or an arrow key sets a typed offset, in steps of 0.1 s; arrow keys
// move it by 0.1 s, with Shift by 1.0 s. During a drag the bar, the match and
// the alignment check follow the mouse; the working-window lanes follow when
// the mouse is released.
function setSyncOffset(mod, sec, {lanes=true}={}){
  const r = syncOffsetRange(syncSource(mod).dur);
  const v = Math.round(Math.max(r.lo, Math.min(r.hi, sec))*10)/10;
  SYNC_STATE[mod].typed = v;
  document.getElementById(SYNC_IDS[mod].input).value = v.toFixed(1);
  updateSyncRow(mod);
  renderSyncCheck();
  if(lanes && typeof renderSliceLanes==='function') renderSliceLanes();
}
Object.keys(SYNC_IDS).forEach(mod=>{
  const bar = document.getElementById(SYNC_IDS[mod].bar);
  bar.tabIndex = 0;
  bar.title = 'Drag to move this recording on the ventilator time line; arrow keys move it by 0.1 s (with Shift: 1.0 s)';
  bar.setAttribute('aria-label', `${SYNC_MODS[mod].name} offset: drag or use the arrow keys`);
  bar.addEventListener('mousedown', e=>{
    if(e.button!==0) return;
    e.preventDefault();
    const track = bar.parentElement.getBoundingClientRect();
    const secPerPx = (SYNC_VIEW.t1 - SYNC_VIEW.t0) / track.width;
    const x0 = e.clientX, o0 = syncOffsetOf(mod);
    let frame = 0, last = o0;
    bar.classList.add('dragging');
    const move = ev=>{
      last = o0 + (ev.clientX - x0)*secPerPx;
      if(!frame) frame = requestAnimationFrame(()=>{ frame = 0; setSyncOffset(mod, last, {lanes:false}); });
    };
    const up = ()=>{
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
      if(frame) cancelAnimationFrame(frame);
      bar.classList.remove('dragging');
      if(last!==o0) setSyncOffset(mod, last);
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  });
  bar.addEventListener('keydown', e=>{
    if(e.key!=='ArrowLeft' && e.key!=='ArrowRight') return;
    e.preventDefault();
    const step = (e.shiftKey ? 1.0 : 0.1) * (e.key==='ArrowLeft' ? -1 : 1);
    setSyncOffset(mod, syncOffsetOf(mod) + step);
  });
});
// "Estimate" is active with the method estimate_offset; it finds the offset
// of each row again and clears the typed values
const syncEstimateBtn = document.getElementById('syncEstimate');
function updateSyncEstimateButton(){
  syncEstimateBtn.disabled = !document.getElementById('syncMethod').value.startsWith('estimate_offset');
}
document.getElementById('syncMethod').addEventListener('change', updateSyncEstimateButton);
syncEstimateBtn.addEventListener('click', ()=>{
  const mods = Object.keys(SYNC_IDS).filter(mod=> mod!=='emg' || EMG_CHANNELS.length);
  mods.forEach(mod=>{
    SYNC_STATE[mod].est = estimateOffset(mod);
    SYNC_STATE[mod].typed = null;
    document.getElementById(SYNC_IDS[mod].input).value = '';
    syncChanged(mod);
  });
  showEdgeNote(syncEstimateBtn, mods.map(mod=> `${SYNC_MODS[mod].name} ${syncSigName(mod)} against ${ventChan(SYNC_STATE[mod].ref).name}: best match at ${fmtOffset(SYNC_STATE[mod].est)} (correlation ${syncMatch(mod, SYNC_STATE[mod].est).toFixed(2)}).`).join(' '));
});

// ---- the signal of each row and the ventilator signal it is compared with ----
const SYNC_SELECTS = {eit:{sig:'syncSigEit', ref:'syncRefEit'}, emg:{sig:'syncSigEmg', ref:'syncRefEmg'}};
function fillSyncSelect(sel, choices, value){
  sel.innerHTML = choices.map(c=> `<option value="${c.id}"${c.title ? ` title="${c.title}"` : ''}>${c.name}</option>`).join('');
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

// ---- the recording of each row ----
// a list on each synchronization row, and the same list on the working window
const SYNC_SRC_SELECTS = {vent:'syncSrcVent', eit:'syncSrcEit', emg:'syncSrcEmg'};
// the same lists on the working window, with "None" in addition: a modality
// set to None is left out of the working window and of the sequences saved
// from it. Its synchronization stays as it is.
const SLICE_SRC_SELECTS = {vent:'sliceSrcVent', eit:'sliceSrcEit', emg:'sliceSrcEmg'};
const SLICE_OFF = {vent:false, eit:false, emg:false};
// the choices follow the files and sections of each card; a row whose
// recording is gone uses the first file
function renderSyncSources(){
  Object.keys(SYNC_SRC_SELECTS).forEach(mod=>{
    const choices = syncSourceChoices(mod);
    if(!choices.some(c=>c.id===SYNC_SRC[mod])) SYNC_SRC[mod] = choices[0].id;
    fillSyncSelect(document.getElementById(SYNC_SRC_SELECTS[mod]), choices, SYNC_SRC[mod]);
    fillSyncSelect(document.getElementById(SLICE_SRC_SELECTS[mod]),
      choices.concat({id:'none', name:'None', title:'Leave this modality out of the working window'}),
      SLICE_OFF[mod] ? 'none' : SYNC_SRC[mod]);
  });
}
// draws the strip, the rows, the check and the working window again
let syncRefDur = REC_SECONDS;
function refreshSync(){
  renderSyncSources();
  drawSyncStrip();
  Object.keys(SYNC_IDS).forEach(updateSyncRow);
  const dur = syncSource('vent').dur;
  if(ZOOM.sync && dur!==syncRefDur) ZOOM.sync.setFullDur(dur);
  else renderSyncCheck();
  syncRefDur = dur;
  if(typeof renderSliceLanes==='function') renderSliceLanes();
  if(typeof sliceDurChanged==='function') sliceDurChanged();
}
// a new recording gives a new estimate: for its own row, or for both rows
// when the reference changes; a typed offset stays as typed
function syncSourceChanged(mod){
  (mod==='vent' ? Object.keys(SYNC_IDS) : [mod]).forEach(m=>{ SYNC_STATE[m].est = estimateOffset(m); });
  refreshSync();
}
Object.keys(SYNC_SRC_SELECTS).forEach(mod=>{
  document.getElementById(SYNC_SRC_SELECTS[mod]).addEventListener('change', e=>{
    SYNC_SRC[mod] = e.target.value;
    syncSourceChanged(mod);
  });
  document.getElementById(SLICE_SRC_SELECTS[mod]).addEventListener('change', e=>{
    const v = e.target.value;
    SLICE_OFF[mod] = v==='none';
    if(v!=='none' && v!==SYNC_SRC[mod]){ SYNC_SRC[mod] = v; syncSourceChanged(mod); return; }
    if(typeof renderSliceLanes==='function') renderSliceLanes();
  });
});
// the files or sections of a card changed (js/prepare/file-merge.js); with
// `pick`, the row uses that recording ("Merge sections" picks 'merged')
function syncFilesChanged(mod, pick){
  const before = SYNC_SRC[mod];
  if(pick) SYNC_SRC[mod] = pick;
  renderSyncSources();
  // the merged recording follows its sections
  if(SYNC_SRC[mod]!==before || SYNC_SRC[mod]==='merged') syncSourceChanged(mod);
}

// first estimate when the page opens; the EMG signals change when one is added or removed
renderSyncSources();
renderSyncSelects();
drawSyncStrip();
Object.keys(SYNC_STATE).forEach(mod=>{ SYNC_STATE[mod].est = estimateOffset(mod); updateSyncRow(mod); });
updateSyncEstimateButton();
renderSyncCheck();
