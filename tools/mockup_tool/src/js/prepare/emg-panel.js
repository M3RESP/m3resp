// ================= EMG signals (one or more muscles, stacked) =================
const EMG_FULL_DUR = REC_SECONDS;   // same session length as every other view
const EMG_PTS = 1400;
const EMG_RR = 16.4;            // br/min, matches emg.evaluate_respiratory_rates
const EMG_HR = 72;              // bpm, the ECG that emg.ecg_gating removes

function emgRand(i){ const s=Math.sin(i*12.9898)*43758.5453; return s-Math.floor(s); }

// How each muscle looks in this mockup (illustrative, not measured):
//   amp   - size of the breathing bursts
//   ecg   - how much ECG leaks in (parasternal sits closest to the heart)
//   lag   - burst start relative to the diaphragm, in seconds (negative = earlier)
//   sharp - higher = the muscle only fires near the top of each breath
//   expir - true for muscles active during breathing out (abdominal)
const EMG_MUSCLES = {
  di:   {short:'EMGdi',   long:'diaphragm',          amp:0.62, ecg:1.00, lag: 0.00, sharp:1.6, color:'#c98a2c'},
  para: {short:'EMGpara', long:'parasternal',        amp:0.45, ecg:1.60, lag:-0.12, sharp:1.8, color:'#d1554f'},
  scm:  {short:'EMGscm',  long:'sternocleidomastoid',amp:0.30, ecg:0.35, lag: 0.10, sharp:3.2, color:'#b25c8a'},
  scal: {short:'EMGscal', long:'scalene',            amp:0.34, ecg:0.50, lag: 0.05, sharp:2.4, color:'#8a67c9'},
  abd:  {short:'EMGabd',  long:'abdominal',          amp:0.40, ecg:0.40, lag: 0.00, sharp:2.0, color:'#3aa88b', expir:true},
  alae: {short:'EMGan',   long:'alae nasi',          amp:0.26, ecg:0.20, lag:-0.30, sharp:1.4, color:'#4f8fd8'},
};

// raw EMG: breathing burst envelope * noise carrier, plus periodic ECG spikes
function emgRawAt(t, i, muscle=EMG_MUSCLES.di, seed=0){
  const phase = 2*Math.PI*(EMG_RR/60)*(t - muscle.lag);
  const wave  = Math.sin(phase);
  const burst = Math.max(0, muscle.expir ? -wave : wave);
  const env   = 0.12 + 0.88*Math.pow(burst, muscle.sharp);
  const k = i + seed*1009;
  const carrier = (emgRand(k)*2-1) + (emgRand(k*7.3)*2-1)*0.5;
  const ecgPhase = (t*(EMG_HR/60)) % 1;
  const ecg = Math.exp(-Math.pow((ecgPhase-0.12)/0.012, 2)) * 1.55
            - Math.exp(-Math.pow((ecgPhase-0.155)/0.010, 2)) * 0.55;
  return env*carrier*muscle.amp + ecg*muscle.ecg;
}

let emgSeq = 0;
function makeEmgChannel(muscleKey, side, ch){
  const m = EMG_MUSCLES[muscleKey];
  const seed = ++emgSeq;          // each signal gets its own noise
  const id = 'emg_'+seed;
  return {
    id, muscleKey, side, ch, on:true,
    name: m.short + (side ? ' '+side : ''),
    desc: m.long + (side ? (side==='L' ? ', left' : ', right') : ''),
    unit:'µV', cat:'electrical_potential', color:m.color,
    fn:(t,i)=>emgRawAt(t, i, m, seed),
  };
}

// start with the diaphragm signal on channel 1, as in the example recording
const EMG_CHANNELS = [ makeEmgChannel('di', '', 'ch 1') ];

function emgPanelFor(chan, t0, t1, H){
  const W=1000, TOP=20, BOT=10;
  const vals=[];
  for(let i=0;i<EMG_PTS;i++){
    const t=t0+(i/(EMG_PTS-1))*(t1-t0);
    vals.push(chan.fn(t, i + Math.floor(t0*997)));
  }
  let lo=Math.min(...vals), hi=Math.max(...vals);
  const pad=(hi-lo)*0.1 || 1; lo-=pad; hi+=pad;
  let pts='';
  for(let i=0;i<EMG_PTS;i++){
    const x=(i/(EMG_PTS-1))*W;
    const y=TOP+(1-(vals[i]-lo)/(hi-lo))*(H-TOP-BOT);
    pts+=`${x.toFixed(1)},${y.toFixed(1)} `;
  }
  let zeroLine='';
  if(lo<0 && hi>0){
    const zy=TOP+(1-(0-lo)/(hi-lo))*(H-TOP-BOT);
    zeroLine=`<line x1="0" y1="${zy.toFixed(1)}" x2="${W}" y2="${zy.toFixed(1)}" stroke="var(--border)" stroke-width="1" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>`;
  }
  const div=document.createElement('div');
  div.className='vent-panel';
  div.id='emgPanel-'+chan.id;
  div.style.borderLeftColor=chan.color;
  div.style.height=H+'px';
  div.innerHTML=`
    <div class="vent-panel-head">
      <span class="nm">${chan.name}</span>
      <span class="un">${chan.desc} · ${chan.unit}</span>
      <span class="ch">${chan.cat} · ${chan.ch}</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${zeroLine}
      <polyline points="${pts.trim()}" fill="none" stroke="${chan.color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    </svg>`;
  return div;
}

const emgStack=document.getElementById('emgStack');
const emgChecks=document.getElementById('emgChecks');

function renderEmgWindow(t0,t1){
  const band=emgStack.querySelector('.zoom-band');
  emgStack.innerHTML='';
  const shown=EMG_CHANNELS.filter(c=>c.on);
  // one signal gets the full card height; several share it, below each other
  const H = shown.length<=1 ? 190 : 124;
  shown.forEach(c=> emgStack.appendChild(emgPanelFor(c,t0,t1,H)));
  if(!shown.length){
    const empty=document.createElement('div');
    empty.className='slice-note';
    empty.style.cssText='padding:28px;text-align:center;border:1px dashed var(--border);border-radius:7px;';
    empty.textContent = EMG_CHANNELS.length
      ? 'All EMG signals are hidden — tick one above to show it.'
      : 'No EMG signals yet — add one above.';
    emgStack.appendChild(empty);
  }
  if(band) emgStack.appendChild(band);
}

// ---- signal chips: tick to show/hide, × to remove ----
function renderEmgChecks(){
  emgChecks.innerHTML='';
  EMG_CHANNELS.forEach(c=>{
    const lab=document.createElement('label');
    lab.className='vent-check'+(c.on?' on':'');
    lab.title=`${c.name} — ${c.desc}, ${c.ch}`;
    lab.innerHTML=`<input type="checkbox" ${c.on?'checked':''} data-ch="${c.id}">
      <span class="sw" style="background:${c.color}"></span>${c.name}
      <span class="cat">${c.ch}</span>
      <button class="rm" data-rm="${c.id}" title="Remove this signal">×</button>`;
    emgChecks.appendChild(lab);
  });
}

function emgChanged(){
  renderEmgChecks();
  if(ZOOM.emg) ZOOM.emg.refresh();
  if(typeof renderSliceLanes==='function') renderSliceLanes();
}

emgChecks.addEventListener('click', e=>{
  const rm=e.target.closest('[data-rm]'); if(!rm) return;
  e.preventDefault();
  const idx=EMG_CHANNELS.findIndex(c=>c.id===rm.dataset.rm);
  if(idx>=0) EMG_CHANNELS.splice(idx,1);
  emgChanged();
});
emgChecks.addEventListener('change', e=>{
  const inp=e.target.closest('input[data-ch]'); if(!inp) return;
  EMG_CHANNELS.find(c=>c.id===inp.dataset.ch).on=inp.checked;
  emgChanged();
});

// ---- add a new EMG signal ----
const emgAddMuscle=document.getElementById('emgAddMuscle');
const emgAddSide=document.getElementById('emgAddSide');
const emgAddChannel=document.getElementById('emgAddChannel');
emgAddMuscle.value='para';

// move the channel picker to the first channel not used yet
function pickFreeEmgChannel(){
  const used=new Set(EMG_CHANNELS.map(c=>c.ch));
  const free=[...emgAddChannel.options].find(o=>!used.has(o.value));
  if(free) emgAddChannel.value=free.value;
}

document.getElementById('emgAddBtn').addEventListener('click', e=>{
  const btn=e.currentTarget;
  const ch=emgAddChannel.value;
  const taken=EMG_CHANNELS.find(c=>c.ch===ch);
  if(taken){
    showEdgeNote(btn, `${ch} is already shown as ${taken.name}. Pick another source channel, or remove ${taken.name} first.`);
    return;
  }
  EMG_CHANNELS.push(makeEmgChannel(emgAddMuscle.value, emgAddSide.value, ch));
  pickFreeEmgChannel();
  emgChanged();
});

renderEmgChecks();
pickFreeEmgChannel();
