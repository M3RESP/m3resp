// ================= EMGdi (channel 1, diaphragm sEMG) =================
const EMG_FULL_DUR = REC_SECONDS;   // same session length as every other view
const EMG_PTS = 1400;
const EMG_RR = 16.4;            // br/min, matches emg.evaluate_respiratory_rates
const EMG_HR = 72;              // bpm, the ECG that emg.ecg_gating removes

function emgRand(i){ const s=Math.sin(i*12.9898)*43758.5453; return s-Math.floor(s); }

// raw EMGdi: respiratory burst envelope * carrier noise, plus periodic ECG spikes
function emgRawAt(t, i){
  const burst = Math.max(0, Math.sin(2*Math.PI*(EMG_RR/60)*t));
  const env   = 0.12 + 0.88*Math.pow(burst, 1.6);
  const carrier = (emgRand(i)*2-1) + (emgRand(i*7.3)*2-1)*0.5;
  const ecgPhase = (t*(EMG_HR/60)) % 1;
  const ecg = Math.exp(-Math.pow((ecgPhase-0.12)/0.012, 2)) * 1.55
            - Math.exp(-Math.pow((ecgPhase-0.155)/0.010, 2)) * 0.55;
  return env*carrier*0.62 + ecg;
}

const EMG_CHANNELS = [
  {id:'emgdi_raw', name:'EMGdi (raw)', unit:'a.u.', cat:'electrical_potential', ch:'ch 1',
   color:'#c98a2c', signed:true,  fn:(t,i)=>emgRawAt(t,i)},
];

function emgPanelFor(chan, t0, t1){
  // EMGdi is the only trace in this section, so it gets the full card height
  const W=1000, H=190, TOP=20, BOT=10;
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
  div.style.borderLeftColor=chan.color;
  div.style.height=H+'px';
  div.innerHTML=`
    <div class="vent-panel-head">
      <span class="nm">${chan.name}</span>
      <span class="un">${chan.unit}</span>
      <span class="ch">${chan.cat} · ${chan.ch}</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${zeroLine}
      <polyline points="${pts.trim()}" fill="none" stroke="${chan.color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    </svg>`;
  return div;
}

const emgStack=document.getElementById('emgStack');
function renderEmgWindow(t0,t1){
  const band=emgStack.querySelector('.zoom-band');
  emgStack.innerHTML='';
  EMG_CHANNELS.forEach(c=> emgStack.appendChild(emgPanelFor(c,t0,t1)));
  if(band) emgStack.appendChild(band);
}

