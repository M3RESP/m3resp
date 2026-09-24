// ================= Ventilator multi-channel stack =================
// Breath-phase waveform model: a ventilator trace is not a sine wave, so each
// channel is generated from the same inspiratory/expiratory phase clock.
const VENT_DUR = REC_SECONDS;      // same session length as every other view
const VENT_RR  = 15.6;            // br/min  (matches ventilator.respiratory_rate)
const VENT_T   = 60 / VENT_RR;    // breath period
const VENT_TI  = 0.34;            // inspiratory fraction of the cycle
const VENT_PTS = 1100;

// phase helpers: ip = 0..1 through inspiration, ep = 0..1 through expiration
function ventPhase(t){
  const p = (t % VENT_T) / VENT_T;
  return p < VENT_TI
    ? {insp:true,  f:p / VENT_TI}
    : {insp:false, f:(p - VENT_TI) / (1 - VENT_TI)};
}

const VENT_CHANNELS = [
  {
    id:'paw', name:'P_aw', unit:'cmH₂O', cat:'airway_pressure', ch:'ch 0',
    color:'#8a67c9', on:true,
    fn:({insp,f})=> insp
      ? 5 + 15 * Math.min(1, f*3.2)                    // fast rise to plateau
      : 5 + 15 * Math.exp(-f*6.5)                      // decay back to PEEP
  },
  {
    id:'flow', name:'Flow', unit:'L/s', cat:'airflow', ch:'ch 1',
    color:'#4f8fd8', on:true,
    fn:({insp,f})=> insp
      ? 0.62 * (1 - 0.25*f)                            // decelerating inspiratory flow
      : -0.80 * Math.exp(-f*4.2)                       // passive expiratory decay
  },
  {
    id:'volume', name:'Volume', unit:'mL', cat:'volume', ch:'ch 2',
    color:'#3aa88b', on:true,
    fn:({insp,f})=> insp
      ? 480 * (f - 0.12*f*f)                           // integral of inspiratory flow
      : 480 * Math.exp(-f*3.6)
  },
  {
    id:'pes', name:'P_es', unit:'cmH₂O', cat:'esophageal_pressure', ch:'unmapped',
    color:'#c98a2c', on:false,
    fn:({insp,f})=> insp
      ? -2 - 6.5 * Math.sin(Math.PI*Math.min(1,f))     // negative swing on effort
      : -2 + 1.2 * (1 - Math.exp(-f*3))
  },
  {
    id:'pga', name:'P_ga', unit:'cmH₂O', cat:'gastric_pressure', ch:'unmapped',
    color:'#b25c8a', on:false,
    fn:({insp,f})=> insp
      ? 6 + 2.2 * Math.sin(Math.PI*Math.min(1,f))
      : 6 - 0.4 * Math.exp(-f*3)
  },
  {
    id:'pl', name:'P_L', unit:'cmH₂O', cat:'transpulmonary_pressure', ch:'derived',
    color:'#d1554f', on:false,
    // transpulmonary pressure is Paw - Pes by definition
    fn:(ph)=> VENT_CHANNELS[0].fn(ph) - VENT_CHANNELS[3].fn(ph)
  },
];

function buildVentPanel(chan){ return ventPanelFor(chan, 0, VENT_DUR); }
function ventPanelFor(chan, t0, t1){
  const W=1000, H=156, TOP=20, BOT=10;
  const vals=[];
  for(let i=0;i<VENT_PTS;i++){
    const t=t0+(i/(VENT_PTS-1))*(t1-t0);
    vals.push(chan.fn(ventPhase(t)));
  }
  let lo=Math.min(...vals), hi=Math.max(...vals);
  const pad=(hi-lo)*0.12 || 1; lo-=pad; hi+=pad;
  let pts='';
  for(let i=0;i<VENT_PTS;i++){
    const x=(i/(VENT_PTS-1))*W;
    const y=TOP+(1-(vals[i]-lo)/(hi-lo))*(H-TOP-BOT);
    pts+=`${x.toFixed(1)},${y.toFixed(1)} `;
  }
  // zero line, where zero falls inside the visible range (flow and Pes need it)
  let zeroLine='';
  if(lo < 0 && hi > 0){
    const zy=TOP+(1-(0-lo)/(hi-lo))*(H-TOP-BOT);
    zeroLine=`<line x1="0" y1="${zy.toFixed(1)}" x2="${W}" y2="${zy.toFixed(1)}" stroke="var(--border)" stroke-width="1" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>`;
  }
  const div=document.createElement('div');
  div.className='vent-panel';
  div.id='ventPanel-'+chan.id;
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
      <polyline points="${pts.trim()}" fill="none" stroke="${chan.color}" stroke-width="1.9" vector-effect="non-scaling-stroke"/>
    </svg>`;
  return div;
}

const ventStack=document.getElementById('ventStack');
const ventChecks=document.getElementById('ventChecks');

// panels are regenerated for whatever window the zoom view is showing
function renderVentWindow(t0,t1){
  const band=ventStack.querySelector('.zoom-band');
  ventStack.innerHTML='';
  VENT_CHANNELS.forEach(c=> ventStack.appendChild(ventPanelFor(c,t0,t1)));
  if(band) ventStack.appendChild(band);
  syncVentVisibility();
}

VENT_CHANNELS.forEach(c=>{
  const lab=document.createElement('label');
  lab.className='vent-check'+(c.on?' on':'');
  lab.innerHTML=`<input type="checkbox" ${c.on?'checked':''} data-ch="${c.id}">
    <span class="sw" style="background:${c.color}"></span>${c.name}
    <span class="cat">${c.cat}</span>`;
  ventChecks.appendChild(lab);
});

function syncVentVisibility(){
  VENT_CHANNELS.forEach(c=>{
    document.getElementById('ventPanel-'+c.id).classList.toggle('hidden', !c.on);
    const inp=ventChecks.querySelector(`input[data-ch="${c.id}"]`);
    inp.closest('.vent-check').classList.toggle('on', c.on);
  });
}
ventChecks.addEventListener('change', e=>{
  const inp=e.target.closest('input[data-ch]'); if(!inp) return;
  const c=VENT_CHANNELS.find(x=>x.id===inp.dataset.ch);
  c.on=inp.checked;
  syncVentVisibility();
  if(typeof renderSliceLanes==='function') renderSliceLanes();
});


