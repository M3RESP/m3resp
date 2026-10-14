// ================= Prepare tab: signal previews + working-window slicer =================
// Session length matches this EIT card's real recording: 60.0 s (draeger .bin).
// The crop strip spans exactly this, so each lane is the same trace its
// modality card shows — not a separate low-resolution stand-in.
const REC_SECONDS = 60.0;

function waveformPoints(w, h, {freqHz, amp, baseline, noise, wobble, rectify, seed}){
  const n = 220;
  let s = seed;
  const rnd = ()=>{ s = (s*1103515245+12345)&0x7fffffff; return s/0x7fffffff; };
  let pts = [];
  for(let i=0;i<=n;i++){
    const t = (i/n)*REC_SECONDS;
    const x = (i/n)*w;
    let y = Math.sin(2*Math.PI*freqHz*t) * amp * (1 + wobble*Math.sin(2*Math.PI*0.02*t));
    if(rectify) y = Math.abs(y)*0.92 + amp*0.06;
    y += (rnd()-0.5)*noise;
    const cy = baseline - y;
    pts.push(x.toFixed(1)+','+cy.toFixed(1));
  }
  return pts.join(' ');
}

function buildSignalSvg(svgEl, {stroke, freqHz, amp, baseline, noise, wobble, rectify, seed, h}){
  const w = 700;
  const pts = waveformPoints(w, h, {freqHz, amp, baseline, noise, wobble, rectify, seed});
  svgEl.innerHTML = `
    <line x1="0" y1="${baseline}" x2="${w}" y2="${baseline}" stroke="var(--border)" stroke-width="1" />
    <polyline points="${pts}" fill="none" stroke="${stroke}" stroke-width="2.6" />
  `;
  // window-crop overlays (one per selected window) are (re)drawn into this
  // same svg by updateMasks() whenever the working-window selection changes.
}



// ---- working-window overview lanes ----
// Each lane is the SAME generator its modality card uses, plotted over the
// full session — not a separate, lower-fidelity stand-in.
function overviewLaneSvg(fn, color, W, H){
  const N=1100, PAD=6;
  const vals=[];
  for(let i=0;i<N;i++) vals.push(fn((i/(N-1))*REC_SECONDS, i));
  let lo=Math.min(...vals), hi=Math.max(...vals);
  const span=(hi-lo)||1;
  let pts='';
  for(let i=0;i<N;i++){
    const x=(i/(N-1))*W;
    const y=PAD+(1-(vals[i]-lo)/span)*(H-2*PAD);
    pts+=`${x.toFixed(1)},${y.toFixed(1)} `;
  }
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio','none');
  svg.innerHTML=`<polyline points="${pts.trim()}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
  return svg;
}

// EIT global impedance over the session (the detail panel shows the real
// EIT excerpt; this overview spans the full session length)
const EIT_FULL = {"q":"4W6C7T8G8UcQfPivmJv5AZDjHgHbItJeJHJVKQKwJgIhFJDnzIxTvTvjsErPohk0hkfVdUcActaz9X5w543U4j3P78bfcsj0m0sLsDyLDcGIHfF4EPFRCOz6zKzQynxevosupJlwmvkeldhkfCd4929D5u4P7C6M676M5u4J487Mc0jbq6uWxuCOEzIQOcPjP2R0OSMhLSJIGMKBH5JQDwCSxNuUqIoEodn6mAiLfEdkc87N5G7O5v6E4E4F3Y3I6K8Ae7gbnsqwtKx4BMHvNWRzTmVLWHVDVJUHUlV0W6StQ9PHN6JyHNEvGfEmAhxdvNr6mel6lHjbfSe9bD9W6W4U7y6J4s4l4d67588MdHk2qSvqBwG1KyOYQ1T2ZvYTZZXJWATOQMQRO5NKMdJUEXB7xkugrCpOplmXjrigd08v7e6f5l6q7J3S4i5j6Da7eKljqtv6znDRJkLiPtRASKVGSxP9TaPZOMOkOUO6KEKQFCD8A0zpAzwov6snpNkTibhCghgEfVdfb58O5L3f4m5L7k5B4F5N7a8CbWllsfxxCVKOLvQ5OUNkQ9S3R4PqOfJQI7CcAZzMyFwZs0oQlQi5evd3bkfn887J5a503u344Y689Zd5g2lNpPuCwzELLVOtSLWBW2XeYSY5ZvZ3VqUWRnPbNbJhKpJ7HhDWB3wBuhrmqfozl1jYfkeZ9L8a7R6A8E7Q6a5Q2s4p2T7odQfrj7lKr6tPwNAkFSL0LmNVR1QoP0MxN0MbN7OjKaI1EeBjzXwryTx1rVqdokjgfOepeIdGd1aD7e8g2h373t2q89738yekfVkmq3tWEBGgKHPuRhS7PkQXTcSOQzPcNTLmF3DsDHD0BZwau7pUkBhQg8gudoao9E9j7P4g2U2l5T9BdNiFn7tLz2EZLaPARtPvMJN9LHIPHoF6AVx1xFskr9nCijgxfhbJ9za86O5f1V3T317w8WaIbifjjfnUryyOD3FFHnM6OeOTRkVkU9V1UKUCSSQNNxNhNhO9KdH8FVCxwIvisgrCp4lxjXfudMaIadb2aR8z7o5z3o1t2Y8dbie1iam3pDvawFCaFFMNOqOpNeKwIwJFJqHjIkHsEeC3AfwJtNsMqhqRpSmZj2fXamaOcgaSaQ8U7r6y42001Q506OcVd2gEmnr2wjF2K0P4S9SoVvUlUeTEVLSwQiO5LHHBDdBcyDwStuqKnylhhVcxbQb49u7g6Y5n4W4r5e7B","n":620,"lo":3.8503276592471475e-17,"hi":183.02750559232663,"dur":60.0};
let EIT_FULL_Q=null;   // decoded lazily; decodeB62() lives in js/prepare/eit-workspace.js
// real reconstructed global impedance, sampled anywhere in the recording
function eitOverviewAt(t){
  if(!EIT_FULL_Q) EIT_FULL_Q = decodeB62(EIT_FULL.q, EIT_FULL.n, 0);   // 0..1 fractions
  const x=Math.max(0, Math.min(EIT_FULL.n-1, (t/REC_SECONDS)*(EIT_FULL.n-1)));
  const i=Math.floor(x), f=x-i;
  const a=EIT_FULL_Q[i], b=EIT_FULL_Q[Math.min(EIT_FULL.n-1,i+1)];
  const frac=a+(b-a)*f;
  return EIT_FULL.lo + frac*(EIT_FULL.hi-EIT_FULL.lo);   // back to real impedance units
}

// ---- one breath clock for the example signals ----
// The breath starts are the lowest points of the real EIT global impedance
// (smoothed over 0.7 s; lowest within ±1.6 s, followed by a rise of more than
// 25 % of the full range). The example ventilator and EMG signals follow these
// breaths, so the three modalities line up once they are synchronized.
// breath starts of any impedance curve fn(t) over 0..dur seconds
function breathStartsOf(fn, dur){
  const dt=0.1, N=Math.round(dur/dt)+1, v=[];
  for(let k=0;k<N;k++){
    let s=0, c=0;
    for(let j=-3;j<=3;j++){ const t=(k+j)*dt; if(t>=0 && t<=dur){ s+=fn(t); c++; } }
    v.push(s/c);
  }
  const range = Math.max(...v) - Math.min(...v);
  const starts = [];
  for(let k=0;k<N;k++){
    let lowest = true;
    for(let j=-16;j<=16;j++){ const m=k+j; if(m>=0 && m<N && v[m]<v[k]){ lowest=false; break; } }
    if(!lowest) continue;
    let rise = 0;
    for(let j=1;j<=30 && k+j<N;j++) rise = Math.max(rise, v[k+j]-v[k]);
    if(rise > 0.25*range) starts.push(Math.round(k*dt*10)/10);
  }
  return starts;
}
let BREATH_STARTS = null;
function breathStarts(){
  if(!BREATH_STARTS) BREATH_STARTS = breathStartsOf(eitOverviewAt, REC_SECONDS);
  return BREATH_STARTS;
}
// where reference time t falls in its breath: p runs from 0 (breath start) to
// 1; before the first and after the last start, breaths repeat at the mean length
function breathPhaseAt(t){
  const b = breathStarts(), n = b.length, mean = (b[n-1]-b[0])/(n-1);
  let start, len;
  if(t < b[0]){ start = b[0] - Math.ceil((b[0]-t)/mean)*mean; len = mean; }
  else if(t >= b[n-1]){ start = b[n-1] + Math.floor((t-b[n-1])/mean)*mean; len = mean; }
  else {
    let lo=0, hi=n-1;
    while(hi-lo>1){ const mid=(lo+hi)>>1; if(b[mid]<=t) lo=mid; else hi=mid; }
    start = b[lo]; len = b[lo+1]-b[lo];
  }
  return {start, len, p:(t-start)/len};
}

// The lanes use the current synchronization offsets (js/prepare/sync-strip.js):
// a recording that starts `o` seconds after the reference is read at t − o.
function sliceLanes(){
  const off = mod=> (typeof syncOffsetOf==='function') ? syncOffsetOf(mod) : 0;
  const oEit = off('eit'), oEmg = off('emg');
  const lanes=[{mod:'eit', label:'EIT · global impedance', color:'var(--eit)', fn:t=>eitOverviewAt(t-oEit)}];
  EMG_CHANNELS.filter(c=>c.on).forEach(c=>{
    lanes.push({mod:'emg', label:'EMG · '+c.name, color:c.color, fn:(t,i)=>c.fn(t-oEmg, i)});
  });
  VENT_CHANNELS.filter(c=>c.on).forEach(c=>{
    lanes.push({mod:'vent', label:'Ventilator · '+c.name, color:c.color, fn:t=>c.fn(ventPhase(t))});
  });
  return lanes;
}

function renderSliceLanes(){
  const track=document.getElementById('sliceTrack');
  [...track.querySelectorAll('.slice-panel')].forEach(el=>el.remove());
  sliceLanes().forEach(l=>{
    const panel=document.createElement('div');
    panel.className='slice-panel mod-'+l.mod;
    panel.style.borderLeftColor=l.color;
    const lane=document.createElement('div');
    lane.className='slice-lane';
    lane.appendChild(overviewLaneSvg(l.fn, l.color, 1000, 96));
    panel.appendChild(lane);
    const lab=document.createElement('span');
    lab.className='slice-panel-label';
    // "EIT · global impedance": the modality in small mono, then the signal name
    const [src, ...rest]=l.label.split(' · ');
    lab.innerHTML=`<span class="src">${src}</span>${rest.join(' · ')}`;
    panel.appendChild(lab);
    track.appendChild(panel); // z-index (not DOM order) keeps windows above lanes
  });
}

// ---- working-window state ----
let sliceMode = 'time';
const EIT_HZ = 50;
// window colour coding: teal (--win-neutral, CSS: the accent) is the plain
// band, green (--ok) = kept, red = removed
const NEUTRAL_COLOR = '#0f8b8d';
const KEEP_COLOR = '#2f9e6e';

