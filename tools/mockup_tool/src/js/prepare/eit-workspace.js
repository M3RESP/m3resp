// ================= EIT analysis workspace (real .bin reconstruction, via eitprocessing) =================
// shared zoom-view registry, declared up front because the EIT cursor reads it
const ZOOM = {};
// EIT (the frame/mask blob) is declared in js/data/eit-frame-data.js, which the
// build orders ahead of this file.

const B62="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const B62IDX={}; for(let i=0;i<B62.length;i++) B62IDX[B62[i]]=i;
function decodeB62(str, count, offset){
  const out=new Float32Array(count);
  for(let i=0;i<count;i++){
    const j=(offset+i)*2;
    out[i]=(B62IDX[str[j]]*62+B62IDX[str[j+1]])/3843;   // 0..1
  }
  return out;
}

// blue -> white ramp matching the colorbar
function eitColor(v){
  const stops=[[0.00,11,20,48],[0.28,23,58,143],[0.58,63,111,219],[0.82,169,196,244],[1.00,255,255,255]];
  v=Math.max(0,Math.min(1,v));
  for(let i=0;i<stops.length-1;i++){
    const a=stops[i],b=stops[i+1];
    if(v<=b[0]){
      const f=(v-a[0])/(b[0]-a[0]||1);
      return `rgb(${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)},${Math.round(a[3]+(b[3]-a[3])*f)})`;
    }
  }
  return 'rgb(255,255,255)';
}

// ---- build a real 32x32 pixel-impedance grid; returns array of <rect> (or null
// for pixels outside the circular chest mask) indexed exactly like EIT.frames ----
const G = EIT.grid;
const maskedRows = [...Array(G).keys()].filter(ry => EIT.mask.slice(ry*G,(ry+1)*G).includes('1'));
const rowMin = Math.min(...maskedRows), rowMax = Math.max(...maskedRows);
const bandEdges = [0,1,2,3,4].map(k => rowMin + (rowMax+1-rowMin)*k/4);
function rowBand(ry){ for(let b=0;b<4;b++) if(ry < bandEdges[b+1] || b===3) return b; }

function buildPixelGrid(container, {showRoiLines, showRoiLabels}){
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox',`0 0 ${G} ${G}`);
  const rects=[];
  for(let ry=0; ry<G; ry++){
    for(let rx=0; rx<G; rx++){
      const idx = ry*G+rx;
      if(EIT.mask[idx]==='1'){
        const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
        rect.setAttribute('x',rx); rect.setAttribute('y',ry);
        rect.setAttribute('width',1); rect.setAttribute('height',1);
        rect.setAttribute('fill','#0b1430');
        rect.setAttribute('shape-rendering','crispEdges');
        svg.appendChild(rect);
        rects.push(rect);
      } else {
        rects.push(null);
      }
    }
  }
  if(showRoiLines){
    for(let b=1;b<4;b++){
      const y=bandEdges[b];
      const line=document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1',0); line.setAttribute('y1',y); line.setAttribute('x2',G); line.setAttribute('y2',y);
      line.setAttribute('stroke','#7d8899'); line.setAttribute('stroke-width',0.14);
      svg.appendChild(line);
    }
  }
  const R=G/2;
  const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
  circle.setAttribute('cx',R); circle.setAttribute('cy',R); circle.setAttribute('r',R-0.3);
  circle.setAttribute('fill','none'); circle.setAttribute('stroke','#e7ebf0'); circle.setAttribute('stroke-width',0.35);
  svg.appendChild(circle);
  for(let k=0;k<16;k++){
    const a=-Math.PI/2+k*(2*Math.PI/16);
    const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
    dot.setAttribute('cx',R+R*Math.cos(a)); dot.setAttribute('cy',R+R*Math.sin(a)); dot.setAttribute('r',0.5);
    dot.setAttribute('fill','#aab4c2'); dot.setAttribute('stroke','#05070a'); dot.setAttribute('stroke-width',0.12);
    svg.appendChild(dot);
  }
  if(showRoiLabels){
    for(let b=0;b<4;b++){
      const y=(bandEdges[b]+bandEdges[b+1])/2;
      const text=document.createElementNS('http://www.w3.org/2000/svg','text');
      text.setAttribute('x',R); text.setAttribute('y',y+0.5);
      text.setAttribute('text-anchor','middle'); text.setAttribute('font-size',1.6);
      text.setAttribute('fill','#e7ebf0'); text.setAttribute('font-weight','700');
      text.textContent = String(b+1);
      svg.appendChild(text);
    }
  }
  container.innerHTML='';
  container.appendChild(svg);
  return rects;
}

const dynPolys = buildPixelGrid(document.getElementById('eitDynamicImg'), {showRoiLines:false, showRoiLabels:false});
const minPolys = buildPixelGrid(document.getElementById('eitMinuteImg'), {showRoiLines:true,  showRoiLabels:true});

// ================= EIT details: one recording at a time =================
// The plots and images show one "detail": a file of the Files block or the
// merged recording (js/prepare/eit-files.js chooses it with setEitDetail).
// A detail holds raw curves on a 20 Hz grid over its length (global
// impedance and ROI 1 to 4, ventral to dorsal), a minute image, and a
// function that gives the dynamic image at time t.
const DETAIL_HZ = 20;
const ROI_KEYS = ['roi1','roi2','roi3','roi4'];
const REAL_MINUTE = decodeB62(EIT.minute, EIT.nelem, 0);
const PIXEL_BAND = Array.from({length:EIT.nelem}, (_,i)=> rowBand(Math.floor(i/G)));
// size of each ROI curve of the real recording, relative to the global curve
const REAL_ROI_WEIGHTS = EIT.mtv.roi.map(v=> v/EIT.mtv.global);
// MTV in display units: the real global MTV shows as 4718, as on the device
const MTV_SCALE = 4718 / EIT.mtv.global;

// the real dynamic image of file 1 at file time t (3.0 to 13.0 s), else null
function realEitFrame(t){
  if(t < EIT.t0 || t > EIT.t0+EIT.dur) return null;
  const idx = Math.round((t-EIT.t0)/EIT.dur*(EIT.nframes-1));
  return decodeB62(EIT.frames, EIT.nelem, idx*EIT.nelem);
}
// the real minute image with a gain per ROI (1 = unchanged), peak scaled to 1
function minuteWithGains(gains){
  const out = new Float32Array(EIT.nelem); let mx = 0;
  for(let i=0;i<out.length;i++){ out[i] = REAL_MINUTE[i]*gains[PIXEL_BAND[i]]; mx = Math.max(mx, out[i]); }
  if(mx>0) for(let i=0;i<out.length;i++) out[i] /= mx;
  return out;
}

// detail of one file. globalAt(t): global impedance at file time t;
// weights: size of each ROI curve against the global curve; gains: minute
// image per ROI; realFrame(t): a real dynamic image, or null
function makeFileDetail({dur, globalAt, weights, gains, realFrame=null}){
  const n = Math.round(dur*DETAIL_HZ)+1;
  const raw = {global:new Float32Array(n)};
  ROI_KEYS.forEach(k=> raw[k]=new Float32Array(n));
  for(let i=0;i<n;i++){
    const t = i/DETAIL_HZ;
    raw.global[i] = globalAt(t);
    // each ROI follows the global curve with its own size, a delay of
    // 0.06 s per band (dorsal regions fill a little later) and a slow change of size
    ROI_KEYS.forEach((k,b)=>{
      raw[k][i] = globalAt(Math.max(0, t-0.06*b)) * weights[b] * (1 + 0.05*Math.sin(2*Math.PI*t/(9+3*b)));
    });
  }
  const d = {dur, raw, minute:minuteWithGains(gains), joins:[]};
  d.frameAt = t=> (realFrame && realFrame(t)) || derivedFrame(d, t);
  return finishDetail(d);
}
// dynamic image from the minute image and the level of each ROI curve at t
function derivedFrame(d, t){
  const i = Math.max(0, Math.min(d.q.global.length-1, Math.round(t*DETAIL_HZ)));
  const lv = ROI_KEYS.map(k=> d.q[k][i]);
  const out = new Float32Array(EIT.nelem);
  for(let p=0;p<out.length;p++) out[p] = d.minute[p]*(0.12 + 0.88*lv[PIXEL_BAND[p]]);
  return out;
}
// sections of file details joined in order: [{detail, start, end}]
function makeMergedDetail(parts){
  const pieces = []; let at = 0, total = 0;
  parts.forEach(p=>{
    const i0 = Math.round(p.start*DETAIL_HZ), i1 = Math.round(p.end*DETAIL_HZ);
    pieces.push({...p, i0, i1, from:at});
    at += (i1-i0)/DETAIL_HZ; total += i1-i0;
  });
  const raw = {};
  ['global', ...ROI_KEYS].forEach(k=>{
    const a = new Float32Array(total+1); let w = 0;
    pieces.forEach(pc=>{ a.set(pc.detail.raw[k].subarray(pc.i0, pc.i1), w); w += pc.i1-pc.i0; });
    const last = pieces[pieces.length-1];
    a[total] = last.detail.raw[k][Math.min(last.i1, last.detail.raw[k].length-1)];
    raw[k] = a;
  });
  // minute image: the mean of the parts, weighted by their length
  const minute = new Float32Array(EIT.nelem);
  pieces.forEach(pc=>{ const w = (pc.i1-pc.i0)/total; for(let p=0;p<minute.length;p++) minute[p] += pc.detail.minute[p]*w; });
  const d = {dur: total/DETAIL_HZ, raw, minute, joins: pieces.slice(1).map(pc=>pc.from)};
  d.frameAt = t=>{
    let pc = pieces[0];
    for(const x of pieces) if(t >= x.from) pc = x;
    return pc.detail.frameAt(pc.start + (t - pc.from));
  };
  return finishDetail(d);
}

// normalized curves, MTV per curve, ROI shares, breaths and rate
function finishDetail(d){
  const n = d.raw.global.length;
  d.q = {}; d.lo = {}; d.hi = {};
  ['global', ...ROI_KEYS].forEach(k=>{
    const a = d.raw[k]; let lo=Infinity, hi=-Infinity;
    for(const v of a){ if(v<lo) lo=v; if(v>hi) hi=v; }
    const q = new Float32Array(n), span = (hi-lo)||1;
    for(let i=0;i<n;i++) q[i] = (a[i]-lo)/span;
    d.q[k]=q; d.lo[k]=lo; d.hi[k]=hi;
  });
  const starts = breathStartsOf(t=> d.raw.global[Math.min(n-1, Math.round(t*DETAIL_HZ))], d.dur);
  d.breaths = Math.max(0, starts.length-1);
  d.rate = starts.length>1 ? 60*(starts.length-1)/(starts[starts.length-1]-starts[0]) : null;
  // MTV: the mean of (highest − lowest) value within each breath
  d.mtv = {};
  ['global', ...ROI_KEYS].forEach(k=>{
    const a = d.raw[k];
    if(starts.length<2){ d.mtv[k] = d.hi[k]-d.lo[k]; return; }
    let sum = 0;
    for(let b=0;b<starts.length-1;b++){
      let lo=Infinity, hi=-Infinity;
      for(let i=Math.round(starts[b]*DETAIL_HZ); i<=Math.round(starts[b+1]*DETAIL_HZ) && i<n; i++){ lo=Math.min(lo,a[i]); hi=Math.max(hi,a[i]); }
      sum += hi-lo;
    }
    d.mtv[k] = sum/(starts.length-1);
  });
  const roiSum = ROI_KEYS.reduce((s,k)=> s+d.mtv[k], 0) || 1;
  d.pct = ROI_KEYS.map(k=> Math.round(d.mtv[k]/roiSum*100));
  return d;
}

// ---- stacked plots ----
const PANELS=[
  {key:'global', name:'Global', h:96, cls:'is-global'},
  {key:'roi1',   name:'ROI 1',  h:64, cls:''},
  {key:'roi2',   name:'ROI 2',  h:64, cls:''},
  {key:'roi3',   name:'ROI 3',  h:64, cls:''},
  {key:'roi4',   name:'ROI 4',  h:64, cls:''},
];
const stack=document.getElementById('eitPlotStack');
const cursorEl=document.getElementById('eitCursor');

// panels are built once; their heads and polylines change with the detail
const eitPolys={}, eitHeads={};
PANELS.forEach(pn=>{
  const W=1000, H=pn.h;
  const div=document.createElement('div');
  div.className='eit-panel '+pn.cls;
  div.style.height=H+'px';
  div.innerHTML=`
    <div class="eit-panel-head">
      <span class="nm">${pn.name}</span>
      <span class="mtv"></span>
      <span class="mx"></span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <polyline fill="none" stroke="${pn.key==='global'?'var(--accent)':'var(--eit)'}" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
    </svg>`;
  stack.insertBefore(div, cursorEl);
  eitPolys[pn.key]=div.querySelector('polyline');
  eitHeads[pn.key]=div.querySelector('.eit-panel-head');
});

let DET = null;          // the detail on screen
// redraw every EIT panel for the visible window [t0,t1] (seconds of the detail)
function renderEitWindow(t0,t1){
  if(!DET) return;
  PANELS.forEach(pn=>{
    const q=DET.q[pn.key], n=q.length;
    const W=1000, H=pn.h, TOP=17, BOT=6;
    const i0=Math.max(0, Math.floor(t0*DETAIL_HZ)), i1=Math.min(n-1, Math.ceil(t1*DETAIL_HZ));
    let pts='';
    for(let i=i0;i<=i1;i++){
      const x=((i/DETAIL_HZ-t0)/(t1-t0))*W;
      const y=TOP+(1-q[i])*(H-TOP-BOT);
      pts+=`${x.toFixed(1)},${y.toFixed(1)} `;
    }
    eitPolys[pn.key].setAttribute('points', pts.trim());
  });
  // dashed lines where two merged sections meet
  stack.querySelectorAll('.eit-join').forEach(el=>el.remove());
  DET.joins.forEach(tj=>{
    if(tj<=t0 || tj>=t1) return;
    const el=document.createElement('div');
    el.className='eit-join';
    el.style.left=((tj-t0)/(t1-t0)*100)+'%';
    el.title=`Join of two sections at ${fmtSec(tj)}`;
    stack.appendChild(el);
  });
  placeEitCursor();
}

// show a detail: heads, minute image, rate, note, time range and cursor
function setEitDetail(d, note){
  const keepTime = DET && eitTime <= d.dur ? eitTime : d.dur*0.55;
  DET = d;
  if(typeof stopPlay==='function' && eitTimer) stopPlay();
  PANELS.forEach((pn,pi)=>{
    const shown = Math.round(d.mtv[pn.key]*MTV_SCALE);
    const pct = pn.key==='global' ? 100 : d.pct[pi-1];
    eitHeads[pn.key].querySelector('.mtv').textContent = `MTV: ${shown} = ${pct} %`;
    eitHeads[pn.key].querySelector('.mx').textContent = d.hi[pn.key].toFixed(3);
  });
  minPolys.forEach((p,i)=>{ if(p) p.setAttribute('fill', eitColor(d.minute[i])); });
  document.getElementById('eitRate').textContent = d.rate ? d.rate.toFixed(1)+' br/min' : '— br/min';
  document.getElementById('eitDetailNote').innerHTML = note || '';
  if(ZOOM.eit) ZOOM.eit.setFullDur(d.dur);
  setEitTime(keepTime);
}

// ---- cursor + transport ----
const EIT_IMAGE_STEP = 0.1;      // s between two dynamic images
let eitTime = 0;                 // cursor time, s of the detail
function placeEitCursor(){
  const v=ZOOM.eit ? ZOOM.eit.window() : {t0:0,t1:DET.dur};
  const frac=(eitTime-v.t0)/(v.t1-v.t0);
  cursorEl.style.display=(frac<0||frac>1)?'none':'block';
  cursorEl.style.left=(frac*100)+'%';
}
function setEitTime(t){
  eitTime=Math.max(0, Math.min(DET.dur, Math.round(t/EIT_IMAGE_STEP)*EIT_IMAGE_STEP));
  placeEitCursor();
  const d=DET.frameAt(eitTime);
  dynPolys.forEach((p,i)=>{ if(p) p.setAttribute('fill', eitColor(d[i])); });
  document.getElementById('eitCursorTime').textContent=fmtSec(eitTime, 2);
  document.getElementById('eitImageNo').textContent=Math.round(eitTime*EIT.fs)+1;
}

function timeAtClientX(cx){
  const r=stack.getBoundingClientRect();
  const v=ZOOM.eit ? ZOOM.eit.window() : {t0:0,t1:DET.dur};
  return v.t0+((cx-r.left)/r.width)*(v.t1-v.t0);
}

let eitTimer=null;
const playBtn=document.querySelector('.eit-tbtn.play');
function stopPlay(){ if(eitTimer){clearInterval(eitTimer); eitTimer=null;} playBtn.classList.remove('on'); playBtn.innerHTML=prepIcon('play'); }
playBtn.innerHTML=prepIcon('play');
document.querySelector('.eit-transport').addEventListener('click', e=>{
  const b=e.target.closest('.eit-tbtn'); if(!b) return;
  const t=b.dataset.t;
  if(t==='start'){ stopPlay(); setEitTime(0); }
  else if(t==='end'){ stopPlay(); setEitTime(DET.dur); }
  else if(t==='back'){ stopPlay(); setEitTime(eitTime-EIT_IMAGE_STEP); }
  else if(t==='fwd'){ stopPlay(); setEitTime(eitTime+EIT_IMAGE_STEP); }
  else if(t==='play'){
    if(eitTimer){ stopPlay(); }
    else {
      playBtn.classList.add('on'); playBtn.innerHTML=prepIcon('pause');
      // 12.5 images per s at 0.1 s per image: a little slower than real time
      eitTimer=setInterval(()=>{
        setEitTime(eitTime>=DET.dur-1e-9 ? 0 : eitTime+EIT_IMAGE_STEP);
      }, 1000/12.5);
    }
  }
});

// ---- "Display" menu in the EIT toolbar (the items are not wired in this mockup) ----
(function wireEitDisplayMenu(){
  const btn=document.querySelector('#cardEit .menu-btn');
  const menu=document.querySelector('#cardEit .menu-pop');
  function setOpen(open){ menu.hidden=!open; btn.setAttribute('aria-expanded', String(open)); }
  btn.addEventListener('click', ()=> setOpen(menu.hidden));
  menu.addEventListener('click', e=>{
    const item=e.target.closest('button'); if(!item) return;
    setOpen(false);
    showEdgeNote(btn, `"${item.textContent}" opens its settings in the real program.`);
  });
  document.addEventListener('mousedown', e=>{ if(!menu.hidden && !e.target.closest('#cardEit .menu-wrap')) setOpen(false); });
  document.addEventListener('keydown', e=>{ if(e.key==='Escape' && !menu.hidden){ setOpen(false); btn.focus(); } });
})();
