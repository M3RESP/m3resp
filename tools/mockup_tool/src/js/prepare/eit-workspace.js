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

// minute image is static
{
  const m=decodeB62(EIT.minute, EIT.nelem, 0);
  minPolys.forEach((p,i)=>{ if(p) p.setAttribute('fill', eitColor(m[i])); });
}

// ---- stacked plots ----
const PANELS=[
  {key:'global', name:'Global',  pct:100,               h:96,  cls:'is-global'},
  {key:'roi1',   name:'ROI 1',   pct:EIT.mtv.pct[0],    h:64,  cls:''},
  {key:'roi2',   name:'ROI 2',   pct:EIT.mtv.pct[1],    h:64,  cls:''},
  {key:'roi3',   name:'ROI 3',   pct:EIT.mtv.pct[2],    h:64,  cls:''},
  {key:'roi4',   name:'ROI 4',   pct:EIT.mtv.pct[3],    h:64,  cls:''},
];
// MTV in display units: scale the raw impedance units to a readable integer scale
const MTV_SCALE = 4718 / EIT.mtv.global;   // keeps Global at a device-like magnitude
const stack=document.getElementById('eitPlotStack');
const cursorEl=document.getElementById('eitCursor');

// panels are built once; their polylines are redrawn whenever the view window changes
const eitCurveCache={}, eitPolys={};
PANELS.forEach((pn,pi)=>{
  eitCurveCache[pn.key]=decodeB62(EIT.curves[pn.key].q, EIT.npts, 0);
  const W=1000, H=pn.h;
  const mtvVal = pn.key==='global' ? EIT.mtv.global : EIT.mtv.roi[pi-1];
  const shown  = Math.round(mtvVal*MTV_SCALE);
  const div=document.createElement('div');
  div.className='eit-panel '+pn.cls;
  div.style.height=H+'px';
  div.innerHTML=`
    <div class="eit-panel-head">
      <span class="nm">${pn.name}</span>
      <span class="mtv">MTV: ${shown} = ${pn.pct} %</span>
      <span class="mx">${EIT.curves[pn.key].hi.toFixed(3)}</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <polyline fill="none" stroke="${pn.key==='global'?'var(--accent)':'var(--eit)'}" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
    </svg>`;
  stack.insertBefore(div, cursorEl);
  eitPolys[pn.key]=div.querySelector('polyline');
});

// redraw every EIT panel for the visible window [t0,t1] (seconds within the excerpt)
function renderEitWindow(t0,t1){
  PANELS.forEach(pn=>{
    const q=eitCurveCache[pn.key];
    const W=1000, H=pn.h, TOP=17, BOT=6;
    const i0=Math.max(0, Math.floor(t0/EIT.dur*(EIT.npts-1)));
    const i1=Math.min(EIT.npts-1, Math.ceil(t1/EIT.dur*(EIT.npts-1)));
    let pts='';
    for(let i=i0;i<=i1;i++){
      const t=(i/(EIT.npts-1))*EIT.dur;
      const x=((t-t0)/(t1-t0))*W;
      const y=TOP+(1-q[i])*(H-TOP-BOT);
      pts+=`${x.toFixed(1)},${y.toFixed(1)} `;
    }
    eitPolys[pn.key].setAttribute('points', pts.trim());
  });
  placeEitCursor();
}

document.getElementById('eitRate').textContent = EIT.rate+' br/min';
document.getElementById('eitBreaths').textContent = EIT.breaths;

// ---- cursor + transport ----
let eitFrame = 0;                                  // 0 .. nframes-1
function placeEitCursor(){
  const t=(eitFrame/(EIT.nframes-1))*EIT.dur;
  const v=ZOOM.eit ? ZOOM.eit.window() : {t0:0,t1:EIT.dur};
  const frac=(t-v.t0)/(v.t1-v.t0);
  cursorEl.style.display=(frac<0||frac>1)?'none':'block';
  cursorEl.style.left=(frac*100)+'%';
}
function setEitFrame(f){
  eitFrame=Math.max(0,Math.min(EIT.nframes-1,f));
  placeEitCursor();
  const d=decodeB62(EIT.frames, EIT.nelem, eitFrame*EIT.nelem);
  dynPolys.forEach((p,i)=>{ if(p) p.setAttribute('fill', eitColor(d[i])); });
  const tsec=(eitFrame/(EIT.nframes-1))*EIT.dur;
  const mm=String(Math.floor(tsec/60)).padStart(2,'0');
  const ss=String(Math.floor(tsec%60)).padStart(2,'0');
  const ms=String(Math.round((tsec%1)*1000)).padStart(3,'0');
  document.getElementById('eitCursorTime').textContent=`${mm}:${ss}.${ms}`;
  document.getElementById('eitImageNo').textContent=Math.round(EIT.t0*EIT.fs)+eitFrame*4+1;
}

function frameAtClientX(cx){
  const r=stack.getBoundingClientRect();
  const v=ZOOM.eit ? ZOOM.eit.window() : {t0:0,t1:EIT.dur};
  const t=v.t0+((cx-r.left)/r.width)*(v.t1-v.t0);
  return Math.round((t/EIT.dur)*(EIT.nframes-1));
}

let eitTimer=null;
const playBtn=document.querySelector('.eit-tbtn.play');
function stopPlay(){ if(eitTimer){clearInterval(eitTimer); eitTimer=null;} playBtn.classList.remove('on'); playBtn.textContent='▶'; }
document.querySelector('.eit-transport').addEventListener('click', e=>{
  const b=e.target.closest('.eit-tbtn'); if(!b) return;
  const t=b.dataset.t;
  if(t==='start'){ stopPlay(); setEitFrame(0); }
  else if(t==='end'){ stopPlay(); setEitFrame(EIT.nframes-1); }
  else if(t==='back'){ stopPlay(); setEitFrame(eitFrame-1); }
  else if(t==='fwd'){ stopPlay(); setEitFrame(eitFrame+1); }
  else if(t==='play'){
    if(eitTimer){ stopPlay(); }
    else {
      playBtn.classList.add('on'); playBtn.textContent='❚❚';
      eitTimer=setInterval(()=>{
        setEitFrame(eitFrame>=EIT.nframes-1 ? 0 : eitFrame+1);
      }, 1000/12.5);
    }
  }
});

setEitFrame(Math.round(EIT.nframes*0.55));


