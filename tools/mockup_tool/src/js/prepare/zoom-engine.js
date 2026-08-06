// ================= shared zoom / range-select engine =================

function fmtAxisTime(sec, span){
  if(span >= 120){
    const m=Math.floor(sec/60), s=Math.round(sec%60);
    return `${m}:${String(s).padStart(2,'0')}`;
  }
  return span < 2 ? sec.toFixed(2)+'s' : (span < 20 ? sec.toFixed(1)+'s' : Math.round(sec)+'s');
}

function makeZoomView(name, fullDur, render, {minSpan=0.2, onCursorClick=null}={}){
  const stackEl   = document.querySelector(`[data-zoom="${name}"]`);
  const axisEl    = document.querySelector(`[data-zoom-axis="${name}"]`);
  const toolbarEl = document.querySelector(`[data-zoom-toolbar="${name}"]`);
  const readoutEl = document.querySelector(`[data-zoom-readout="${name}"]`);
  let t0=0, t1=fullDur;

  const band=document.createElement('div');
  band.className='zoom-band';
  stackEl.appendChild(band);

  function drawAxis(){
    const span=t1-t0;
    let out='';
    for(let k=0;k<=6;k++) out+=`<span>${fmtAxisTime(t0+span*(k/6), span)}</span>`;
    axisEl.innerHTML=out;
  }
  function refresh(){
    render(t0,t1);
    drawAxis();
    const span=t1-t0;
    if(readoutEl) readoutEl.textContent=`${fmtAxisTime(t0,span)} – ${fmtAxisTime(t1,span)}  (${span<2?span.toFixed(2):span.toFixed(1)} s)`;
    if(toolbarEl){
      const outBtn=toolbarEl.querySelector('[data-z="out"]');
      const inBtn =toolbarEl.querySelector('[data-z="in"]');
      if(outBtn) outBtn.disabled = (span>=fullDur-1e-6);
      if(inBtn)  inBtn.disabled  = (span<=minSpan+1e-6);
    }
  }
  function setWindow(a,b){
    let span=Math.max(minSpan, Math.min(fullDur, b-a));
    a=Math.max(0, Math.min(fullDur-span, a));
    t0=a; t1=a+span; refresh();
  }
  function zoomBy(factor, anchorFrac=0.5){
    const span=t1-t0, anchorT=t0+span*anchorFrac;
    const ns=Math.max(minSpan, Math.min(fullDur, span*factor));
    setWindow(anchorT-ns*anchorFrac, anchorT-ns*anchorFrac+ns);
  }

  if(toolbarEl){
    toolbarEl.addEventListener('click', e=>{
      const b=e.target.closest('[data-z]'); if(!b) return;
      if(b.dataset.z==='in') zoomBy(0.5);
      else if(b.dataset.z==='out') zoomBy(2);
      else if(b.dataset.z==='reset') setWindow(0, fullDur);
    });
  }

  stackEl.addEventListener('wheel', e=>{
    e.preventDefault();
    const r=stackEl.getBoundingClientRect();
    zoomBy(e.deltaY>0?1.25:0.8, (e.clientX-r.left)/r.width);
  }, {passive:false});

  // drag = rubber-band select a range; a click without drag falls through to onCursorClick
  let dragging=false, ax=0, bx=0;
  stackEl.addEventListener('mousedown', e=>{
    if(e.button!==0) return;
    dragging=true; ax=bx=e.clientX; band.classList.remove('on'); e.preventDefault();
  });
  document.addEventListener('mousemove', e=>{
    if(!dragging) return;
    bx=e.clientX;
    const r=stackEl.getBoundingClientRect();
    const l=Math.max(r.left,Math.min(ax,bx)), rr=Math.min(r.right,Math.max(ax,bx));
    if(Math.abs(bx-ax)>4){
      band.classList.add('on');
      band.style.left=(l-r.left)+'px';
      band.style.width=(rr-l)+'px';
    }
  });
  document.addEventListener('mouseup', e=>{
    if(!dragging) return;
    dragging=false;
    const moved=Math.abs(bx-ax);
    band.classList.remove('on');
    const r=stackEl.getBoundingClientRect();
    if(moved>4){
      const span=t1-t0;
      const fa=(Math.min(ax,bx)-r.left)/r.width, fb=(Math.max(ax,bx)-r.left)/r.width;
      setWindow(t0+span*Math.max(0,fa), t0+span*Math.min(1,fb));
    } else if(onCursorClick){
      onCursorClick(ax);
    }
  });

  const api={window:()=>({t0,t1}), setWindow, zoomBy, refresh};
  ZOOM[name]=api;
  refresh();
  return api;
}

makeZoomView('eit',  EIT.dur,      renderEitWindow,  {minSpan:0.4, onCursorClick:cx=>setEitFrame(frameAtClientX(cx))});
makeZoomView('vent', VENT_DUR,     renderVentWindow, {minSpan:0.5});
makeZoomView('emg',  EMG_FULL_DUR, renderEmgWindow,  {minSpan:1.0});

// crop-strip lanes last: they read VENT_CHANNELS, declared above
renderSliceLanes();


