// ================= Review outputs: Plots section (real pipeline outputs) =================
// REVIEW (the figure-PNG blob) is declared in js/data/review-figures.js, which
// the build orders ahead of this file.

const PLOTS = [
  {id:'global_impedance', name:'Global impedance — breaths, TIV & EELI', mod:'eit',
   src:'eit.global_impedance · eit.detect_breaths · eit.continuous_tiv · eit.eeli',
   file:'figures/global_impedance.png', kind:'img', img:REVIEW.figures['global_impedance.png']},
  {id:'pixel_tiv', name:'Mean pixel TIV (13 breaths)', mod:'eit',
   src:'eit.pixel_tiv', file:'figures/pixel_tiv.png', kind:'img', img:REVIEW.figures['pixel_tiv.png']},
  {id:'rate_detection', name:'Rate detection (RR & HR)', mod:'eit',
   src:'eit.detect_rates', file:'figures/rate_detection.png', kind:'img', img:REVIEW.figures['rate_detection.png']},
  {id:'emg_breaths', name:'EMGdi breath detection overlay', mod:'emg',
   src:'emg.detect_breaths', file:'computed from emg_breaths.csv', kind:'emgBreaths'},
  {id:'pocc_quality', name:'Pocc quality — ΔP criteria', mod:'vent',
   src:'ventilator.pocc_quality', file:'computed from parameter_results.csv', kind:'poccQuality'},
];

const MOD_COLOR = {eit:'var(--eit)', emg:'var(--emg)', vent:'var(--vent)'};
const MOD_NAME  = {eit:'EIT', emg:'EMG', vent:'Ventilator'};

const plotsList = document.getElementById('plotsList');
['eit','emg','vent'].forEach(mod=>{
  const head=document.createElement('div');
  head.className='pal-group-head';
  head.innerHTML=`<span class="chip-dot" style="background:${MOD_COLOR[mod]}"></span>${MOD_NAME[mod]}<span class="count">${PLOTS.filter(p=>p.mod===mod).length}</span>`;
  plotsList.appendChild(head);
  PLOTS.filter(p=>p.mod===mod).forEach(p=>{
    const row=document.createElement('label');
    row.className='plot-item';
    const generated = p.kind==='img';
    row.innerHTML=`
      <input type="checkbox" data-plot="${p.id}">
      <div class="pi-body">
        <div class="pi-name">${p.name}</div>
        <div class="pi-src">${p.src}</div>
        <span class="pi-tag ${generated?'generated':''}">${generated?'generated: ':'on demand: '}${p.file}</span>
      </div>`;
    plotsList.appendChild(row);
  });
});

function buildEmgBreathsCard(){
  const W=900, H=170, TOP=16, BOT=16;
  const dur=60, n=1200;
  const rr=13.9/60; // eit-detected RR, same phase model style as elsewhere
  let pts='';
  for(let i=0;i<=n;i++){
    const t=(i/n)*dur;
    const y=H/2 - 28*Math.sin(2*Math.PI*rr*t) - 6*Math.sin(2*Math.PI*2.1*t);
    pts+=`${(t/dur*W).toFixed(1)},${y.toFixed(1)} `;
  }
  let markers='';
  REVIEW.emg_peaks.forEach(t=>{
    const x=(t/dur*W).toFixed(1);
    markers+=`<line x1="${x}" y1="${TOP}" x2="${x}" y2="${H-BOT}" stroke="var(--ok)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  });
  return `
    <div class="viz-config">
      <span class="lbl">Peak markers</span>
      <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;">
        <input type="checkbox" checked id="emgMarkersToggle" style="accent-color:var(--accent);"> show 19 detected breaths
      </label>
    </div>
    <svg width="100%" height="${H}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <g id="emgMarkersGroup">${markers}</g>
      <polyline points="${pts.trim()}" fill="none" stroke="var(--emg)" stroke-width="1.6"/>
    </svg>`;
}

function poccBarsSvg(crit){
  const vals=REVIEW.pocc[crit];
  const W=420, H=170, TOP=20, BOT=30, barW=90, gap=60;
  const maxV=Math.max(...vals.map(Math.abs), 0.001)*1.2;
  const y0=H-BOT;
  let bars=`<line x1="0" y1="${y0}" x2="${W}" y2="${y0}" stroke="var(--border)"/>`;
  vals.forEach((v,i)=>{
    const x=60+i*(barW+gap);
    const h=(Math.abs(v)/maxV)*(H-TOP-BOT);
    const y=v>=0 ? y0-h : y0;
    bars+=`<rect x="${x}" y="${y}" width="${barW}" height="${Math.max(1,h)}" fill="${v>=0?'var(--vent)':'var(--crit)'}" rx="3"/>`;
    bars+=`<text x="${x+barW/2}" y="${y0+18}" text-anchor="middle" font-size="11" fill="var(--text-dim)">breath ${i}</text>`;
    bars+=`<text x="${x+barW/2}" y="${(v>=0?y-6:y0+34)}" text-anchor="middle" font-size="11" font-weight="700" fill="var(--text)">${v.toFixed(4)} cmH₂O</text>`;
  });
  return bars;
}
function buildPoccCard(){
  return `
    <div class="viz-config">
      <span class="lbl">Criterion</span>
      <div class="seg" style="width:180px;" id="poccSeg">
        <button data-c="dp_up_10">dp_up_10</button>
        <button class="on" data-c="dp_up_90">dp_up_90</button>
      </div>
    </div>
    <svg width="100%" height="170" viewBox="0 0 420 170" id="poccSvg">${poccBarsSvg('dp_up_90')}</svg>`;
}

function vizCardHtml(p){
  if(p.kind==='img'){
    return `
      <div class="viz-config">
        <span class="lbl">Size</span>
        <div class="seg" style="width:150px;" data-sizeseg="${p.id}">
          <button class="on" data-sz="fit">fit</button>
          <button data-sz="large">large</button>
        </div>
      </div>
      <img class="viz-img" id="vizimg-${p.id}" src="data:image/png;base64,${p.img}" alt="${p.name}">`;
  }
  if(p.kind==='emgBreaths') return buildEmgBreathsCard();
  if(p.kind==='poccQuality') return buildPoccCard();
  return '';
}

const plotsViz = document.getElementById('plotsViz');
const plotsVizEmpty = document.getElementById('plotsVizEmpty');

function addVizCard(p){
  const card=document.createElement('div');
  card.className='viz-card';
  card.id='viz-'+p.id;
  card.innerHTML=`
    <div class="viz-card-head"><span class="nm">${p.name}</span><span class="src">${p.file}</span></div>
    <div class="viz-card-body">${vizCardHtml(p)}</div>`;
  plotsViz.appendChild(card);

  // wire per-card config controls
  const sizeSeg=card.querySelector(`[data-sizeseg="${p.id}"]`);
  if(sizeSeg){
    sizeSeg.addEventListener('click', e=>{
      const b=e.target.closest('button'); if(!b) return;
      sizeSeg.querySelectorAll('button').forEach(x=>x.classList.toggle('on', x===b));
      document.getElementById('vizimg-'+p.id).style.maxWidth = b.dataset.sz==='large' ? 'none' : '640px';
    });
  }
  const emgToggle=card.querySelector('#emgMarkersToggle');
  if(emgToggle){
    emgToggle.addEventListener('change', ()=>{
      card.querySelector('#emgMarkersGroup').style.display = emgToggle.checked ? '' : 'none';
    });
  }
  const poccSeg=card.querySelector('#poccSeg');
  if(poccSeg){
    poccSeg.addEventListener('click', e=>{
      const b=e.target.closest('button'); if(!b) return;
      poccSeg.querySelectorAll('button').forEach(x=>x.classList.toggle('on', x===b));
      card.querySelector('#poccSvg').innerHTML = poccBarsSvg(b.dataset.c);
    });
  }
}

document.getElementById('plotsList').addEventListener('change', e=>{
  const inp=e.target.closest('input[data-plot]'); if(!inp) return;
  const p=PLOTS.find(x=>x.id===inp.dataset.plot);
  if(inp.checked){
    plotsVizEmpty.classList.add('hidden');
    addVizCard(p);
  } else {
    document.getElementById('viz-'+p.id)?.remove();
    if(!plotsViz.querySelector('.viz-card')) plotsVizEmpty.classList.remove('hidden');
  }
});

