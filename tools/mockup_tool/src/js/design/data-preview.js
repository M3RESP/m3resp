// ================= "Available data" preview pop-up =================
// Clicking any row visualizes it. The mockup has no backend to fetch real
// values from, so each preview picks the most honest representation for that
// artifact TYPE: the Prepare tab's own waveform generators where the type has
// a natural shape, a clearly-labelled illustrative preview for
// detection/result/mask types that don't, and a plain explanation for plumbing
// types that were never meant to be plotted.
let dataVizOverlay = null;
function ensureDataVizOverlay(){
  if(dataVizOverlay) return dataVizOverlay;
  const overlay = document.createElement('div');
  overlay.className = 'data-viz-overlay';
  overlay.innerHTML = `
    <div class="data-viz-card" role="dialog">
      <div class="data-viz-head">
        <span class="dvtitle" id="dvTitle"></span>
        <span class="dvsub" id="dvSub"></span>
        <button class="data-viz-close" id="dvClose">×</button>
      </div>
      <div class="data-viz-body" id="dvBody"></div>
    </div>`;
  document.body.appendChild(overlay);
  overlay.addEventListener('click', e=>{ if(e.target===overlay) closeDataViz(); });
  overlay.querySelector('#dvClose').addEventListener('click', closeDataViz);
  document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeDataViz(); });
  dataVizOverlay = overlay;
  return overlay;
}
function closeDataViz(){
  if(dataVizOverlay) dataVizOverlay.classList.remove('on');
}

function seededRand(seedStr){
  let s=0; for(let i=0;i<seedStr.length;i++) s=(s*31+seedStr.charCodeAt(i))>>>0;
  return ()=>{ s=(s*1103515245+12345)>>>0; return (s>>>8)/0x1000000; };
}

// pick one of the Prepare tab's generators based on the producing op's
// modality, so the preview is a genuine representative shape rather than one
// fabricated for this popup
function signalGeneratorFor(op, seed){
  const prefix = (op||'').split('.')[0];
  if(prefix==='eit') return t=>eitOverviewAt(t % REC_SECONDS);
  if(prefix==='emg') return (t,i)=>emgRawAt(t, i);
  if(prefix==='ventilator') return t=>VENT_CHANNELS[0].fn(ventPhase(t));
  // generic/session/export/sync producers: no natural per-op waveform, fall
  // back to a gentle seeded oscillation so the shape still means "a signal",
  // just not one tied to a specific modality
  const rnd = seededRand(seed);
  const phase = rnd()*Math.PI*2, wob = 0.85+rnd()*0.3;
  return t => Math.sin(t*0.6*wob+phase);
}

function svgLineFromSamples(vals, {w=520,h=140,stroke='var(--accent)'}={}){
  let lo=Math.min(...vals), hi=Math.max(...vals);
  const span=(hi-lo)||1;
  let pts='';
  vals.forEach((v,i)=>{
    const x=(i/(vals.length-1))*w;
    const y=8+(1-(v-lo)/span)*(h-16);
    pts+=`${x.toFixed(1)},${y.toFixed(1)} `;
  });
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="width:100%;height:${h}px;display:block;">
    <line x1="0" y1="${h/2}" x2="${w}" y2="${h/2}" stroke="var(--border)" stroke-width="1" stroke-dasharray="3 3"/>
    <polyline points="${pts.trim()}" fill="none" stroke="${stroke}" stroke-width="2"/>
  </svg>`;
}

function renderKeyPreview(key, type, nodeId, op){
  const color = tc(type);
  // signal-like family: draw a real representative waveform
  if(color==='var(--type-signal)'){
    const gen = signalGeneratorFor(op, key);
    const N=300, dur=Math.min(REC_SECONDS,60);
    const vals=[]; for(let i=0;i<N;i++) vals.push(gen((i/N)*dur, i));
    return `<div class="data-viz-plot">${svgLineFromSamples(vals,{stroke:color})}</div>
      <p class="data-viz-note">Illustrative preview generated from ${op}'s signal shape over the working window — this mockup has no backend to fetch <code>${key}</code>'s real values from.</p>`;
  }
  // index/event-like: a timeline of representative tick marks
  if(color==='var(--type-index)'){
    const rnd = seededRand(key);
    const n = 6+Math.floor(rnd()*8);
    const ticks=[]; let t=rnd()*3;
    for(let i=0;i<n;i++){ t += 2+rnd()*4; ticks.push(Math.min(58,t)); }
    const w=520;
    const marks = ticks.map(tt=>`<line x1="${(tt/60*w).toFixed(1)}" y1="10" x2="${(tt/60*w).toFixed(1)}" y2="50" stroke="${color}" stroke-width="2.5"/>`).join('');
    return `<div class="data-viz-plot"><svg viewBox="0 0 ${w} 60" style="width:100%;height:60px;display:block;">
        <line x1="0" y1="30" x2="${w}" y2="30" stroke="var(--border)" stroke-width="1"/>${marks}
      </svg></div>
      <p class="data-viz-note">Illustrative event markers — ${n} representative occurrences along the working window. <code>${key}</code> holds the real indices once this step actually runs.</p>`;
  }
  // mask-like: a small binary-ish grid, seeded so it always looks the same for this key
  if(color==='var(--type-mask)'){
    const rnd = seededRand(key);
    const cx=8, cy=8, r=6.2;
    let cells='';
    for(let y=0;y<16;y++) for(let x=0;x<16;x++){
      const inCircle = Math.hypot(x-cx,y-cy) <= r;
      const on = inCircle && rnd() > 0.35;
      cells += `<div style="background:${on?color:'#0b1430'};"></div>`;
    }
    return `<div class="data-viz-grid">${cells}</div>
      <p class="data-viz-note">Illustrative ROI mask over a 16×16 grid — a stand-in for <code>${key}</code>'s real pixel selection.</p>`;
  }
  // result/scalar: a single representative number
  if(color==='var(--type-result)'){
    const rnd = seededRand(key);
    const val = (rnd()*40+2).toFixed(2);
    return `<div class="data-viz-bignum">${val}<span class="unit">a.u.</span></div>
      <p class="data-viz-note">Illustrative value — <code>${key}</code> is a computed result; its real number appears here once <b>${op}</b> has actually run.</p>`;
  }
  // generic/plumbing: honestly say there's nothing to plot
  return `<div class="data-viz-info">
      <div style="font-size:26px;margin-bottom:8px;">⚙️</div>
      <code>${key}</code> is plumbing/bookkeeping data (type <code>${type}</code>), not a scientific signal — there's nothing meaningful to chart here.
    </div>`;
}

function renderSequencePreview(seq){
  const lanes = [
    {label:'EIT · global impedance', color:'var(--eit)', fn:eitOverviewAt},
    {label:'EMGdi', color:'var(--emg)', fn:emgRawAt},
    {label:'Ventilator · P_aw', color:'var(--vent)', fn:t=>VENT_CHANNELS[0].fn(ventPhase(t))},
  ];
  const sorted=[...seq.windows].sort((a,b)=>a.start-b.start);
  let html='';
  lanes.forEach(l=>{
    const vals=[];
    sorted.forEach(w=>{
      const n=Math.max(8, Math.round((w.end-w.start)*4));
      for(let i=0;i<n;i++) vals.push(l.fn(w.start+(i/n)*(w.end-w.start), i));
    });
    html += `<div style="margin-bottom:10px;">
      <div style="font-size:10.5px;color:var(--text-faint);margin-bottom:4px;">${l.label}</div>
      <div class="data-viz-plot">${svgLineFromSamples(vals,{h:80,stroke:l.color})}</div>
    </div>`;
  });
  html += `<p class="data-viz-note">Real per-window samples from each modality's own signal generator, concatenated in time order exactly as <b>Save as new sequence</b> would — ${sorted.length} window${sorted.length===1?'':'s'}, ${fmtDuration(seq.total)} total.</p>`;
  return html;
}

function openDataViz(opts){
  const overlay = ensureDataVizOverlay();
  const title = overlay.querySelector('#dvTitle');
  const sub = overlay.querySelector('#dvSub');
  const body = overlay.querySelector('#dvBody');
  if(opts.kind==='sequence'){
    title.textContent = opts.seq.name;
    sub.textContent = 'saved sequence';
    body.innerHTML = renderSequencePreview(opts.seq);
  } else {
    title.textContent = opts.key;
    sub.innerHTML = `<span class="type-chip">${opts.type}</span>`;
    body.innerHTML = `<div class="data-viz-meta">from <b>${opts.node}</b> · <code>${opts.op}</code></div>` + renderKeyPreview(opts.key, opts.type, opts.node, opts.op);
  }
  overlay.classList.add('on');
}

// ---- clicking a saved sequence in "Available data" drops it on the canvas
// as a source node, the way a real "load this prepared sequence" step would ----
function loadSequenceAsNode(seqId){
  const sequences = (typeof SAVED_SEQUENCES!=='undefined') ? SAVED_SEQUENCES : [];
  const seq = sequences.find(s=>s.id===seqId);
  if(!seq) return;
  const existing = nodeById(seq.name);
  if(existing){
    selectNode(existing.id);
    const el = inner.querySelector(`.node[data-id="${existing.id}"]`);
    if(el) el.scrollIntoView({behavior:'smooth', block:'center', inline:'center'});
    return;
  }
  const baseY = Math.max(...NODES.map(n=>n.y+nodeHeight(n)), 480) + 40;
  const col = addSlotIndex % 5; addSlotIndex++;
  const n = {
    id: seq.name, op:'prepare.sequence', mod:'generic',
    x: 30 + col*(NODE_W+30), y: baseY,
    // port is literally named 'sequence' (not the sequence's own name) so it
    // matches load_eit/load_emg/load_ventilator's 'sequence' input exactly —
    // the node id already carries the sequence's unique identity.
    ins: [], outs: [['sequence', 'sequence_bundle']],
    status: 'ok',
  };
  NODES.push(n);

  // A saved sequence is a synchronized multi-modal crop: wire it into every
  // load step that can extract its own channel from it, unless that step
  // already sources from a different sequence (never silently swap it out).
  const wiredTo = [];
  ['load_eit','load_emg','load_ventilator'].forEach(loadId=>{
    const target = nodeById(loadId);
    if(!target || !target.ins.some(([nm])=>nm==='sequence')) return;
    if(EDGES.some(e=>e.dst===loadId && e.key==='sequence')) return;
    EDGES.push({src:n.id, dst:loadId, key:'sequence'});
    wiredTo.push(loadId);
  });

  renderAllNodes();
  drawEdges();
  selectNode(n.id);
  const el = inner.querySelector(`.node[data-id="${n.id}"]`);
  if(el){
    el.scrollIntoView({behavior:'smooth', block:'center', inline:'center'});
    const base = `Loaded "${seq.name}" — ${seq.windows.length} window${seq.windows.length===1?'':'s'}, ${fmtDuration(seq.total)}, saved in 1 · Prepare.`;
    showEdgeNote(el, wiredTo.length
      ? `${base} Wired into ${wiredTo.join(', ')} — each extracts its own channel from the sequence.`
      : `${base} Its load steps already source from another sequence — drag from its port to switch one manually.`);
  }
}

