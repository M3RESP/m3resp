// ================= several files of one modality: sections and merge =================
// Used by the EIT, EMG and ventilator cards (eit-files.js, emg-files.js,
// vent-files.js).
// A modality card can hold more than one file. A drag on the lane of a file
// selects a section of it; a drag on a section slides it, a drag on its edge
// changes its length. The sections, in the order of the "Merge order" list,
// join into one recording ("Merge sections"). The plots under the Files block
// show one file or the merged recording ("Show" in the toolbar, or a click on
// a file name or on the merged preview).
// A click on the "Merge order" head opens or closes the merge block;
// "Merge sections" closes it, and a change of the sections after a merge
// opens it again.
//
// makeFileMerger(cfg) reads these settings:
//   prefix      start of the element ids: <prefix>FileList, <prefix>MergeBtn, …
//   name        modality name in the text ("EIT", "EMG")
//   color       colour of the lanes
//   files       [{id, name, dur (s), fs (Hz), …}]; the list changes in place
//   sections    [{id, fileId, start (s), end (s)}]
//   laneFn(f)   the value drawn on the lane of file f, at file time t; or a
//               list of {fn, color, label}, one line each, below each other
//   newFile(name, n)   a file object for an added file (example signal)
//   show(id, M)        draw the plots for a file id or 'merged'
//   note(id, M)        (optional) text under the plots

const SECTION_MIN = 0.5;                       // s
const snap01 = t=> Math.round(t*10)/10;

// height of the name strip above each signal, in the units of the drawing
const LABEL_ROOM = 15;

// one lane: signals over `dur` seconds from file time t0. Several signals sit
// below each other, each scaled to its own row; a label names each row and
// sits in the space above its signal. The merged preview has no labels.
function fileLaneSvg(traces, dur, t0=0, {labels:showLabels=true}={}){
  const n=Math.max(1, traces.length), named = showLabels && traces.length>1;
  // each row has 60 units for its signal (the same as a one-signal lane),
  // plus LABEL_ROOM units above it for the signal name
  const N=700, W=1000, PAD=5, rowH = 60 + (named ? LABEL_ROOM : 0), H=rowH*n;
  const TOP = named ? LABEL_ROOM + PAD : PAD;
  const lines = traces.map(({fn, color}, k)=>{
    const vals=[];
    for(let i=0;i<N;i++) vals.push(fn(t0 + (i/(N-1))*dur));
    const lo=Math.min(...vals), hi=Math.max(...vals), span=(hi-lo)||1, y0=k*rowH;
    let pts='';
    for(let i=0;i<N;i++) pts+=`${((i/(N-1))*W).toFixed(1)},${(y0+TOP+(1-(vals[i]-lo)/span)*(rowH-TOP-PAD)).toFixed(1)} `;
    const sep = k ? `<line x1="0" y1="${y0}" x2="${W}" y2="${y0}" stroke="var(--border)" stroke-width="1" vector-effect="non-scaling-stroke"/>` : '';
    return sep + `<polyline points="${pts.trim()}" fill="none" stroke="${color}" stroke-width="1.4" vector-effect="non-scaling-stroke"/>`;
  }).join('');
  const labels = showLabels && traces.length>1 ? traces.map((tr, k)=>
    `<span class="ef-lane-label" style="top:${(k/traces.length*100).toFixed(2)}%">${tr.label || ''}</span>`).join('') : '';
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${lines}</svg>${labels}`;
}
// lane height in px for n signals below each other: each signal is drawn as
// tall as the signal in a one-signal lane (56 px), and its name gets a strip
// of LABEL_ROOM units above it
const fileLaneHeight = n=> n<=1 ? 56 : n*56*(60+LABEL_ROOM)/60;
// height of the merged preview in px: 44 px for each signal
const previewHeight = n=> 44*Math.max(1, n);

function makeFileMerger(cfg){
  const $ = id=> document.getElementById(cfg.prefix + id);
  const M = {
    files: cfg.files, sections: cfg.sections,
    shown: cfg.files[0].id,        // file id or 'merged'
    merged: false,                 // true after "Merge sections", until the next change
    collapsed: false,              // true when the merge block is closed
    active: cfg.sections[0] ? cfg.sections[0].id : null,
    sectionSeq: cfg.sections.length, fileSeq: cfg.files.length,
  };
  const list = $('FileList'), secList = $('SectionList'), preview = $('MergePreview');
  const summary = $('MergeSummary'), warn = $('MergeWarn'), mergeBtn = $('MergeBtn');
  const sourceSelect = $('SourceSelect');
  const mergeBox = mergeBtn.closest('.ef-merge'), mergeHead = mergeBox.querySelector('.ef-merge-head');

  M.file = id=> M.files.find(f=>f.id===id);
  const tracesOf = f=>{ const r = cfg.laneFn(f); return Array.isArray(r) ? r : [{fn:r, color:cfg.color}]; };
  M.fileNo = id=> M.files.findIndex(f=>f.id===id)+1;
  M.mergedLength = ()=> M.sections.reduce((sum, s)=> sum + (s.end-s.start), 0);
  M.fileCount = ()=> new Set(M.sections.map(s=>s.fileId)).size;
  const pl = n=> n===1 ? '' : 's';

  function changed(){
    if(M.merged) setCollapsed(false);
    M.merged = false;
    render();
    // the merged recording on screen follows the sections
    if(M.shown==='merged') showSource(M.sections.length ? 'merged' : M.files[0].id);
    // the recording choices of the synchronization rows (js/prepare/sync-strip.js)
    if(typeof syncFilesChanged==='function') syncFilesChanged(cfg.prefix);
  }

  function render(){
    $('FileCount').textContent = String(M.files.length);
    list.innerHTML = '';
    M.files.forEach((f, fi)=>{
      const row = document.createElement('div');
      row.className = 'ef-file' + (M.shown===f.id ? ' shown' : '');
      row.innerHTML = `
        <div class="ef-file-head">
          <button class="ef-show" data-show title="Show this file in the plots below"><span class="ef-no">${fi+1}</span><span class="ef-name">${f.name}</span></button>
          <span class="ef-meta">${fmtSec(f.dur)} · ${f.fs} Hz</span>
          <div class="tb-spacer"></div>
          <button class="tb-btn small quiet" data-whole title="Add the whole file as one section">Whole file</button>
          ${M.files.length>1 ? `<button class="tb-btn small icon quiet remove" data-remove-file title="Remove this file" aria-label="Remove this file">${prepIcon('close')}</button>` : ''}
        </div>
        <div class="ef-lane" style="height:${fileLaneHeight(tracesOf(f).length)}px" title="Drag to select a section">${fileLaneSvg(tracesOf(f), f.dur)}</div>`;
      const lane = row.querySelector('.ef-lane');
      // the bands of the sections of this file, numbered by merge order
      M.sections.forEach((s, si)=>{
        if(s.fileId!==f.id) return;
        const band = document.createElement('div');
        band.className = 'ef-band' + (s.id===M.active ? ' active' : '');
        band.style.left = (s.start/f.dur*100)+'%';
        band.style.width = ((s.end-s.start)/f.dur*100)+'%';
        band.innerHTML = `<span class="ef-edge" data-edge="start"></span><span class="ef-band-no">${si+1}</span><span class="ef-band-range">${fmtRange(s.start, s.end)}</span><button class="ef-band-remove" title="Remove this section" aria-label="Remove section ${si+1}">${prepIcon('close', 12)}</button><span class="ef-edge" data-edge="end"></span>`;
        band.title = `Section ${si+1}: ${fmtRange(s.start, s.end)}. Drag to slide, drag an edge to change the length; arrow keys move it by 0.1 s (with Shift: 1.0 s).`;
        band.tabIndex = 0;
        band.setAttribute('aria-label', `Section ${si+1}, ${fmtRange(s.start, s.end)}`);
        wireBandDrag(band, lane, f, s);
        lane.appendChild(band);
      });
      wireLaneSelect(lane, f);
      row.querySelector('[data-whole]').addEventListener('click', ()=> addSection(f.id, 0, f.dur));
      row.querySelector('[data-show]').addEventListener('click', ()=> showSource(f.id));
      row.querySelector('[data-remove-file]')?.addEventListener('click', ()=>{
        M.files.splice(M.files.indexOf(f), 1);
        M.sections = M.sections.filter(s=>s.fileId!==f.id);
        if(M.shown===f.id) showSource(M.files[0].id);
        changed();
      });
      list.appendChild(row);
    });
    renderSections();
  }

  // drag on a lane: a new section from where the mouse went down to where it came up
  function wireLaneSelect(lane, f){
    lane.addEventListener('mousedown', e=>{
      if(e.button!==0) return;
      e.preventDefault();
      const r = lane.getBoundingClientRect();
      const tAt = x=> Math.max(0, Math.min(f.dur, (x-r.left)/r.width*f.dur));
      const t0 = tAt(e.clientX);
      const ghost = document.createElement('div');
      ghost.className = 'ef-band ghost';
      lane.appendChild(ghost);
      const move = ev=>{
        const a = Math.min(t0, tAt(ev.clientX)), b = Math.max(t0, tAt(ev.clientX));
        ghost.style.left = (a/f.dur*100)+'%';
        ghost.style.width = ((b-a)/f.dur*100)+'%';
      };
      const up = ev=>{
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        ghost.remove();
        const a = Math.min(t0, tAt(ev.clientX)), b = Math.max(t0, tAt(ev.clientX));
        if(b-a >= SECTION_MIN) addSection(f.id, a, b);
      };
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
  }

  // drag on a band: the middle slides the section (same length), an edge moves
  // that end. Times snap to 0.1 s; a section is at least 0.5 s long and stays
  // inside its file.
  function placeBand(band, f, s){
    band.style.left = (s.start/f.dur*100)+'%';
    band.style.width = ((s.end-s.start)/f.dur*100)+'%';
    band.querySelector('.ef-band-range').textContent = fmtRange(s.start, s.end);
    // the matching row of the merge order follows the drag
    const row = secList.children[M.sections.indexOf(s)];
    if(row && row.classList.contains('ef-sec')){
      row.querySelector('[data-edge="start"]').value = s.start.toFixed(1);
      row.querySelector('[data-edge="end"]').value = s.end.toFixed(1);
      row.querySelector('.ef-sec-len').textContent = fmtSec(s.end-s.start);
    }
  }
  function wireBandDrag(band, lane, f, s){
    band.querySelector('.ef-band-remove').addEventListener('click', e=>{
      e.stopPropagation();
      M.sections = M.sections.filter(x=>x.id!==s.id);
      changed();
    });
    band.addEventListener('mousedown', e=>{
      if(e.button!==0 || e.target.closest('.ef-band-remove')){ e.stopPropagation(); return; }
      e.preventDefault(); e.stopPropagation();
      const mode = e.target.closest('.ef-edge')?.dataset.edge || 'slide';
      const pxPerSec = lane.getBoundingClientRect().width / f.dur;
      const x0 = e.clientX, start0 = s.start, end0 = s.end, len = end0 - start0;
      let moved = false;
      band.classList.add('dragging', 'active');
      lane.classList.add('dragging');
      const move = ev=>{
        const dt = (ev.clientX - x0) / pxPerSec;
        if(!moved && Math.abs(ev.clientX - x0) < 3) return;
        moved = true;
        if(mode==='slide'){
          s.start = snap01(Math.max(0, Math.min(f.dur - len, start0 + dt)));
          s.end = snap01(s.start + len);
        } else if(mode==='start'){
          s.start = snap01(Math.max(0, Math.min(end0 - SECTION_MIN, start0 + dt)));
        } else {
          s.end = snap01(Math.min(f.dur, Math.max(start0 + SECTION_MIN, end0 + dt)));
        }
        placeBand(band, f, s);
      };
      const up = ()=>{
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        band.classList.remove('dragging');
        lane.classList.remove('dragging');
        M.active = s.id;
        if(moved) changed();
        else render();
      };
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
    band.addEventListener('keydown', e=>{
      if(e.key!=='ArrowLeft' && e.key!=='ArrowRight') return;
      e.preventDefault();
      const len = s.end - s.start, step = (e.shiftKey ? 1.0 : 0.1) * (e.key==='ArrowLeft' ? -1 : 1);
      s.start = snap01(Math.max(0, Math.min(f.dur - len, s.start + step)));
      s.end = snap01(s.start + len);
      M.active = s.id;
      const no = M.sections.indexOf(s);
      changed();
      // keep the keyboard on the same band after the redraw
      [...list.querySelectorAll('.ef-band')]
        .find(b=> b.querySelector('.ef-band-no').textContent===String(no+1))?.focus();
    });
  }

  function addSection(fileId, start, end){
    M.sectionSeq++;
    const s = {id:'s'+M.sectionSeq, fileId, start:snap01(start), end:snap01(end)};
    M.sections.push(s);
    M.active = s.id;
    changed();
  }

  function renderSections(){
    secList.innerHTML = '';
    if(!M.sections.length){
      secList.innerHTML = '<div class="windows-empty">No sections yet. Drag on a file above, or click "Whole file".</div>';
    }
    M.sections.forEach((s, i)=>{
      const f = M.file(s.fileId);
      const row = document.createElement('div');
      row.className = 'ef-sec' + (s.id===M.active ? ' active' : '');
      row.innerHTML = `
        <span class="ef-sec-no">${i+1}</span>
        <span class="ef-sec-file" title="${f.name}">File ${M.fileNo(f.id)}</span>
        <input type="number" step="0.1" min="0" max="${f.dur}" value="${s.start.toFixed(1)}" data-edge="start" aria-label="Start of section ${i+1} in seconds">
        <span class="ef-dash">–</span>
        <input type="number" step="0.1" min="0" max="${f.dur}" value="${s.end.toFixed(1)}" data-edge="end" aria-label="End of section ${i+1} in seconds">
        <span class="ef-unit">s</span>
        <span class="ef-sec-len">${fmtSec(s.end-s.start)}</span>
        <div class="tb-spacer"></div>
        <div class="tb-seg small ef-order">
          <button data-move="-1" title="Earlier in the merged recording" aria-label="Move up" ${i===0?'disabled':''}>${prepIcon('chevron-down').replace('<svg ', '<svg style="transform:rotate(180deg)" ')}</button>
          <button data-move="1" title="Later in the merged recording" aria-label="Move down" ${i===M.sections.length-1?'disabled':''}>${prepIcon('chevron-down')}</button>
        </div>
        <button class="tb-btn small icon quiet remove" data-remove title="Remove this section" aria-label="Remove this section">${prepIcon('close')}</button>`;
      row.addEventListener('click', e=>{
        if(e.target.closest('button, input')) return;
        M.active = s.id; render();
      });
      row.querySelectorAll('input[data-edge]').forEach(inp=>{
        inp.addEventListener('change', ()=>{
          const v = Math.max(0, Math.min(f.dur, parseFloat(inp.value)||0));
          if(inp.dataset.edge==='start') s.start = Math.min(v, s.end-SECTION_MIN);
          else s.end = Math.max(v, s.start+SECTION_MIN);
          M.active = s.id;
          changed();
        });
      });
      row.querySelectorAll('[data-move]').forEach(btn=> btn.addEventListener('click', ()=>{
        const j = i + (+btn.dataset.move);
        [M.sections[i], M.sections[j]] = [M.sections[j], M.sections[i]];
        changed();
      }));
      row.querySelector('[data-remove]').addEventListener('click', ()=>{
        M.sections = M.sections.filter(x=>x.id!==s.id);
        changed();
      });
      secList.appendChild(row);
    });

    // merged preview: the sections side by side, as wide as their length
    preview.innerHTML = '';
    preview.hidden = !M.sections.length;
    preview.classList.toggle('shown', M.shown==='merged');
    if(M.sections.length) preview.style.height = previewHeight(tracesOf(M.file(M.sections[0].fileId)).length)+'px';
    M.sections.forEach((s, i)=>{
      const f = M.file(s.fileId);
      const seg = document.createElement('div');
      seg.className = 'ef-seg';
      seg.style.flex = String(s.end-s.start);
      seg.title = `Section ${i+1}: file ${M.fileNo(f.id)}, ${fmtRange(s.start, s.end)}`;
      seg.innerHTML = `<span class="ef-seg-no">${i+1}</span>` + fileLaneSvg(tracesOf(f), s.end-s.start, s.start, {labels:false});
      preview.appendChild(seg);
    });

    const n = M.sections.length, files = M.fileCount();
    summary.textContent = n ? `${n} section${pl(n)} from ${files} file${pl(files)} → ${fmtSec(M.mergedLength())}${M.merged ? ' · merged' : ''}` : '';
    const rates = [...new Set(M.sections.map(s=>M.file(s.fileId).fs))];
    warn.hidden = rates.length<2;
    warn.textContent = rates.length<2 ? '' : `Sampling rates differ (${rates.join(' Hz, ')} Hz). Merged sections need one rate.`;
    mergeBtn.disabled = !n || rates.length>1 || M.merged;
    mergeBtn.lastChild.textContent = M.merged ? 'Merged' : 'Merge sections';
    renderSourceSelect();
  }

  // ---- which recording the plots show ----
  function renderSourceSelect(){
    sourceSelect.innerHTML = M.files.map((f,i)=>
      `<option value="${f.id}">File ${i+1} · ${fmtSec(f.dur)}</option>`).join('')
      + (M.sections.length ? `<option value="merged">Merged${M.merged?'':' (preview)'} · ${fmtSec(M.mergedLength())}</option>` : '');
    sourceSelect.value = M.shown;
  }
  sourceSelect.addEventListener('change', ()=> showSource(sourceSelect.value));
  preview.addEventListener('click', ()=> showSource('merged'));
  preview.title = 'Show the merged recording in the plots below';

  function showSource(id){
    if(id==='merged' && !M.sections.length) id = M.files[0].id;
    if(id!=='merged' && !M.file(id)) id = M.files[0].id;
    M.shown = id;
    cfg.show(id, M);
    list.querySelectorAll('.ef-file').forEach((row,i)=> row.classList.toggle('shown', M.files[i] && M.files[i].id===id));
    preview.classList.toggle('shown', id==='merged');
    sourceSelect.value = id;
  }

  // ---- open and close the merge block ----
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'ef-merge-toggle';
  toggle.innerHTML = prepIcon('chevron-down');
  mergeHead.prepend(toggle);
  function setCollapsed(c){
    M.collapsed = c;
    mergeBox.classList.toggle('collapsed', c);
    toggle.setAttribute('aria-expanded', String(!c));
    toggle.setAttribute('aria-label', c ? 'Open merge order' : 'Close merge order');
    mergeHead.title = c ? 'Show the merge order' : 'Hide the merge order';
  }
  mergeHead.addEventListener('click', ()=> setCollapsed(!M.collapsed));
  setCollapsed(false);

  // "Merge sections": the card head and the side panel show the merged recording,
  // the synchronization row of this modality uses it, and the merge block closes
  mergeBtn.addEventListener('click', ()=>{
    M.merged = true;
    const total = M.mergedLength(), files = M.fileCount(), card = document.getElementById(cfg.cardId);
    card.querySelector('.card-head .meta').textContent =
      `${cfg.mergedName} · ${fmtSec(total)} · ${M.file(M.sections[0].fileId).fs} Hz · ${M.sections.length} sections from ${files} file${pl(files)}`;
    card.querySelector('.card-head .pill').textContent = 'Merged';
    document.querySelector(`#${cfg.modRowId} .file`).textContent = `${cfg.mergedName} · ${fmtSec(total)}`;
    renderSections();
    showSource('merged');
    setCollapsed(true);
    if(typeof syncFilesChanged==='function') syncFilesChanged(cfg.prefix, 'merged');
    showEdgeNote(summary, `Merged ${M.sections.length} sections into one ${cfg.name} recording of ${fmtSec(total)}. The plots below show it now.`);
  });

  // "Add files…" adds one example file and "Add folder…" adds two. Each one
  // gets the name of file 1 with its own number (…_1_… → …_3_…) and an
  // example signal from cfg.newFile.
  const firstName = cfg.files[0].name;
  function addExampleFiles(count, btnId){
    for(let i=0;i<count;i++){
      M.fileSeq++;
      M.files.push(cfg.newFile(firstName.replace(/_\d+_/, `_${M.fileSeq}_`), M.fileSeq));
    }
    changed();
    showEdgeNote($(btnId), `Added ${count} example ${cfg.name} file${pl(count)}. The × on a file removes it.`);
  }
  $('AddFilesBtn').addEventListener('click', ()=> addExampleFiles(1, 'AddFilesBtn'));
  $('AddFolderBtn').addEventListener('click', ()=> addExampleFiles(2, 'AddFolderBtn'));

  // the time map of a merged recording: merged time t → [file, file time]
  M.mergedPieces = ()=>{
    let from = 0;
    return M.sections.map(s=>{ const p = {file:M.file(s.fileId), start:s.start, end:s.end, from}; from += s.end-s.start; return p; });
  };

  // ---- the state of the Files block, kept with the session (js/core/session.js) ----
  // A file is kept as its id and name. An example file of the start list
  // comes back as it was; any other file gets its example signal again from
  // cfg.newFile(name, n), with n the number in its id ('f3' → 3).
  const startFiles = [...cfg.files];
  M.snapshot = ()=>({
    files: M.files.map(f=>({id:f.id, name:f.name})),
    sections: M.sections.map(s=>({id:s.id, fileId:s.fileId, start:s.start, end:s.end})),
    shown: M.shown, merged: M.merged, active: M.active, collapsed: M.collapsed,
    sectionSeq: M.sectionSeq, fileSeq: M.fileSeq,
  });
  M.restore = st=>{
    if(!Array.isArray(st.files) || !st.files.length) throw new Error(`${cfg.name}: the session has no files`);
    const files = st.files.map(sf=>{
      const known = startFiles.find(f=>f.id===sf.id && f.name===sf.name);
      if(known) return known;
      const n = Number(String(sf.id).replace(/^f/, ''));
      if(!Number.isInteger(n) || n<1) throw new Error(`${cfg.name}: file id "${sf.id}" is not f1, f2, …`);
      return cfg.newFile(String(sf.name), n);
    });
    M.files.splice(0, M.files.length, ...files);   // the same list as cfg.files
    M.sections = (st.sections || []).filter(s=> M.file(s.fileId))
      .map(s=>({id:String(s.id), fileId:s.fileId, start:+s.start, end:+s.end}));
    M.active = st.active || null;
    M.merged = !!st.merged && M.sections.length>0;
    M.sectionSeq = +st.sectionSeq || M.sections.length;
    M.fileSeq = +st.fileSeq || M.files.length;
    render();
    showSource(st.shown);
    setCollapsed(!!st.collapsed);
  };

  M.render = render;
  M.showSource = showSource;
  render();
  return M;
}
