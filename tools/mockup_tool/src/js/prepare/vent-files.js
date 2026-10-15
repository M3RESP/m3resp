// ================= Ventilator: several files, sections, merge =================
// The Files block of the ventilator card (shared code: js/prepare/file-merge.js).
// File 1 is the ventilator recording of this card; file 2 is an example of a
// second file. Every file holds the same channels (the chips in the Files
// block). An example file reads the channels of file 1 from another start time:
//   shift  file time 0 of this file is file time `shift` of file 1 (s)
// Files added with the file picker get an example signal too: this mockup
// reads no file contents.

const VENT_FILES = [
  {id:'f1', name:'m3resp_multimodal_1_ventilator.Poly5', dur:REC_SECONDS, fs:200, shift:0, real:true},
  {id:'f2', name:'m3resp_multimodal_2_ventilator.Poly5', dur:50.0, fs:200, shift:20.0},
];
// file time t of file f → file time of file 1
const ventFileAt = (f, t)=> ({tau:(t + f.shift) % REC_SECONDS, gain:1});

// the file lanes draw each ticked channel, below each other
function ventLaneFn(f){
  return VENT_CHANNELS.filter(c=> c.on).map(chan=>({
    color: chan.color, label: chan.name,
    fn: t=> chan.fn(ventPhase(ventFileAt(f, t).tau)),
  }));
}

function ventSourceNote(id, M){
  const pf = s=> s===1 ? '' : 's';
  if(id==='merged'){
    const files = M.fileCount();
    return `<b>Merged recording${M.merged?'':' (preview)'}</b>: ${M.sections.length} section${pf(M.sections.length)} from ${files} file${pf(files)}, joined in merge order; dashed lines mark the joins. Synchronization and the working window use file 1.`;
  }
  const f = M.file(id);
  if(f.real) return `<b>File ${M.fileNo(id)}</b>: <code>${f.name}</code>, ${fmtSec(f.dur)} at ${f.fs} Hz.`;
  return `<b>File ${M.fileNo(id)}</b> (example): <code>${f.name}</code>. The channels are those of file 1 from ${fmtSec(f.shift)} on. Synchronization and the working window use file 1.`;
}

const VENT_MERGE = makeFileMerger({
  prefix:'vent', name:'ventilator', color:'var(--vent)', cardId:'cardVent', modRowId:'modRow-vent', mergedName:'merged_ventilator',
  files:VENT_FILES,
  sections:[
    {id:'s1', fileId:'f1', start:2.0, end:24.0},
    {id:'s2', fileId:'f2', start:8.0, end:26.0},
  ],
  laneFn:ventLaneFn,
  exts:/\.poly5$/i, extText:'.Poly5',
  newFile:(name, n)=> ({id:'f'+n, name, dur:REC_SECONDS, fs:200, shift:(n*7.3) % REC_SECONDS}),
  show:(id, M)=>{
    if(id==='merged'){
      const pieces = M.mergedPieces();
      const last = pieces[pieces.length-1];
      VENT_VIEW = {
        dur: last.from + (last.end-last.start),
        at: t=>{
          let p = pieces[0];
          for(const q of pieces) if(t >= q.from) p = q;
          return ventFileAt(p.file, Math.min(p.end, p.start + (t - p.from)));
        },
        joins: pieces.slice(1).map(p=>p.from),
      };
    } else {
      const f = M.file(id);
      VENT_VIEW = {dur:f.dur, at:t=> ventFileAt(f, Math.min(f.dur, t)), joins:[]};
    }
    document.getElementById('ventSourceNote').innerHTML = ventSourceNote(id, M);
    if(ZOOM.vent) ZOOM.vent.setFullDur(VENT_VIEW.dur);
  },
});
VENT_MERGE.showSource('f1');
