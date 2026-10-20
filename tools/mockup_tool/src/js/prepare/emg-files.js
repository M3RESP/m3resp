// ================= EMG: several files, sections, merge =================
// The Files block of the EMG card (shared code: js/prepare/file-merge.js).
// File 1 is the EMG recording of this card; file 2 is an example of a second
// file. Every file holds the same EMG signals (the chips in the toolbar).
// An example file reads the signals of file 1 from another start time:
//   shift  file time 0 of this file is file time `shift` of file 1 (s)
//   gain   size of the signals against file 1
// Files added with the file picker get an example signal too: this mockup
// reads no file contents.

const EMG_FILES = [
  {id:'f1', name:'m3resp_multimodal_1_emg.Poly5', dur:REC_SECONDS, fs:2048, shift:0, gain:1, real:true},
  {id:'f2', name:'m3resp_multimodal_2_emg.Poly5', dur:45.0, fs:2048, shift:12.5, gain:0.8},
];
// file time t of file f → file time of file 1 and size
const emgFileAt = (f, t)=> ({tau:(t + f.shift) % REC_SECONDS, gain:f.gain});

// the file lanes draw the envelope (ECG removed) of each ticked EMG signal,
// below each other
function emgLaneFn(f){
  return EMG_CHANNELS.filter(c=> c.on).map(chan=>({
    color: chan.color, label: chan.name,
    fn: t=>{ const p = emgFileAt(f, t); return p.gain * emgEnvelopeAt(p.tau, chan.id); },
  }));
}

function emgSourceNote(id, M){
  const pf = s=> s===1 ? '' : 's';
  if(id==='merged'){
    const files = M.fileCount();
    return `<b>Merged recording${M.merged?'':' (preview)'}</b>: ${M.sections.length} section${pf(M.sections.length)} from ${files} file${pf(files)}, joined in merge order; dashed lines mark the joins. Synchronization and the working window use file 1.`;
  }
  const f = M.file(id);
  if(f.real) return `<b>File ${M.fileNo(id)}</b>: <code>${f.name}</code>, ${fmtSec(f.dur)} at ${f.fs} Hz.`;
  return `<b>File ${M.fileNo(id)}</b> (example): <code>${f.name}</code>. The signals are those of file 1 from ${fmtSec(f.shift)} on, at ${Math.round(f.gain*100)} % size. Synchronization and the working window use file 1.`;
}

const EMG_MERGE = makeFileMerger({
  prefix:'emg', name:'EMG', color:'var(--emg)', cardId:'cardEmg', modRowId:'modRow-emg', mergedName:'merged_emg',
  files:EMG_FILES,
  sections:[
    {id:'s1', fileId:'f1', start:6.0, end:26.0},
    {id:'s2', fileId:'f2', start:5.0, end:20.0},
  ],
  laneFn:emgLaneFn,
  exts:/\.poly5$/i, extText:'.Poly5',
  newFile:(name, n)=> ({id:'f'+n, name, dur:REC_SECONDS, fs:2048, shift:(n*7.3) % REC_SECONDS, gain:0.7 + 0.1*(n % 4)}),
  show:(id, M)=>{
    if(id==='merged'){
      const pieces = M.mergedPieces();
      const last = pieces[pieces.length-1];
      EMG_VIEW = {
        dur: last.from + (last.end-last.start),
        at: t=>{
          let p = pieces[0];
          for(const q of pieces) if(t >= q.from) p = q;
          return emgFileAt(p.file, Math.min(p.end, p.start + (t - p.from)));
        },
        joins: pieces.slice(1).map(p=>p.from),
      };
    } else {
      const f = M.file(id);
      EMG_VIEW = {dur:f.dur, at:t=> emgFileAt(f, Math.min(f.dur, t)), joins:[]};
    }
    document.getElementById('emgSourceNote').innerHTML = emgSourceNote(id, M);
    if(ZOOM.emg) ZOOM.emg.setFullDur(EMG_VIEW.dur);
  },
});
EMG_MERGE.showSource('f1');
