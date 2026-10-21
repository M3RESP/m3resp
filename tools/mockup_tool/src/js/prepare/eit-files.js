// ================= EIT: several files, sections, merge =================
// The Files block of the EIT card (shared code: js/prepare/file-merge.js).
// File 1 is the real recording of this card; file 2 is an example of a
// second file (a part of the same signal, scaled, with other ROI shares).
// Files added with "Add files…" or "Add folder…" get an example signal too.
// The plots and images under the Files block show one file, or the merged
// recording.
//   roiWeights: size of each ROI curve (ventral to dorsal) against the global curve
//   gains:      minute image per ROI, against the real one

const EIT_FILES = [
  {id:'f1', name:'m3resp_multimodal_1_eit_draeger.bin', dur:REC_SECONDS, fs:20, fn:eitOverviewAt,
   roiWeights:REAL_ROI_WEIGHTS, gains:[1,1,1,1], real:true},
  {id:'f2', name:'m3resp_multimodal_2_eit_draeger.bin', dur:45.0, fs:20,
   fn:t=> eitOverviewAt((t + 9.5) % REC_SECONDS)*0.85 + 8,
   roiWeights:[0.10, 0.40, 0.41, 0.07], gains:[1.5, 0.9, 1.0, 2.2]},
];
const eitDetailCache = {};
function fileDetailOf(f){
  if(!eitDetailCache[f.id]) eitDetailCache[f.id] = makeFileDetail({
    dur:f.dur, globalAt:f.fn, weights:f.roiWeights, gains:f.gains, realFrame: f.real ? realEitFrame : null});
  return eitDetailCache[f.id];
}

function eitDetailNote(id, d, M){
  const pf = s=> s===1 ? '' : 's';
  const tail = ` ROI 1 to 4 are ventral to dorsal quarters of the chest. MTV is the mean tidal variation over ${d.breaths} breath${pf(d.breaths)}.`;
  if(id==='merged'){
    const files = M.fileCount();
    return `<b>Merged recording${M.merged?'':' (preview)'}</b>: ${M.sections.length} section${pf(M.sections.length)} from ${files} file${pf(files)}, joined in merge order; dashed lines mark the joins.` + tail;
  }
  const f = M.file(id);
  if(f.real) return `<b>File ${M.fileNo(id)}</b>: the global impedance and the minute image come from the real reconstruction of <code>${f.name}</code> (32 × 32 pixels, loaded with <code>eitprocessing</code>). The dynamic image is real from 3.0 to 13.0 s; elsewhere, and for the ROI curves, it comes from the global impedance and the minute image.` + tail;
  return `<b>File ${M.fileNo(id)}</b> (example): <code>${f.name}</code>. Curves and images come from an example global impedance and the minute image of file 1 with other ROI shares.` + tail;
}

const EIT_MERGE = makeFileMerger({
  prefix:'eit', name:'EIT', color:'var(--eit)', cardId:'cardEit', modRowId:'modRow-eit', mergedName:'merged_eit',
  files:EIT_FILES,
  sections:[
    {id:'s1', fileId:'f1', start:4.0, end:22.0},
    {id:'s2', fileId:'f2', start:10.0, end:30.0},
  ],
  laneFn:f=> f.fn,
  newFile:(name, n)=>{
    const shift = (n*7.3) % REC_SECONDS, k = n % 3;
    return {id:'f'+n, name, dur:REC_SECONDS, fs:20,
      fn:t=> eitOverviewAt((t + shift) % REC_SECONDS),
      roiWeights:[0.06+0.02*k, 0.44-0.03*k, 0.42+0.02*k, 0.05+0.01*k], gains:[1+0.3*k, 1, 1-0.1*k, 1+0.5*k]};
  },
  show:(id, M)=>{
    const d = id==='merged'
      ? makeMergedDetail(M.sections.map(s=>({detail:fileDetailOf(M.file(s.fileId)), start:s.start, end:s.end})))
      : fileDetailOf(M.file(id));
    setEitDetail(d, eitDetailNote(id, d, M));
  },
});
EIT_MERGE.showSource('f1');
