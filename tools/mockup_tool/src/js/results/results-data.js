// ================= 4 · Results: the values, plots and files of the run =================
// The values are those of the multimodal-full run (the means, medians and
// single values). The values of single breaths are example values with the
// same mean as the run (qcSeries in js/check/check-data.js); the EMG
// breaths with their times are those of 3 · Check.
//   scope   'run' one value, 'breath' one value per breath, 'occlusion' one
//           value per occlusion manoeuvre
//   stat    how the row value comes from the breath values ('mean')
//   show    the text in the value column when it is no single number
//   file    the file that holds the value, and key: its name in that file

// EIT breath i of the run: the peak from eit.detect_breaths, and a window
// from half way to the previous peak to half way to the next one
const RUN_EIT_PEAKS = REVIEW.eit_peaks;
function resEitBreath(i){
  const p = RUN_EIT_PEAKS;
  const before = i>0 ? (p[i]-p[i-1])/2 : 2.0, after = i<p.length-1 ? (p[i+1]-p[i])/2 : 2.0;
  return {t0: p[i]-before, t1: p[i]+after, peak: p[i]};
}
const resBreaths = (n, at, values)=> values.map((value, i)=>{ const b = at(i); return {id:String(i+1), t0:b.t0, t1:b.t1, value}; });
const resEitItems = values=> resBreaths(RUN_EIT_PEAKS.length, resEitBreath, values);
const resEmgItems = values=> resBreaths(RUN_EMG_PEAKS.length, qcEmgBreath, values);
const qcItemsOf = code=> QC_CHECKS.find(c=> c.code===code).items;

const RES_MODS = [
  {id:'eit',     name:'EIT',        valCard:'resValEit',     plotCard:'resPlotEit',  prepCard:'cardEit',
   meta:'13 breaths · Dräger, 20 Hz'},
  {id:'emg',     name:'EMG',        valCard:'resValEmg',     plotCard:'resPlotEmg',  prepCard:'cardEmg',
   meta:'19 breaths · EMGdi, 2048 Hz'},
  {id:'vent',    name:'Ventilator', valCard:'resValVent',    plotCard:'resPlotVent', prepCard:'cardVent',
   meta:'2 occlusions · P_aw, 200 Hz'},
  {id:'session', name:'Session',    valCard:'resValSession', plotCard:null,          prepCard:'cardSync',
   meta:'alignment of the three recordings'},
];

const RES_VALUES = [
  // ---------------- EIT ----------------
  {mod:'eit', code:'respiratory_rate', name:'Respiratory rate', value:13.9, unit:'br/min', digits:1, scope:'run', top:true,
   what:'Breathing rate of the global impedance, from the highest peak of its power spectrum.',
   step:'eit.detect_rates', stepMs:72.2, settings:[['method','Welch'], ['window','30 s']],
   file:'parameter_results.csv', key:'respiratory_rate'},
  {mod:'eit', code:'heart_rate', name:'Heart rate', value:95.8, unit:'bpm', digits:1, scope:'run', top:true,
   what:'Heart rate in the EIT signal. eit.mdn_filter removes this rate from the impedance before the breaths are found.',
   step:'eit.detect_rates', stepMs:72.2, settings:[['method','Welch'], ['window','30 s']],
   file:'parameter_results.csv', key:'heart_rate'},
  {mod:'eit', code:'pixel_breaths', name:'Pixel breaths found', value:84.2, unit:'%', digits:1, scope:'breath', stat:'mean', top:true,
   what:'Part of the pixel-breaths with a start, a peak and an end. A pixel-breath is one breath in one pixel; the run has 11212 of 13320.',
   step:'eit.pixel_breaths', stepMs:369.3, settings:[],
   file:'parameter_result_arrays.npz', key:'pixel_breaths',
   items: resEitItems(qcSeries(13, 84.2, 7, 70))},
  {mod:'eit', code:'pixel_tivs', name:'Pixel TIV found', value:100, unit:'%', digits:1, scope:'breath', stat:'mean',
   what:'Part of the breaths with a tidal impedance variation (TIV) in every lung pixel. The run has 13 of 13.',
   step:'eit.pixel_tiv', stepMs:23.5, settings:[],
   file:'parameter_result_arrays.npz', key:'pixel_tivs',
   items: resEitItems(Array(13).fill(100))},
  {mod:'eit', code:'watershed_lungspace_mask', name:'Lung area, watershed', value:302, unit:'px', digits:0, scope:'run', top:true,
   show:'302 of 1024 px', mask:{px:302, seed:1},
   what:'Pixels inside the lungs, found with the watershed method on the pixel TIV. 302 of the 1024 pixels (29.5 %) are in the lungs.',
   step:'eit.roi_watershed', stepMs:785.1, settings:[],
   file:'parameter_result_arrays.npz', key:'watershed_lungspace_mask'},
  {mod:'eit', code:'tiv_lungspace_mask', name:'Lung area, TIV', value:301, unit:'px', digits:0, scope:'run',
   show:'301 of 1024 px', mask:{px:301, seed:2},
   what:'Pixels inside the lungs: the pixels with a TIV of at least 0.15 times the largest pixel TIV.',
   step:'eit.roi_tiv_lungspace', stepMs:23.3, settings:[['threshold','0.15']],
   file:'parameter_result_arrays.npz', key:'tiv_lungspace_mask'},

  // ---------------- EMG ----------------
  {mod:'emg', code:'snr_pseudo', name:'Signal-to-noise ratio', value:0.879, unit:'', digits:3, scope:'breath', stat:'mean', top:true,
   what:'Height of each EMG breath peak divided by the baseline at the peak (pseudo signal-to-noise ratio).',
   step:'emg.snr_pseudo', stepMs:0.9, settings:[],
   file:'parameter_results.csv', key:'snr_pseudo',
   items: resEmgItems(qcSeries(19, 0.879, 0.14, 90))},
  {mod:'emg', code:'percentage_under_baseline', name:'Time under baseline', value:37.7, unit:'%', digits:1, scope:'breath', stat:'mean', top:true,
   what:'Part of each EMG breath (onset to offset) that lies under the baseline.',
   step:'emg.percentage_under_baseline', stepMs:22.9, settings:[['aub_window_seconds','the breath (onset to offset)']],
   limit:40, dir:'max',
   file:'parameter_results.csv', key:'percentage_under_baseline',
   items: resEmgItems(qcItemsOf('percentage_under_baseline').map(x=> x.value))},
  {mod:'emg', code:'interpeak_dist_emg_median', name:'EMG peak spacing', value:2.484, unit:'s', digits:3, scope:'run', top:true,
   what:'Median time between two EMG breath peaks.',
   step:'emg.interpeak_dist', stepMs:0.7, settings:[['threshold','1.1']], checks:['interpeak_dist'],
   file:'parameter_results.csv', key:'interpeak_dist_emg_median'},
  {mod:'emg', code:'interpeak_dist_ecg_median', name:'ECG peak spacing', value:2.123, unit:'s', digits:3, scope:'run',
   what:'Median time between two ECG peaks in the EMG signal.',
   step:'emg.interpeak_dist', stepMs:0.7, settings:[['threshold','1.1']], checks:['interpeak_dist'],
   file:'parameter_results.csv', key:'interpeak_dist_ecg_median'},
  {mod:'emg', code:'ecg_gate_mask', name:'ECG gate width', value:205, unit:'samples', digits:0, scope:'run',
   what:'Width of the gate around each ECG peak. emg.ecg_gating fills the EMG signal inside each gate.',
   step:'emg.ecg_gating', stepMs:3.8, settings:[['fill_method','1'], ['envelope','RMS, 0.5 s']],
   file:'parameter_result_arrays.npz', key:'ecg_gate_mask'},

  // ---------------- ventilator ----------------
  {mod:'vent', code:'pocc_quality_dp_up_90', name:'Pocc upslope, 9th decile', unit:'cmH₂O', digits:4, scope:'occlusion', top:true,
   limit:2, dir:'max',
   what:'9th decile of the pressure upslope after each occlusion. A high value means that the occlusion ended before the breath effort.',
   step:'ventilator.pocc_quality', stepMs:0.7, settings:[['dp_up_90_threshold','2']], checks:['pocc_quality'],
   file:'parameter_results.csv', key:'pocc_quality_dp_up_90',
   items: REVIEW.pocc.dp_up_90.map((v, i)=> ({id:'Pocc '+(i+1), t0:[40.2, 43.3][i], t1:[41.4, 44.5][i], value:v}))},
  {mod:'vent', code:'pocc_quality_dp_up_10', name:'Pocc upslope, 1st decile', unit:'cmH₂O', digits:4, scope:'occlusion', top:true,
   limit:0, dir:'min',
   what:'1st decile of the pressure upslope after each occlusion. A value under 0 means that the upslope dips.',
   step:'ventilator.pocc_quality', stepMs:0.7, settings:[['dp_up_10_threshold','0']], checks:['pocc_quality'],
   file:'parameter_results.csv', key:'pocc_quality_dp_up_10',
   items: REVIEW.pocc.dp_up_10.map((v, i)=> ({id:'Pocc '+(i+1), t0:[40.2, 43.3][i], t1:[41.4, 44.5][i], value:v}))},

  // ---------------- session ----------------
  {mod:'session', code:'raw_alignment.method', name:'Alignment method', show:'manual offset', scope:'run', top:true,
   what:'How the three recordings were put on one time line: here with the offsets set in 1 · Prepare.',
   step:'session.sync_raw', stepMs:14.0, settings:[],
   file:'summary.json', key:'raw_alignment.method'},
  {mod:'session', code:'raw_alignment.offset_seconds', name:'Offsets', show:'EIT 0.0 · EMG −2.0 · Ventilator 0.0 s', scope:'run', top:true,
   what:'Start of each recording on the ventilator time line. The EMG file starts 2.0 s before the ventilator.',
   step:'session.sync_raw', stepMs:14.0, settings:[],
   file:'summary.json, parameters.csv', key:'raw_alignment.offset_seconds'},
  {mod:'session', code:'raw_alignment.cropped_samples', name:'EMG samples cut', value:4096, unit:'samples', digits:0, scope:'run',
   what:'Samples cut from the start of the EMG recording so that it starts with the ventilator (2.0 s at 2048 Hz).',
   step:'session.sync_raw', stepMs:14.0, settings:[],
   file:'summary.json', key:'raw_alignment.cropped_samples (emg)'},
];
RES_VALUES.forEach(v=>{ v.id = 'v-'+v.code.replace(/[^a-z0-9]+/gi, '-'); });

// plots: 'img' a figure the run saved, the others are drawn from the saved values
const RES_PLOTS = [
  {id:'global_impedance', mod:'eit', kind:'img', name:'Global impedance with breaths, TIV and EELI',
   step:'eit.global_impedance', src:'figures/global_impedance.png', img:REVIEW.figures['global_impedance.png']},
  {id:'pixel_tiv', mod:'eit', kind:'img', name:'Mean pixel TIV, 13 breaths',
   step:'eit.pixel_tiv', src:'figures/pixel_tiv.png', img:REVIEW.figures['pixel_tiv.png']},
  {id:'rate_detection', mod:'eit', kind:'img', name:'Respiratory and heart rate',
   step:'eit.detect_rates', src:'figures/rate_detection.png', img:REVIEW.figures['rate_detection.png']},
  {id:'lung_masks', mod:'eit', kind:'masks', name:'Lung areas',
   step:'eit.roi_watershed', src:'parameter_result_arrays.npz'},
  {id:'emg_breaths', mod:'emg', kind:'emgBreaths', name:'EMGdi envelope with breaths',
   step:'emg.detect_breaths', src:'emg_breaths.csv'},
  {id:'emg_breath_values', mod:'emg', kind:'breathValues', name:'Time under baseline per breath',
   step:'emg.percentage_under_baseline', src:'parameter_results.csv'},
  {id:'pocc_quality', mod:'vent', kind:'pocc', name:'Pocc upslope against the limits',
   step:'ventilator.pocc_quality', src:'parameter_results.csv'},
];

// the files of the run (sizes of the tables and summaries are example sizes)
const RES_OUT_DIR = 'outputs/2026-06-10_153009/multimodal-full_9907d215dda2';
const pngKB = b64=> Math.round(b64.length*0.75/1024);
const RES_FILES = [
  {group:'Values', name:'parameter_results.csv', what:'One row per value; per-breath values have one row per breath', rows:'53', size:'14 KB', part:'resValues'},
  {group:'Values', name:'parameter_result_arrays.npz', what:'Arrays: pixel breaths, pixel TIV, lung masks, ECG gate', rows:'5 arrays', size:'61 KB', part:'resValues'},
  {group:'Values', name:'parameters.csv', what:'Settings of the session: offsets and step parameters', rows:'38', size:'3 KB'},
  {group:'Breaths and checks', name:'eit_breaths.csv', what:'EIT breaths: start, peak and end', rows:'13', size:'1 KB'},
  {group:'Breaths and checks', name:'emg_breaths.csv', what:'EMG breaths: onset, peak and offset', rows:'19', size:'2 KB'},
  {group:'Breaths and checks', name:'ventilator_breaths.csv', what:'Ventilator breaths and occlusions', rows:'14', size:'1 KB'},
  {group:'Breaths and checks', name:'linked_breaths.csv', what:'EMG and EIT breaths linked to each ventilator breath', rows:'14', size:'2 KB'},
  {group:'Breaths and checks', name:'quality_flags.csv', what:'One row per quality check and breath', rows:'96', size:'11 KB', check:true},
  {group:'Figures', name:'figures/global_impedance.png', what:'Global impedance with breaths, TIV and EELI', size:pngKB(REVIEW.figures['global_impedance.png'])+' KB', part:'resPlotEit'},
  {group:'Figures', name:'figures/pixel_tiv.png', what:'Mean pixel TIV', size:pngKB(REVIEW.figures['pixel_tiv.png'])+' KB', part:'resPlotEit'},
  {group:'Figures', name:'figures/rate_detection.png', what:'Respiratory and heart rate', size:pngKB(REVIEW.figures['rate_detection.png'])+' KB', part:'resPlotEit'},
  {group:'Run', name:'summary.json', what:'Metadata, quality, values and provenance in one file', size:'48 KB'},
  {group:'Run', name:'run_manifest.json', what:'Run id, start, end and status', size:'1 KB'},
  {group:'Run', name:'processing_history.json', what:'Each of the 47 steps with its settings', size:'22 KB'},
  {group:'Run', name:'session_metadata.json', what:'Session name, recordings and devices', size:'2 KB'},
  {group:'Run', name:'signals_manifest.csv', what:'Each signal with its unit and sampling rate', rows:'9', size:'1 KB'},
];
