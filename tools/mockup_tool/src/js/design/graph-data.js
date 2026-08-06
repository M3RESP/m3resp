// ---- palette data (subset of the real 60-step registry, grouped by modality/category) ----
const PALETTE = [
  {mod:'eit', label:'EIT', color:'var(--eit)', groups:[
    {cat:'loading', ops:[['eit.load','load']]},
    {cat:'preprocessing', ops:[['eit.mdn_filter','mdn_filter'],['eit.butterworth_filter','butterworth_filter'],['eit.global_impedance','global_impedance'],['eit.slice','slice']]},
    {cat:'detection', ops:[['eit.detect_breaths','detect_breaths'],['eit.detect_rates','detect_rates'],['eit.pixel_breaths','pixel_breaths']]},
    {cat:'roi', ops:[['eit.roi_watershed','roi_watershed'],['eit.roi_filter_by_size','roi_filter_by_size']]},
    {cat:'parameters', ops:[['eit.continuous_tiv','continuous_tiv'],['eit.eeli','eeli']]},
  ]},
  {mod:'emg', label:'EMG', color:'var(--emg)', groups:[
    {cat:'loading', ops:[['emg.load','load']]},
    {cat:'preprocessing', ops:[['emg.preprocess','preprocess'],['emg.ecg_gating','ecg_gating'],['emg.ecg_wavelet_denoising','ecg_wavelet_denoising',true]]},
    {cat:'detection', ops:[['emg.detect_breaths','detect_breaths'],['emg.peak_indices','peak_indices'],['emg.onoffpeak_baseline_crossing','onoffpeak_baseline_crossing']]},
    {cat:'features', ops:[['emg.time_product','time_product'],['emg.amplitude','amplitude'],['emg.pseudo_slope','pseudo_slope']]},
    {cat:'quality', ops:[['emg.snr_pseudo','snr_pseudo'],['emg.evaluate_bell_curve_error','evaluate_bell_curve_error'],['emg.detect_local_high_aub','detect_local_high_aub']]},
  ]},
  {mod:'vent', label:'Ventilator', color:'var(--vent)', groups:[
    {cat:'loading', ops:[['ventilator.load','load']]},
    {cat:'preprocessing', ops:[['ventilator.channels','channels']]},
    {cat:'detection', ops:[['ventilator.detect_breaths','detect_breaths'],['ventilator.find_occluded_breaths','find_occluded_breaths']]},
    {cat:'parameters', ops:[['ventilator.respiratory_rate','respiratory_rate'],['ventilator.pocc_time_product','pocc_time_product']]},
  ]},
  {mod:'generic', label:'Generic', color:'var(--generic)', groups:[
    {cat:'synchronization', ops:[['session.sync_raw','sync_raw'],['sync.estimate_offset','estimate_offset']]},
    {cat:'export', ops:[['export.session_summary','session_summary'],['export.json_file','json_file']]},
  ]},
];

const STEP_DEFS = {"eit.load":{"ins":[["sequence","sequence_bundle"]],"outs":[["raw_eit","eit_pixel_signal"],["raw_global_impedance","eit_global_impedance"],["eit_sequence","eit_sequence"]]},"eit.mdn_filter":{"ins":[["signal","eit_pixel_signal"],["respiratory_rate_hz","scalar_metric"],["heart_rate_hz","scalar_metric"]],"outs":[["filtered_eit","eit_pixel_signal"],["filter_captures","diagnostic_summary"],["filtered_eit_signal","signal"]]},"eit.butterworth_filter":{"ins":[["signal","eit_pixel_signal"],["eit_sequence","eit_sequence"]],"outs":[["filtered_eit","eit_pixel_signal"],["filter_captures","diagnostic_summary"]]},"eit.global_impedance":{"ins":[["signal","eit_pixel_signal"],["eit_sequence","eit_sequence"]],"outs":[["global_impedance","eit_global_impedance"]]},"eit.slice":{"ins":[["signal","any"]],"outs":[["result","any"]]},"eit.detect_breaths":{"ins":[["signal","eit_global_impedance"]],"outs":[["breath_intervals","interval_collection"],["breath_detector","eit_breath_detector"]]},"eit.detect_rates":{"ins":[["signal","eit_pixel_signal"],["session","m3session"]],"outs":[["respiratory_rate_hz","scalar_metric"],["heart_rate_hz","scalar_metric"],["rate_detector","eit_rate_detector"]]},"eit.pixel_breaths":{"ins":[["eit_data","eit_pixel_signal"],["timing_data","eit_global_impedance"],["eit_sequence","eit_sequence"]],"outs":[["pixel_breaths","eit_sparse_data"],["pixel_breath_timing_result","parameter_result"]]},"eit.roi_watershed":{"ins":[["eit_data","eit_pixel_signal"],["timing_data","eit_global_impedance"],["session","m3session"]],"outs":[["watershed_lungspace_mask","roi_mask"],["watershed_captures","diagnostic_summary"],["watershed_lungspace_result","parameter_result"]]},"eit.roi_filter_by_size":{"ins":[["mask","roi_mask"],["session","m3session"]],"outs":[["size_filtered_roi_mask","roi_mask"],["size_filtered_roi_result","parameter_result"]]},"eit.continuous_tiv":{"ins":[["signal","eit_global_impedance"],["eit_sequence","eit_sequence"],["breath_detector","eit_breath_detector"]],"outs":[["continuous_tiv","eit_sparse_data"]]},"eit.eeli":{"ins":[["signal","eit_global_impedance"],["eit_sequence","eit_sequence"],["breath_detector","eit_breath_detector"]],"outs":[["eeli","eit_sparse_data"],["eeli_result","parameter_result"]]},"emg.load":{"ins":[["sequence","sequence_bundle"]],"outs":[["emg_recording","emg_recording"],["raw_emg_signals","signal_list"]]},"emg.preprocess":{"ins":[["session","m3session"]],"outs":[["processed_emg","emg_processed_bundle"]]},"emg.ecg_gating":{"ins":[["session","m3session"],["processed_emg","emg_processed_bundle"],["ecg_peak_indices","index_array"]],"outs":[["ecg_gated_emg","signal_array"],["processed_emg_after_ecg","emg_processed_bundle"],["ecg_gated_signal","signal"]]},"emg.ecg_wavelet_denoising":{"ins":[["session","m3session"],["processed_emg","emg_processed_bundle"],["ecg_peak_indices","index_array"]],"outs":[["ecg_wavelet_cleaned_emg","signal_array"],["processed_emg_after_ecg","emg_processed_bundle"],["ecg_wavelet_cleaned_signal","signal"]]},"emg.detect_breaths":{"ins":[["session","m3session"]],"outs":[["emg_breath_events","breath_event_list"]]},"emg.peak_indices":{"ins":[["events","breath_event_list"],["processed_emg","emg_processed_bundle"]],"outs":[["peak_indices","index_array"]]},"emg.onoffpeak_baseline_crossing":{"ins":[["processed_emg","emg_processed_bundle"],["baseline","signal_array"],["peak_indices","index_array"]],"outs":[["start_indices","index_array"],["end_indices","index_array"],["start_end_validity","boolean_array"]]},"emg.time_product":{"ins":[["processed_emg","emg_processed_bundle"],["start_indices","index_array"],["end_indices","index_array"]],"outs":[["time_product","array"]]},"emg.amplitude":{"ins":[["processed_emg","emg_processed_bundle"],["peak_indices","index_array"],["baseline","signal_array"]],"outs":[["amplitude","array"]]},"emg.pseudo_slope":{"ins":[["processed_emg","emg_processed_bundle"],["start_indices","index_array"],["end_indices","index_array"]],"outs":[["pseudo_slope","array"]]},"emg.snr_pseudo":{"ins":[["session","m3session"],["processed_emg","emg_processed_bundle"],["peak_indices","index_array"]],"outs":[["snr_pseudo","array"],["snr_pseudo_results","parameter_result_list"],["snr_pseudo_flags","quality_flag_list"]]},"emg.evaluate_bell_curve_error":{"ins":[["session","m3session"],["peak_indices","index_array"],["start_indices","index_array"]],"outs":[["evaluate_bell_curve_error","array"],["evaluate_bell_curve_error_results","parameter_result_list"],["evaluate_bell_curve_error_flags","quality_flag_list"]]},"emg.detect_local_high_aub":{"ins":[["session","m3session"],["area_under_baseline","array"],["peak_indices","index_array"]],"outs":[["detect_local_high_aub","boolean_array"],["detect_local_high_aub_flags","quality_flag_list"],["detect_local_high_aub_threshold_result","parameter_result"]]},"ventilator.load":{"ins":[["sequence","sequence_bundle"]],"outs":[["ventilator_raw","ventilator_recording"]]},"ventilator.channels":{"ins":[["ventilator_raw","ventilator_recording"]],"outs":[["ventilator_signals","ventilator_channel_bundle"]]},"ventilator.detect_breaths":{"ins":[["ventilator_signals","ventilator_channel_bundle"]],"outs":[["ventilator_breath_indices","index_array"]]},"ventilator.find_occluded_breaths":{"ins":[["ventilator_signals","ventilator_channel_bundle"]],"outs":[["pocc_indices","index_array"]]},"ventilator.respiratory_rate":{"ins":[["ventilator_breath_indices","index_array"],["ventilator_signals","ventilator_channel_bundle"]],"outs":[["ventilator_respiratory_rate","scalar_metric"]]},"ventilator.pocc_time_product":{"ins":[["session","m3session"],["ventilator_signals","ventilator_channel_bundle"],["pocc_start_indices","index_array"]],"outs":[["pocc_time_products","array"],["pocc_time_product_result","parameter_result"]]},"session.sync_raw":{"ins":[["session","m3session"]],"outs":[["sync_summary","sync_summary"]]},"sync.estimate_offset":{"ins":[["session","m3session"]],"outs":[["estimated_offset_seconds","scalar_metric"],["offset_estimation","diagnostic_summary"]]},"export.session_summary":{"ins":[["session","m3session"]],"outs":[["output_dir","directory_path"]]},"export.json_file":{"ins":[["payload","mapping"]],"outs":[["json_path","file_path"]]}};
const paletteEl = document.getElementById('palette');
PALETTE.forEach(mod=>{
  mod.groups.forEach(g=>{
    const gid = mod.mod+'-'+g.cat;
    const head = document.createElement('div');
    head.className='pal-group-head';
    head.innerHTML = `<svg class="chevron" width="8" height="8" viewBox="0 0 8 8"><path d="M2 1l4 3-4 3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg><span class="chip-dot" style="background:${mod.color}"></span>${mod.label} / ${g.cat}<span class="count">${g.ops.length}</span>`;
    const body = document.createElement('div');
    body.id = 'g-'+gid;
    body.classList.add('hidden'); // collapsed by default on load
    g.ops.forEach(([full,short,unavailable])=>{
      const it = document.createElement('div');
      it.className = 'pal-item'+(unavailable?' unavailable':'');
      it.innerHTML = `<span class="pal-name">${short}</span><span class="op">${full}</span><span class="add-hint">+ add</span>`;
      if(!unavailable){
        it.title = 'Click to add '+full+' to the canvas';
        it.addEventListener('click', ()=> addNodeFromPalette(full, short, mod.mod));
      }
      body.appendChild(it);
    });
    head.addEventListener('click', ()=>{
      body.classList.toggle('hidden');
      head.classList.toggle('open', !body.classList.contains('hidden'));
    });
    const wrap = document.createElement('div'); wrap.className='pal-group';
    wrap.appendChild(head); wrap.appendChild(body);
    paletteEl.appendChild(wrap);
  });
});

// ---- node graph: real steps from examples/multimodal_full/multimodal-full.pipeline.yaml ----
// Edge colour = artifact-type FAMILY, deliberately independent of the modality
// accent colours (blue/orange/purple/pink) used for node borders and the
// palette. On this particular graph the "signal-like" family happens to only
// ever appear on EIT/EMG/vent-flavoured signal types, so it can look like
// colour-by-modality by coincidence — it isn't: index/event, mask, result and
// generic/session types cut across all three modalities and always render in
// their own family colour regardless of which modality produced them.
const TYPE_COLOR = {
  // signal-like: continuous/raw signal data
  eit_pixel_signal:'var(--type-signal)', eit_global_impedance:'var(--type-signal)', eit_sequence:'var(--type-signal)',
  eit_sparse_data:'var(--type-signal)', signal:'var(--type-signal)', signal_array:'var(--type-signal)',
  signal_list:'var(--type-signal)', emg_recording:'var(--type-signal)', boolean_array:'var(--type-signal)',
  // index/event-like: detected events, indices, detectors, quality flags
  index_array:'var(--type-index)', interval_collection:'var(--type-index)', breath_event_list:'var(--type-index)',
  eit_breath_detector:'var(--type-index)', eit_rate_detector:'var(--type-index)', quality_flag_list:'var(--type-index)',
  // mask-like: spatial ROI selections
  roi_mask:'var(--type-mask)',
  // result-like / scalar: computed numeric outputs
  scalar_metric:'var(--type-result)', array:'var(--type-result)', parameter_result:'var(--type-result)',
  parameter_result_list:'var(--type-result)', diagnostic_summary:'var(--type-result)',
  // bundle: multi-signal containers (kept in the signal family — they carry signals, just several at once)
  emg_processed_bundle:'var(--type-signal)', ventilator_channel_bundle:'var(--type-signal)', ventilator_recording:'var(--type-signal)',
  // a saved Prepare sequence: a synchronized multi-modal crop a load step can pull its own channel from
  sequence_bundle:'var(--type-signal)',
  // generic/session/path: plumbing, never scientific data itself
  sync_summary:'var(--generic)', m3session:'var(--generic)', directory_path:'var(--generic)',
  file_path:'var(--generic)', mapping:'var(--generic)',
};
function tc(t){return TYPE_COLOR[t] || 'var(--text-faint)';}

const NODES = [
  {id:'load_eit', op:'eit.load', mod:'eit', x:30, y:40, ins:[['sequence','sequence_bundle']], outs:[['raw_eit','eit_pixel_signal']], status:'ok'},
  {id:'load_emg', op:'emg.load', mod:'emg', x:30, y:320, ins:[['sequence','sequence_bundle']], outs:[['emg_recording','emg_recording']], status:'ok'},
  {id:'load_ventilator', op:'ventilator.load', mod:'vent', x:30, y:600, ins:[['sequence','sequence_bundle']], outs:[['ventilator_raw','ventilator_recording']], status:'ok'},

  {id:'mdn_filter', op:'eit.mdn_filter', mod:'eit', x:626, y:40, ins:[['signal','eit_pixel_signal']], outs:[['filtered_eit','eit_pixel_signal']], status:'ok'},
  {id:'global_impedance', op:'eit.global_impedance', mod:'eit', x:924, y:40, ins:[['signal','eit_pixel_signal']], outs:[['global_impedance','eit_global_impedance']], status:'ok'},
  {id:'detect_breaths_eit', op:'eit.detect_breaths', mod:'eit', x:1222, y:40, ins:[['signal','eit_global_impedance']], outs:[['breath_intervals','interval_collection']], status:'ok'},
  {id:'continuous_tiv', op:'eit.continuous_tiv', mod:'eit', x:1520, y:40, ins:[['signal','eit_global_impedance']], outs:[['continuous_tiv','eit_sparse_data']], status:'pending'},

  {id:'emg_preprocess', op:'emg.preprocess', mod:'emg', x:626, y:320, ins:[['session','m3session']], outs:[['processed_emg','emg_processed_bundle']], status:'ok'},
  {id:'emg_ecg_gating', op:'emg.ecg_gating', mod:'emg', x:924, y:320, ins:[['processed_emg','emg_processed_bundle']], outs:[['ecg_gated_emg','signal_array']], status:'warn'},
  {id:'emg_detect_breaths', op:'emg.detect_breaths', mod:'emg', x:1222, y:320, ins:[['session','m3session']], outs:[['emg_breath_events','breath_event_list']], status:'ok'},
  {id:'emg_time_product', op:'emg.time_product', mod:'emg', x:1520, y:320, ins:[['processed_emg','emg_processed_bundle']], outs:[['time_product','array']], status:'pending'},

  {id:'vent_channels', op:'ventilator.channels', mod:'vent', x:626, y:600, ins:[['ventilator_raw','ventilator_recording']], outs:[['ventilator_signals','ventilator_channel_bundle']], status:'ok'},
  {id:'vent_detect_breaths', op:'ventilator.detect_breaths', mod:'vent', x:924, y:600, ins:[['ventilator_signals','ventilator_channel_bundle']], outs:[['ventilator_breath_indices','index_array']], status:'ok'},
  {id:'vent_respiratory_rate', op:'ventilator.respiratory_rate', mod:'vent', x:1222, y:600, ins:[['ventilator_breath_indices','index_array']], outs:[['ventilator_respiratory_rate','scalar_metric']], status:'pending'},

  {id:'export_session_summary', op:'export.session_summary', mod:'generic', x:1818, y:320, ins:[['session','m3session']], outs:[['output_dir','directory_path']], status:'pending'},
];

const EDGES = [
  // load_* -> first processing step: an explicit named binding (source and
  // destination port names differ per the real step registry, but the type
  // matches — eit_pixel_signal / emg_recording / ventilator_recording — so
  // this is a real drawn wire, not an implicit session dependency).
  {src:'load_eit', dst:'mdn_filter', key:'signal', srcKey:'raw_eit'},
  {src:'mdn_filter', dst:'global_impedance', key:'filtered_eit'},
  {src:'global_impedance', dst:'detect_breaths_eit', key:'global_impedance'},
  {src:'detect_breaths_eit', dst:'continuous_tiv', key:'breath_intervals'},
  {src:'load_emg', dst:'emg_preprocess', key:'session', srcKey:'emg_recording'},
  {src:'emg_preprocess', dst:'emg_ecg_gating', key:'processed_emg'},
  {src:'emg_ecg_gating', dst:'emg_detect_breaths', key:'ecg_gated_emg', session:true},
  {src:'emg_detect_breaths', dst:'emg_time_product', key:'emg_breath_events'},
  {src:'load_ventilator', dst:'vent_channels', key:'ventilator_raw'},
  {src:'vent_channels', dst:'vent_detect_breaths', key:'ventilator_signals'},
  {src:'vent_detect_breaths', dst:'vent_respiratory_rate', key:'ventilator_breath_indices'},
  {src:'continuous_tiv', dst:'export_session_summary', key:'session', session:true},
  {src:'emg_time_product', dst:'export_session_summary', key:'session', session:true},
  {src:'vent_respiratory_rate', dst:'export_session_summary', key:'session', session:true},
];

// Snapshot of the starting graph, taken before anything on the canvas can be
// touched, so "Auto layout" has something real to restore rather than just
// re-arranging whatever happens to be there.
const DEFAULT_NODES = JSON.parse(JSON.stringify(NODES));
const DEFAULT_EDGES = JSON.parse(JSON.stringify(EDGES));

const NODE_W = 208, PORT_H = 18, HEAD_H = 44;
function nodeById(id){ return NODES.find(n=>n.id===id); }
function nodeHeight(n){ return HEAD_H + Math.max(n.ins.length, n.outs.length, 1) * PORT_H + 10; }
function portY(n, idx){
  const rowsStart = HEAD_H + 8;
  return n.y + rowsStart + idx*PORT_H + PORT_H/2;
}
function updateStepCountPill(){
  const pill=document.getElementById('stepCountPill');
  if(pill) pill.textContent = `${NODES.length} / 45 steps shown · multimodal-full.pipeline.yaml`;
}
function updateCanvasSize(){
  const maxX = Math.max(...NODES.map(n=>n.x+NODE_W), 900);
  const maxY = Math.max(...NODES.map(n=>n.y+nodeHeight(n)), 480);
  inner.style.width = (maxX+60)+'px';
  inner.style.height = (maxY+40)+'px';
}

const inner = document.getElementById('canvasInner');
const svg = document.getElementById('edgesSvg');

