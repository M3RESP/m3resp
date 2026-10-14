// ================= 3 · Check: the quality checks of the run =================
// One entry per quality check of the multimodal-full run. The pass counts
// (35 of 96) are those of the run. The values of single breaths and
// occlusions are example values that agree with these counts.
// The fields follow m3resp.data.quality.QualityFlag: one item per flag
// (breath_id, start_time, end_time, value, threshold, passed, message).
//   scope   what one flag covers: 'breath', 'occlusion', 'pair' (an EMG
//           breath with its ventilator breath) or 'run' (one flag)
//   limit   threshold of the rule, with dir 'max' (value ≤ limit passes) or
//           'min' (value ≥ limit passes); limitText when the step sets it
//           from the data

// example random numbers, the same on every load
function qcRand(k){ const s=Math.sin(k*91.7+3.1)*43758.5453; return s-Math.floor(s); }

// EMG breath i of the recording (s)
const qcEmgBreath = i=> ({t0: 0.9 + i*3.12, t1: 0.9 + i*3.12 + 1.35});

// items for a per-breath check: `fails` lists the breath numbers (0-based)
// that failed; value(i, passed) gives the example value
function qcBreathItems(n, fails, value, message){
  return Array.from({length:n}, (_, i)=>{
    const passed = !fails.includes(i), b = qcEmgBreath(i);
    const v = value(i, passed);
    return {id:String(i+1), t0:b.t0, t1:b.t1, value:v, passed, message: passed ? '' : message(v, i)};
  });
}
const ALL19 = Array.from({length:19}, (_, i)=> i);

const QC_CHECKS = [
  // ---------------- ventilator ----------------
  {mod:'vent', code:'pocc_quality', step:'ventilator.pocc_quality', stepMs:0.7, severity:'info', scope:'occlusion',
   name:'Occlusion upslope', signal:'P_aw',
   what:'Checks the pressure upslope of each airway occlusion (Pocc) manoeuvre (Warnaar et al. 2024).',
   rule:'Passes when the normalized pressure rise at 90 % of the upslope is at least 0.80, the rise at 90 % is at least 2 and the rise at 10 % is at least 0.',
   params:[['dp_up_90_norm_threshold','0.8'], ['dp_up_90_threshold','2'], ['dp_up_10_threshold','0']],
   valueName:'normalized rise at 90 %', unit:'', limit:0.8, dir:'min', digits:2,
   items:[
     {id:'Pocc 1', t0:40.2, t1:41.4, value:0.91, passed:true, message:''},
     {id:'Pocc 2', t0:43.3, t1:44.5, value:0.64, passed:false, message:'Normalized rise at 90 % is 0.64, under 0.80.'},
   ]},
  {mod:'vent', code:'detect_non_consecutive_manoeuvres', step:'ventilator.detect_non_consecutive_manoeuvres', stepMs:0.4, severity:'info', scope:'occlusion',
   name:'Occlusions apart', signal:'P_aw',
   what:'Checks the occlusion manoeuvres against the ventilator breaths: two occlusions need normal breaths between them.',
   rule:'Passes when at least 1 normal ventilator breath comes before the occlusion.',
   params:[],
   valueName:'normal breaths before', unit:'breaths', limit:1, dir:'min', digits:0,
   items:[
     {id:'Pocc 1', t0:40.2, t1:41.4, value:4, passed:true, message:''},
     {id:'Pocc 2', t0:43.3, t1:44.5, value:0, passed:false, message:'Pocc 2 comes directly after Pocc 1.'},
   ]},

  // ---------------- EMG ----------------
  {mod:'emg', code:'evaluate_event_timing_unmatched', step:'emg.evaluate_event_timing', stepMs:0.7, severity:'warning', scope:'run',
   name:'Unpaired breaths', signal:'EMGdi',
   what:'Counts the EMG and ventilator breaths that found no partner. Their timing is not checked.',
   rule:'Passes when every breath has a partner.',
   params:[],
   valueName:'breaths without a partner', unit:'breaths', limit:0, dir:'max', digits:0,
   items:[{id:'run', value:6, passed:false, message:'6 event(s) had no paired counterpart and were not assessed.'}]},
  {mod:'emg', code:'percentage_under_baseline', step:'emg.percentage_under_baseline', stepMs:22.9, severity:'info', scope:'breath',
   name:'Time under baseline', signal:'EMGdi',
   what:'Computes the part of each EMG breath that lies under the baseline.',
   rule:'Passes when at most 40 % of the breath lies under the baseline.',
   params:[['aub_threshold','40 %'], ['aub_window_seconds','the breath (onset to offset)']],
   valueName:'under baseline', unit:'%', limit:40, dir:'max', digits:1,
   items: qcBreathItems(19, [6, 14], (i, ok)=> ok ? 8 + 26*qcRand(i) : (i===6 ? 46.3 : 52.8),
     v=> `${v.toFixed(1)} % of the breath under the baseline, over 40 %.`)},
  {mod:'emg', code:'evaluate_event_timing', step:'emg.evaluate_event_timing', stepMs:0.7, severity:'info', scope:'pair',
   name:'Timing against ventilator', signal:'EMGdi',
   what:'Scores the timing agreement of each EMG breath with its ventilator breath.',
   rule:'Passes when the EMG breath and its ventilator breath start close together.',
   params:[],
   valueName:'start difference', unit:'s', limit:null, limitText:'set in the step', dir:'max', digits:2,
   items: [0,1,2,4,5,7,8,9,11,12,14,16,17].map((bi, k)=>{
     const b = qcEmgBreath(bi), passed = k!==8;
     const v = passed ? 0.04 + 0.26*qcRand(bi+40) : 0.71;
     return {id:`${bi+1}`, t0:b.t0, t1:b.t1, value:v, passed, message: passed ? '' : 'EMG breath starts 0.71 s from its ventilator breath.'};
   })},
  {mod:'emg', code:'evaluate_bell_curve_error', step:'emg.evaluate_bell_curve_error', stepMs:4322.3, severity:'info', scope:'breath',
   name:'Bell-curve shape', signal:'EMGdi',
   what:'Scores how well each EMG breath matches a bell-curve shape.',
   rule:'Passes when the shape error is under the limit of the step.',
   params:[],
   valueName:'shape error', unit:'', limit:null, limitText:'set in the step', dir:'max', digits:2,
   items: qcBreathItems(19, ALL19.filter(i=> i!==3 && i!==11), (i, ok)=> ok ? 0.08 + 0.04*qcRand(i+7) : 0.21 + 0.34*qcRand(i+9),
     v=> `Shape error ${v.toFixed(2)}.`)},
  {mod:'emg', code:'detect_local_high_aub', step:'emg.detect_local_high_aub', stepMs:0.6, severity:'info', scope:'breath',
   name:'Local high area under baseline', signal:'EMGdi',
   what:'Flags EMG breaths whose area under the baseline is high against the other breaths.',
   rule:'Passes when the area under the baseline is at most 4 times its 75th percentile over all breaths.',
   params:[['threshold_percentile','75'], ['threshold_factor','4']],
   valueName:'area under baseline', unit:'µV·s', limit:null, limitText:'4 × 75th percentile', dir:'max', digits:2,
   items: qcBreathItems(19, ALL19, i=> 0.8 + 1.6*qcRand(i+21), ()=> 'Flagged by the step.')},
  {mod:'emg', code:'detect_extreme_time_products', step:'emg.detect_extreme_time_products', stepMs:0.6, severity:'info', scope:'breath',
   name:'Extreme time-product', signal:'EMGdi',
   what:'Flags EMG breaths whose time-product lies outside bounds taken from all breaths.',
   rule:'Passes when the time-product lies between 0.1 times its 5th and 10 times its 95th percentile.',
   params:[['lower_percentile','5'], ['lower_factor','0.1'], ['upper_percentile','95'], ['upper_factor','10']],
   valueName:'time-product', unit:'µV·s', limit:null, limitText:'0.1 × P5 to 10 × P95', dir:'max', digits:2,
   items: qcBreathItems(19, ALL19, i=> 4 + 5*qcRand(i+33), ()=> 'Flagged by the step.')},
  {mod:'emg', code:'evaluate_respiratory_rates', step:'emg.evaluate_respiratory_rates', stepMs:0.5, severity:'info', scope:'run',
   name:'Respiratory rate agreement', signal:'EMGdi',
   what:'Compares the breaths found in the EMG with the breaths the ventilator rate expects.',
   rule:'Passes when at least 0.10 of the expected EMG breaths are found.',
   params:[['minimum_fraction','0.1']],
   valueName:'fraction of expected breaths', unit:'', limit:0.1, dir:'min', digits:2,
   items:[{id:'run', value:0.95, passed:true, message:''}]},
  {mod:'emg', code:'interpeak_dist', step:'emg.interpeak_dist', stepMs:0.7, severity:'info', scope:'run',
   name:'ECG and EMG peak spacing', signal:'EMGdi',
   what:'Compares the median distance between EMG peaks with that between ECG peaks (Warnaar et al. 2024).',
   rule:'Passes when the EMG / ECG ratio of the median peak distances is at most 1.10.',
   params:[['threshold','1.1']],
   valueName:'EMG / ECG ratio', unit:'', limit:1.1, dir:'max', digits:2,
   items:[{id:'run', value:1.04, passed:true, message:''}]},
];

const QC_MODS = [
  {id:'eit',  name:'EIT',        card:'checkEit',  prepCard:'cardEit',
   empty:'This pipeline runs no EIT quality checks.'},
  {id:'emg',  name:'EMG',        card:'checkEmg',  prepCard:'cardEmg'},
  {id:'vent', name:'Ventilator', card:'checkVent', prepCard:'cardVent'},
];
