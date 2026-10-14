// ---- one time format for 1 · Prepare and 2 · Design ----
// A time is in seconds with the unit: "4.0 s". A range is "4.0–22.0 s" and a
// length is "18.0 s". Use 2 decimals only where the step is finer than 0.1 s
// (the EIT cursor, a zoom range below 2 s).
function fmtSec(sec, digits=1){ return `${(+sec).toFixed(digits)} s`; }
function fmtRange(a, b, digits=1){ return `${(+a).toFixed(digits)}–${(+b).toFixed(digits)} s`; }
function fmtDuration(sec, digits=1){ return fmtSec(sec, digits); }
