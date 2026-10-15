// ---- shared parts of the three palette lists (Operations, Workflows,
// Available data): sections that open one at a time, a search box that opens every
// section with a match, and a switch between icons and a list ----
const PAL_CHEVRON_SVG = '<svg class="chevron" width="8" height="8" viewBox="0 0 8 8"><path d="M2 1l4 3-4 3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
// sec: {key, label, dot (a CSS colour), count, open, body}
function paletteSectionHtml(sec){
  return `<div class="tb-mod${sec.open?' open':''}">
    <div class="pal-group-head${sec.open?' open':''}" data-sec="${sec.key}">
      ${PAL_CHEVRON_SVG}<span class="chip-dot" style="background:${sec.dot}"></span>${sec.label}<span class="count">${sec.count}</span>
    </div>
    ${sec.open ? `<div class="tb-mod-body">${sec.body}</div>` : ''}
  </div>`;
}
// the open section after a click on the head of section `key`
function nextOpenSection(current, key){ return current===key ? null : key; }
function wireViewSeg(segId, onChange){
  document.getElementById(segId).addEventListener('click', e=>{
    const btn = e.target.closest('button[data-palview]');
    if(!btn) return;
    document.querySelectorAll(`#${segId} button`).forEach(b=>b.classList.toggle('on', b===btn));
    onChange(btn.dataset.palview);
  });
}
// one small icon per family of data types, drawn in the type colour
const TYPE_ICONS = {
  'var(--type-signal)': 'M2 10c2-5 3.5-5 5 0s3 5 5 0 3.5-5 6 0',
  'var(--type-index)':  'M4 4v12M8.5 6v10M13 4v12M17 8v8',
  'var(--type-mask)':   'M4 4h12v12H4zM4 10h12M10 4v12',
  'var(--type-result)': 'M15 4H5l5 6-5 6h10',
};
function typeIconSvg(type, size){
  const d = TYPE_ICONS[tc(type)] || NODE_ICONS.other;
  return `<svg width="${size}" height="${size}" viewBox="0 0 20 20" fill="none" aria-hidden="true">
    <path d="${d}" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
