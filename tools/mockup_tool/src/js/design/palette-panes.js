// ---- parts of the two side panels: Steps and Available data on the left;
// Inspector, Edge legend and Help on the right. A click on a part head
// closes or opens the part. The bar under an open part changes its size
// against the next open part below it, and a double-click on the bar makes
// the two equal. The open parts share the height of the panel by their
// --grow values, so one open part alone fills the panel. ----
const SIDE_PANELS = [document.getElementById('palettePanel'), document.getElementById('inspector')];
const PANE_MIN_PX = 70;   // smallest height of an open part while a bar is dragged

document.querySelectorAll('.pal-pane-chevron').forEach(el=>{ el.innerHTML = PAL_CHEVRON_SVG; });
document.querySelectorAll('.pal-pane-head').forEach(head=> head.setAttribute('aria-expanded', 'true'));

function paneIsOpen(pane){ return !pane.classList.contains('collapsed'); }
function paneGrow(pane){ return parseFloat(getComputedStyle(pane).getPropertyValue('--grow')) || 50; }
// the two parts a bar resizes: the part just above it, and the nearest open
// part below it
function splitterPanes(bar){
  const above = bar.previousElementSibling;
  let below = bar.nextElementSibling;
  while(below && !(below.classList.contains('pal-pane') && paneIsOpen(below))) below = below.nextElementSibling;
  return {above, below};
}
// a bar shows only when there is an open part on both sides of it
function updateSplitters(panel){
  panel.querySelectorAll(':scope > .pal-splitter').forEach(bar=>{
    const {above, below} = splitterPanes(bar);
    bar.classList.toggle('hidden', !(above && paneIsOpen(above) && below));
  });
}

SIDE_PANELS.forEach(panel=>{
  panel.addEventListener('click', e=>{
    const head = e.target.closest('.pal-pane-head');
    if(!head) return;
    const pane = head.closest('.pal-pane');
    const closed = pane.classList.toggle('collapsed');
    head.setAttribute('aria-expanded', String(!closed));
    updateSplitters(panel);
  });

  panel.addEventListener('mousedown', e=>{
    const bar = e.target.closest('.pal-splitter');
    if(!bar || e.button!==0) return;
    e.preventDefault();
    const {above, below} = splitterPanes(bar);
    const top = above.getBoundingClientRect().top;
    const height = above.offsetHeight + below.offsetHeight;
    const grow = paneGrow(above) + paneGrow(below);
    bar.classList.add('dragging');
    document.body.classList.add('pal-resizing');
    const move = ev=>{
      const h = Math.min(height-PANE_MIN_PX, Math.max(PANE_MIN_PX, ev.clientY - top));
      above.style.setProperty('--grow', grow*h/height);
      below.style.setProperty('--grow', grow*(height-h)/height);
    };
    const up = ()=>{
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
      bar.classList.remove('dragging');
      document.body.classList.remove('pal-resizing');
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  });

  panel.addEventListener('dblclick', e=>{
    const bar = e.target.closest('.pal-splitter');
    if(!bar) return;
    const {above, below} = splitterPanes(bar);
    const half = (paneGrow(above) + paneGrow(below)) / 2;
    above.style.setProperty('--grow', half);
    below.style.setProperty('--grow', half);
  });

  updateSplitters(panel);
});

// the help box shows what a palette part holds while the mouse is on its head
document.getElementById('palettePanel').addEventListener('mouseover', e=>{
  const head = e.target.closest('.pal-pane-head');
  if(!head) return;
  const pane = head.closest('.pal-pane').dataset.pane;
  if(pane==='data') showHelpIdle('data');
  else if(pane==='steps') showHelpIdle(document.querySelector('.pal-tab.active').dataset.paltab);
});
