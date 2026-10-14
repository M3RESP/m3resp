// ---- parts of the side panels: Steps and Available data on the left;
// Inspector, Edge legend and Help on the right; Recording, Steps and Saved
// sequences on 1 · Prepare; Run, Report and Legend on 3 · Check.
// A click on a part head closes or opens the part. Open parts sit at the top
// in their first order; closed parts go to the bottom of the panel. A bar
// between two open parts changes their sizes (a double-click makes the two
// equal). The open parts share the height of the panel by their --grow
// values, so one open part alone fills the panel. ----
const SIDE_PANELS = [document.getElementById('palettePanel'), document.getElementById('inspector'), document.getElementById('prepSidebar'), document.getElementById('checkSidebar')];
const PANE_MIN_PX = 70;   // smallest height of an open part while a bar is dragged
const PANE_REOPEN_GROW = 160;   // size of a part that was closed from the start, when it opens

document.querySelectorAll('.pal-pane-chevron').forEach(el=>{ el.innerHTML = PAL_CHEVRON_SVG; });
document.querySelectorAll('.pal-pane-head').forEach(head=> head.setAttribute('aria-expanded', 'true'));

function paneIsOpen(pane){ return !pane.classList.contains('collapsed'); }
function paneGrow(pane){ return parseFloat(getComputedStyle(pane).getPropertyValue('--grow')) || 50; }
const panesOf = panel=> [...panel.querySelectorAll(':scope > .pal-pane')];

// open parts first, in their first order, then the closed parts; one bar
// between each two open parts
function arrangePanes(panel){
  const items = [...panel.children].filter(el=> el.matches('.pal-pane, .pal-splitter'));
  if(!items.length) return;
  const ref = items[items.length-1].nextSibling;
  items.filter(el=> el.matches('.pal-splitter')).forEach(el=> el.remove());
  const panes = panesOf(panel).sort((x, y)=> x.dataset.paneOrder - y.dataset.paneOrder);
  const open = panes.filter(paneIsOpen), closed = panes.filter(p=> !paneIsOpen(p));
  [...open, ...closed].forEach((pane, i)=>{
    if(i>0 && i<open.length){
      const bar = document.createElement('div');
      bar.className = 'pal-splitter';
      bar.title = 'Drag to change the size of the panels; double-click to make them equal';
      panel.insertBefore(bar, ref);
    }
    panel.insertBefore(pane, ref);
  });
}
// the first change of a panel: every open part keeps the height it has on
// screen (parts marked .fit take the height of their content until then)
function fixPaneSizes(panel){
  if(panel.dataset.sized || !panel.offsetHeight) return;
  panesOf(panel).forEach(p=>{
    if(paneIsOpen(p)) p.style.setProperty('--grow', p.offsetHeight);
    else if(p.classList.contains('fit')) p.style.setProperty('--grow', PANE_REOPEN_GROW);
    p.classList.remove('fit');
  });
  panel.dataset.sized = '1';
}
// the two parts a bar resizes: the parts just above and just below it
function splitterPanes(bar){
  return {above: bar.previousElementSibling, below: bar.nextElementSibling};
}

SIDE_PANELS.forEach(panel=>{
  panesOf(panel).forEach((p, i)=> p.dataset.paneOrder = i);

  panel.addEventListener('click', e=>{
    const head = e.target.closest('.pal-pane-head');
    if(!head) return;
    fixPaneSizes(panel);
    const pane = head.closest('.pal-pane');
    const closed = pane.classList.toggle('collapsed');
    head.setAttribute('aria-expanded', String(!closed));
    arrangePanes(panel);
  });

  panel.addEventListener('mousedown', e=>{
    const bar = e.target.closest('.pal-splitter');
    if(!bar || e.button!==0) return;
    e.preventDefault();
    fixPaneSizes(panel);
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
    fixPaneSizes(panel);
    const {above, below} = splitterPanes(bar);
    const half = (paneGrow(above) + paneGrow(below)) / 2;
    above.style.setProperty('--grow', half);
    below.style.setProperty('--grow', half);
  });

  arrangePanes(panel);
});

// the help box shows what a palette part holds while the mouse is on its head
document.getElementById('palettePanel').addEventListener('mouseover', e=>{
  const head = e.target.closest('.pal-pane-head');
  if(!head) return;
  const pane = head.closest('.pal-pane').dataset.pane;
  if(pane==='data') showHelpIdle('data');
  else if(pane==='steps') showHelpIdle(document.querySelector('.pal-tab.active').dataset.paltab);
});
