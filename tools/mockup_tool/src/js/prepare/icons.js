// ---- small line icons for 1 · Prepare ----
// Same drawing rules as the step icons on 2 · Design: a 20 × 20 grid, round
// line ends, stroke width 1.6, drawn in the text colour of the parent.
// An element with data-icon="name" gets the icon put in front of its text.
const PREP_ICONS = {
  folder:  'M3 6v9a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8a1 1 0 0 0-1-1H9.5L8 5H4a1 1 0 0 0-1 1z',
  start:   'M5 4v12M15 4.5l-7 5.5 7 5.5z',
  back:    'M12.5 4.5L7 10l5.5 5.5',
  play:    'M6.5 4.5l9 5.5-9 5.5z',
  pause:   'M7.5 4.5v11M12.5 4.5v11',
  fwd:     'M7.5 4.5L13 10l-5.5 5.5',
  end:     'M15 4v12M5 4.5l7 5.5-7 5.5z',
  plus:    'M10 4.5v11M4.5 10h11',
  minus:   'M4.5 10h11',
  reset:   'M4.5 10a5.5 5.5 0 1 0 1.7-4M4.5 4.5V8H8',
  close:   'M6 6l8 8M14 6l-8 8',
  help:    'M10 3a7 7 0 1 0 0 14a7 7 0 1 0 0-14M8 8.2a2 2 0 1 1 2.7 1.9c-.5.2-.7.6-.7 1.1v.4M10 14v.1',
  roi:     'M10 3a7 7 0 1 0 0 14a7 7 0 1 0 0-14M3 10h14M10 3v14',
  sliders: 'M4 6.5h12M4 13.5h12M8 4.5v4M12.5 11.5v4',
  whole:   'M3 10h14M6 7l-3 3 3 3M14 7l3 3-3 3',
  check:   'M5 10.5l3 3 7-7',
  seq:     'M3 5h14v10H3zM7.5 5v10M12.5 5v10',
  dot:     'M10 8.5a1.5 1.5 0 1 0 0 3a1.5 1.5 0 1 0 0-3',
  'chevron-down': 'M6 8l4 4 4-4',
};
function prepIcon(name, size=14){
  return `<svg width="${size}" height="${size}" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="${PREP_ICONS[name]}" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
function placeIcons(root){
  root.querySelectorAll('[data-icon]').forEach(el=>{
    if(el.querySelector(':scope > svg[data-placed]')) return;
    el.insertAdjacentHTML('afterbegin', prepIcon(el.dataset.icon, +(el.dataset.iconSize || 14)).replace('<svg ', '<svg data-placed ' ));
  });
}
placeIcons(document.getElementById('view-prepare'));
