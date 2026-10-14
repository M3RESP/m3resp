// ---- canvas pan: hold Space and drag with the left mouse button to move the
// view. The drag changes the scroll position of the canvas, so the steps keep
// their places. While Space is down, a press anywhere on the canvas (on a
// step, a dot, a link, a note) starts the pan and nothing else. ----
let spaceHeld = false;

function isTypingTarget(el){
  return !!(el && el.closest && el.closest('input, textarea, select, [contenteditable], .annot-text'));
}
function setSpaceHeld(on){
  spaceHeld = on;
  canvasScrollEl.classList.toggle('pan-ready', on);
}

document.addEventListener('keydown', e=>{
  if(e.code!=='Space' || !document.getElementById('view-design').classList.contains('active')) return;
  if(isTypingTarget(e.target)) return;
  // stops the page from scrolling and a focused button from clicking
  e.preventDefault();
  if(!spaceHeld) setSpaceHeld(true);
});
document.addEventListener('keyup', e=>{
  if(e.code!=='Space' || !spaceHeld) return;
  e.preventDefault();
  setSpaceHeld(false);
});
window.addEventListener('blur', ()=>{ if(spaceHeld) setSpaceHeld(false); });

// the capture phase runs this before the step, dot, note and box-select presses
canvasScrollEl.addEventListener('mousedown', e=>{
  if(!spaceHeld || e.button!==0 || e.target.closest('.canvas-hud')) return;
  e.preventDefault();
  e.stopPropagation();
  const startX = e.clientX, startY = e.clientY;
  const startLeft = canvasScrollEl.scrollLeft, startTop = canvasScrollEl.scrollTop;
  canvasScrollEl.classList.add('panning');
  const move = ev=>{
    canvasScrollEl.scrollLeft = startLeft-(ev.clientX-startX);
    canvasScrollEl.scrollTop = startTop-(ev.clientY-startY);
  };
  const up = ()=>{
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    canvasScrollEl.classList.remove('panning');
    // the release also fires a click on the step or link under the mouse;
    // this stops that click from selecting the step or removing the link
    // (a release outside the canvas fires no click, so the stop ends after one tick)
    const stopClick = ev=>{ ev.stopPropagation(); ev.preventDefault(); };
    window.addEventListener('click', stopClick, {capture:true, once:true});
    setTimeout(()=> window.removeEventListener('click', stopClick, true), 0);
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}, true);
// two quick pans with Space down would count as a double-click and open the quick menu
canvasScrollEl.addEventListener('dblclick', e=>{
  if(spaceHeld){ e.stopPropagation(); e.preventDefault(); }
}, true);
