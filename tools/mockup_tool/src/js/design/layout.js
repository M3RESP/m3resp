// ---- space on the canvas: no two steps on top of each other ----
// A new step goes to the nearest free place. An opened workflow, or a step
// dropped on top of other steps, pushes those steps out of its way. Every
// move of this kind is animated, so the user can follow it.
//
// The layout works on "blocks": a single step, a closed workflow box, or an
// opened workflow with its frame (its steps move together).
const LAYOUT_GAP = 26;          // free space kept between two blocks
const LAYOUT_ANIM_MS = 260;

// frame of an opened workflow, in canvas coordinates
function groupFrameRect(g){
  const members = g.members.map(id=>NODES.find(x=>x.id===id)).filter(Boolean);
  if(!members.length) return null;
  const PAD = NODE_VIEW==='compact' ? 16 : 26, TOP = NODE_VIEW==='compact' ? 30 : 36;
  const x = Math.min(...members.map(m=>m.x)) - PAD;
  const y = Math.min(...members.map(m=>m.y)) - PAD - TOP;
  return {
    x, y,
    w: Math.max(...members.map(m=>m.x+nodeWidth(m))) + PAD - x,
    h: Math.max(...members.map(m=>m.y+nodeHeight(m))) + PAD - y,
  };
}

function layoutBlocks(){
  const blocks = [];
  GROUPS.forEach(g=>{
    if(g.collapsed){
      blocks.push({id:g.id, g, rect:{x:g.x, y:g.y, w:nodeWidth({}), h:nodeHeight(groupVirtualNode(g))}});
    } else {
      const r = groupFrameRect(g);
      if(r) blocks.push({id:g.id, g, rect:r});
    }
  });
  NODES.forEach(n=>{
    if(!groupOfNode(n.id)) blocks.push({id:n.id, n, rect:{x:n.x, y:n.y, w:nodeWidth(n), h:nodeHeight(n)}});
  });
  return blocks;
}
// the block a step belongs to: its workflow when it is in one, else itself
function blockIdOf(nodeId){
  const g = groupOfNode(nodeId);
  return g ? g.id : nodeId;
}
function rectsOverlap(a, b, gap){
  return a.x < b.x+b.w+gap && b.x < a.x+a.w+gap && a.y < b.y+b.h+gap && b.y < a.y+a.h+gap;
}
function moveBlock(b, dx, dy){
  b.rect.x += dx; b.rect.y += dy;
  if(b.n){ b.n.x += dx; b.n.y += dy; return; }
  // a workflow moves with all its steps, also the hidden ones of a closed box
  b.g.x += dx; b.g.y += dy;
  b.g.members.forEach(id=>{ const m = NODES.find(x=>x.id===id); if(m){ m.x += dx; m.y += dy; } });
}

// Push every block that overlaps one of `anchorIds` out of the way; a pushed
// block can push further blocks. Each block takes the shortest of the four
// moves (right, down, left, up) that does not end on a block that already
// stays in place.
function pushApart(anchorIds){
  const blocks = layoutBlocks();
  const fixed = new Set(anchorIds);
  const queue = blocks.filter(b=>fixed.has(b.id));
  let guard = 0;
  while(queue.length && guard++ < 400){
    const a = queue.shift();
    fixed.add(a.id);
    blocks.forEach(b=>{
      if(fixed.has(b.id) || !rectsOverlap(a.rect, b.rect, LAYOUT_GAP)) return;
      const A = a.rect, B = b.rect, G = LAYOUT_GAP;
      const moves = [
        [A.x+A.w+G-B.x, 0], [0, A.y+A.h+G-B.y],
        [A.x-G-(B.x+B.w), 0], [0, A.y-G-(B.y+B.h)],
      ].filter(([dx, dy])=>{
        const r = {x:B.x+dx, y:B.y+dy, w:B.w, h:B.h};
        return r.x>=0 && r.y>=0 && !blocks.some(o=>o!==b && fixed.has(o.id) && rectsOverlap(r, o.rect, G));
      }).sort((m1, m2)=> Math.hypot(...m1)-Math.hypot(...m2));
      const [dx, dy] = moves[0] || [A.x+A.w+G-B.x, 0];
      moveBlock(b, dx, dy);
      if(!queue.includes(b)) queue.push(b);
    });
  }
}

// nearest top-left corner to (x, y) where a w × h box touches no block
function findFreeSpot(x, y, w, h, ignoreIds){
  const rects = layoutBlocks().filter(b=>!(ignoreIds||[]).includes(b.id)).map(b=>b.rect);
  const free = (px, py)=> px>=0 && py>=0 && !rects.some(r=>rectsOverlap({x:px, y:py, w, h}, r, LAYOUT_GAP));
  x = Math.max(0, Math.round(x)); y = Math.max(0, Math.round(y));
  if(free(x, y)) return {x, y};
  const STEP = 20;
  for(let ring=1; ring<=80; ring++){
    const spots = [];
    for(let i=-ring; i<=ring; i++){
      spots.push([x+i*STEP, y-ring*STEP], [x+i*STEP, y+ring*STEP], [x-ring*STEP, y+i*STEP], [x+ring*STEP, y+i*STEP]);
    }
    spots.sort((p, q)=> Math.hypot(p[0]-x, p[1]-y)-Math.hypot(q[0]-x, q[1]-y));
    const hit = spots.find(([px, py])=>free(px, py));
    if(hit) return {x:hit[0], y:hit[1]};
  }
  return {x, y};
}

// ---- animation ----
function snapshotLayout(){
  return {
    nodes: NODES.map(n=>[n, n.x, n.y]),
    groups: GROUPS.map(g=>[g, g.x, g.y]),
  };
}
function applyLayoutDom(){
  inner.querySelectorAll('.node').forEach(el=>{
    const n = nodeById(el.dataset.id);
    if(n){ el.style.left = n.x+'px'; el.style.top = n.y+'px'; }
  });
  inner.querySelectorAll('.group-frame').forEach(el=>{
    const r = groupFrameRect(groupById(el.dataset.group));
    if(r){ el.style.left = r.x+'px'; el.style.top = r.y+'px'; el.style.width = r.w+'px'; el.style.height = r.h+'px'; }
  });
  drawEdges();
}
let layoutAnim = null;       // {frameId, to} while nodes glide
function setLayout(snap){
  snap.nodes.forEach(([n, x, y])=>{ n.x = x; n.y = y; });
  snap.groups.forEach(([g, x, y])=>{ g.x = x; g.y = y; });
}
// Run `change` (which moves nodes in the data), then glide every node from
// where it was to where it is now; `onDone` (optional) runs at the end.
function animateLayout(change, onDone){
  // a glide still going on ends at once, so the new change starts from the
  // final places
  if(layoutAnim) finishLayoutAnim();
  const from = snapshotLayout();
  change();
  const to = snapshotLayout();
  const moved = from.nodes.some(([n, x, y], i)=> to.nodes[i] && (to.nodes[i][1]!==x || to.nodes[i][2]!==y));
  if(!moved){ updateCanvasSize(); if(onDone) onDone(); return; }
  const start = performance.now();
  const ease = t=> 1-Math.pow(1-t, 3);
  function frame(now){
    const t = Math.min(1, (now-start)/LAYOUT_ANIM_MS), k = ease(t);
    from.nodes.forEach(([n, x, y], i)=>{ n.x = x+(to.nodes[i][1]-x)*k; n.y = y+(to.nodes[i][2]-y)*k; });
    from.groups.forEach(([g, x, y], i)=>{ g.x = x+(to.groups[i][1]-x)*k; g.y = y+(to.groups[i][2]-y)*k; });
    applyLayoutDom();
    if(t<1) layoutAnim.frameId = requestAnimationFrame(frame);
    else finishLayoutAnim();
  }
  // the timer ends the glide also when the browser sends no more frames
  // (for example in a background tab)
  layoutAnim = {to, onDone, frameId: requestAnimationFrame(frame), timer: setTimeout(finishLayoutAnim, LAYOUT_ANIM_MS+60)};
}
// put every node at the end of the glide that is going on, and stop it
function finishLayoutAnim(){
  if(!layoutAnim) return;
  cancelAnimationFrame(layoutAnim.frameId);
  clearTimeout(layoutAnim.timer);
  setLayout(layoutAnim.to);
  const onDone = layoutAnim.onDone;
  layoutAnim = null;
  applyLayoutDom();
  updateCanvasSize();
  if(onDone) onDone();
}
// make room around these blocks, with animation
function makeRoomFor(blockIds){
  animateLayout(()=> pushApart(blockIds));
}
// move one block to the nearest free place, with animation
function slideToFreeSpot(blockId){
  animateLayout(()=>{
    const b = layoutBlocks().find(x=>x.id===blockId);
    if(!b) return;
    const spot = findFreeSpot(b.rect.x, b.rect.y, b.rect.w, b.rect.h, [blockId]);
    moveBlock(b, spot.x-b.rect.x, spot.y-b.rect.y);
  });
}
// push apart every overlap on the canvas, from the top left down; blocks
// earlier in that order stay in place
function resolveAllOverlaps(){
  const order = layoutBlocks().sort((a, b)=> a.rect.y-b.rect.y || a.rect.x-b.rect.x).map(b=>b.id);
  const done = [];
  order.forEach(id=>{ done.push(id); pushApart([...done]); });
}
