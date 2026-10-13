// ---- "Arrange": put the steps in columns by graph order ----
// The column of a block is the longest chain of links that leads to it, so
// data always flows from left to right. Each modality has its own lane (EIT,
// EMG, ventilator, other), like the rows of the default graph. Only the
// places change: the steps, links and settings stay the same.
const ARRANGE_LANES = ['eit', 'emg', 'vent', 'generic'];

function blockMod(b){ return (b.n || b.g).mod || 'generic'; }

function arrangeByDepth(){
  const blocks = layoutBlocks();
  if(!blocks.length) return;
  const ids = new Set(blocks.map(b=>b.id));
  // links between blocks (a link inside one workflow does not count)
  const links = [];
  EDGES.forEach(e=>{
    const a = blockIdOf(e.src), b = blockIdOf(e.dst);
    if(a!==b && ids.has(a) && ids.has(b)) links.push([a, b]);
  });

  // column = longest path from a block with no incoming link
  const depth = new Map(blocks.map(b=>[b.id, 0]));
  const waiting = new Map(blocks.map(b=>[b.id, 0]));
  links.forEach(([, b])=> waiting.set(b, waiting.get(b)+1));
  const ready = blocks.filter(b=>waiting.get(b.id)===0).map(b=>b.id);
  const seen = new Set();
  while(ready.length){
    const id = ready.shift();
    seen.add(id);
    links.filter(([a])=>a===id).forEach(([, b])=>{
      depth.set(b, Math.max(depth.get(b), depth.get(id)+1));
      waiting.set(b, waiting.get(b)-1);
      if(waiting.get(b)===0) ready.push(b);
    });
  }
  // blocks in a loop go one column after the deepest one
  const maxDepth = Math.max(...depth.values());
  blocks.forEach(b=>{ if(!seen.has(b.id)) depth.set(b.id, maxDepth+1); });

  // inside a column and lane: keep the old top-to-bottom order. An opened
  // workflow is wide, so it does not set the width of its column; it goes at
  // its column, under the other steps of its lane that it would cover.
  const cols = [...new Set(depth.values())].sort((a, b)=>a-b);
  const lane = b=> Math.max(0, ARRANGE_LANES.indexOf(blockMod(b)));
  const isFrame = b=> b.g && !b.g.collapsed;
  const cell = (c, l)=> blocks.filter(b=>depth.get(b.id)===c && lane(b)===l && !isFrame(b)).sort((a, b)=>a.rect.y-b.rect.y);

  const gapX = NODE_VIEW==='compact' ? 90 : 70, gapY = LAYOUT_GAP + 14;
  const colX = [];
  let x = 30;
  cols.forEach(c=>{
    colX[c] = x;
    const widths = blocks.filter(b=>depth.get(b.id)===c && !isFrame(b)).map(b=>b.rect.w);
    x += (widths.length ? Math.max(...widths) : nodeWidth({})) + gapX;
  });
  let laneTop = 40;
  ARRANGE_LANES.forEach((_, l)=>{
    const placed = [];
    cols.forEach(c=>{
      let cy = laneTop;
      cell(c, l).forEach(b=>{
        moveBlock(b, colX[c]-b.rect.x, cy-b.rect.y);
        placed.push(b.rect);
        cy += b.rect.h + gapY;
      });
    });
    blocks.filter(b=>lane(b)===l && isFrame(b)).sort((a, b)=>depth.get(a.id)-depth.get(b.id)).forEach(b=>{
      const fx = colX[depth.get(b.id)];
      const under = placed.filter(r=> r.x < fx+b.rect.w && fx < r.x+r.w);
      const fy = under.length ? Math.max(...under.map(r=>r.y+r.h)) + gapY : laneTop;
      moveBlock(b, fx-b.rect.x, fy-b.rect.y);
      placed.push(b.rect);
    });
    if(placed.length) laneTop = Math.max(...placed.map(r=>r.y+r.h)) + gapY + 16;
  });

  // the session band follows the EMG lane
  const emgLane = ARRANGE_LANES.indexOf('emg');
  const emgBlocks = blocks.filter(b=>lane(b)===emgLane);
  sessionRailBox = emgBlocks.length ? {
    top: Math.min(...emgBlocks.map(b=>b.rect.y)) - 14,
    height: Math.max(...emgBlocks.map(b=>b.rect.y+b.rect.h)) - Math.min(...emgBlocks.map(b=>b.rect.y)) + 28,
  } : null;
}

document.getElementById('arrangeBtn').addEventListener('click', ()=>{
  animateLayout(arrangeByDepth, ()=> fitCanvasToView(1));
});
