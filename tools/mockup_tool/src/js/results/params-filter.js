// ================= Review outputs: Parameters table filter =================
const paramsSearch=document.getElementById('paramsSearch');
const paramsTable=document.getElementById('paramsTable');
const paramsCount=document.getElementById('paramsCount');
const modChips=[...document.querySelectorAll('.mod-chip')];
function applyParamsFilter(){
  const q=paramsSearch.value.trim().toLowerCase();
  const activeMods=new Set(modChips.filter(c=>c.classList.contains('on')).map(c=>c.dataset.pmod));
  let shown=0;
  paramsTable.querySelectorAll('tbody tr').forEach(tr=>{
    const mod=tr.dataset.mod;
    const text=tr.textContent.toLowerCase();
    const ok = activeMods.has(mod) && (!q || text.includes(q));
    tr.classList.toggle('pfiltered', !ok);
    if(ok) shown++;
  });
  paramsCount.textContent = shown+' value'+(shown===1?'':'s');
}
paramsSearch.addEventListener('input', applyParamsFilter);
modChips.forEach(c=>c.addEventListener('click', ()=>{ c.classList.toggle('on'); applyParamsFilter(); }));
applyParamsFilter();

