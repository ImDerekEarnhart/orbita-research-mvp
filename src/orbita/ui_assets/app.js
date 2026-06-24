const CSRF = document.querySelector('meta[name="orbita-csrf"]').content;
const state = {view:'overview', claims:[], selectedClaim:null, analyses:[], selectedAnalysis:null, discoveries:[], selectedDiscovery:null, evaluations:[], selectedEvaluation:null, executions:[], selectedExecution:null, drafts:[], schedules:[], proposals:[], selectedProposal:null, graphs:{snapshots:[],diffs:[]}};
const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const esc = (v='') => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const fmt = (d) => { if(!d) return '—'; const x=new Date(d); return Number.isNaN(x.valueOf()) ? esc(d) : x.toLocaleString(); };
const short = (v,n=34) => String(v ?? '').length>n ? `${String(v).slice(0,n)}…` : String(v ?? '');
const jsonPretty = v => JSON.stringify(v ?? {}, null, 2);
const badge = (text, cls=text) => `<span class="badge ${esc(cls)}">${esc(text)}</span>`;
const isOk = v => v === true;

async function api(path, options={}) {
  const opts={credentials:'same-origin',...options,headers:{...(options.headers||{})}};
  if(opts.method && opts.method !== 'GET') {
    opts.headers['Content-Type']='application/json';
    opts.headers['X-Orbita-CSRF']=CSRF;
  }
  const response=await fetch(path,opts);
  const data=await response.json().catch(()=>({error:`HTTP ${response.status}`}));
  if(!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function flash(message,type='success') {
  const el=$('#flash'); el.textContent=message; el.className=`flash show ${type}`;
  clearTimeout(flash.timer); flash.timer=setTimeout(()=>el.className='flash',5000);
}

function setBusy(button,busy,label='Working…') {
  if(!button) return; if(busy){button.dataset.old=button.textContent;button.textContent=label;button.disabled=true;}else{button.textContent=button.dataset.old||button.textContent;button.disabled=false;}
}

const titles={overview:['System state','Overview'],language:['Meaning-first generation','Warranted language'],claims:['Knowledge ledger','Typed claims'],analyses:['Reproducible evidence','Dataset analyses'],discoveries:['Governed science','Discovery investigations'],evaluations:['Measured governance','Comparative evaluations'],executions:['Proof-carrying action','Container executions'],automation:['Governed computer work','Automation'],proposals:['Model boundary','Proposal review'],graphs:['Dependency reasoning','Collapse graphs'],audit:['Permanent provenance','Audit trail']};
async function switchView(view) {
  state.view=view;
  $$('.view').forEach(x=>x.classList.toggle('active',x.id===`view-${view}`));
  $$('.nav-item').forEach(x=>x.classList.toggle('active',x.dataset.view===view));
  $('#view-eyebrow').textContent=titles[view][0]; $('#view-title').textContent=titles[view][1];
  try {
    if(view==='overview') await loadDashboard();
    if(view==='claims') await loadClaims();
    if(view==='analyses') await loadAnalyses();
    if(view==='discoveries') await loadDiscoveries();
    if(view==='evaluations') await loadEvaluations();
    if(view==='executions') await loadExecutions();
    if(view==='automation') await loadAutomation();
    if(view==='proposals') await loadProposals();
    if(view==='graphs') await loadGraphs();
    if(view==='audit') await loadAudit();
  } catch(e){flash(e.message,'error');}
}

function eventTimeline(items) {
  if(!items.length) return '<div class="empty-state horizontal"><div class="empty-icon">≡</div><div><h3>No events yet</h3><p>Mutations will appear here as immutable audit entries.</p></div></div>';
  return items.map(ev=>`<div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-body"><strong>${esc(ev.event_type)}</strong><p>${esc(ev.entity_type)} · <span class="mono">${esc(short(ev.entity_id,24))}</span> · ${esc(ev.actor)} (${esc(ev.actor_role)})</p><time>${fmt(ev.created_at)}</time></div></div>`).join('');
}

async function health(){
  try{const h=await api('/api/health');$('#health-label').textContent=`Runtime ${h.ui_version} active`;}
  catch(e){$('#health-label').textContent='Runtime unavailable';}
}

async function loadDashboard(){
  const d=await api('/api/dashboard');
  const support=d.claims.by_support||{};
  const metrics=[
    ['Claims',d.claims.total,`${support.supported||0} currently supported`],
    ['Analysis receipts',d.analyses.total,`${d.analyses.integrity_failures} integrity failures`],
    ['Investigations',d.discoveries.total,`${d.discoveries.awaiting_approval} awaiting approval · ${d.discoveries.concluded} concluded`],
    ['Evaluation suites',d.evaluations.suites,`${d.evaluations.runs} scored runs · ${d.evaluations.reports} reports`],
    ['Container runs',d.executions.total,`${d.executions.waiting_approval} awaiting approval · ${d.executions.succeeded} succeeded`],
    ['Automations',d.automation.schedules,`${d.automation.active} active · ${d.automation.blocked} blocked · ${d.automation.drafts} drafts`],
    ['Review queue',d.proposals.pending_review,`${d.proposals.batches} model batches retained`],
    ['Graph records',d.graphs.snapshots+d.graphs.diffs,`${d.graphs.snapshots} snapshots · ${d.graphs.diffs} diffs`]
  ];
  $('#overview-metrics').innerHTML=metrics.map(m=>`<article class="metric"><label>${esc(m[0])}</label><strong>${esc(m[1])}</strong><span>${esc(m[2])}</span></article>`).join('');
  const max=Math.max(1,...Object.values(support));
  $('#support-bars').innerHTML=['supported','challenged','unknown','unsupported'].map(k=>`<div class="bar-row"><span>${esc(k)}</span><div class="bar-track"><div class="bar-fill" style="width:${((support[k]||0)/max)*100}%"></div></div><strong>${support[k]||0}</strong></div>`).join('');
  $('#recent-events').innerHTML=eventTimeline(d.recent_events||[]);
}

function renderLanguageResponse(d){
  const sentenceCards=(d.sentences||[]).map((s,i)=>`<div class="evidence-card"><div class="record-meta">${badge(s.semantic_act,s.semantic_act)}${badge(s.support_state,s.support_state)}<span>sentence ${i+1}</span></div><p><strong>${esc(s.text)}</strong></p><div class="idline">claims: ${esc((s.claim_ids||[]).join(', ')||'none')}</div><div class="idline">evidence: ${esc((s.evidence_ids||[]).join(', ')||'none')}</div><div class="idline">proofs: ${esc((s.proof_ids||[]).join(', ')||'none')}</div></div>`).join('');
  $('#language-answer').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Warranted response</p><h2>${esc(d.answer_text)}</h2><div class="idline">${esc(d.id)} · ${esc(short(d.response_hash,26))}</div></div>${badge(d.status,d.status)}</div><div class="detail-section"><h3>Sentence receipts</h3>${sentenceCards||'<p class="muted">No sentences were generated.</p>'}</div>`;
  $('#language-trace').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Semantic trace</p><h2>Frame and grounding</h2></div></div><div class="two-column"><div><h3>Parsed frame</h3><pre class="json">${esc(jsonPretty(d.frame))}</pre></div><div><h3>Grounding</h3><pre class="json">${esc(jsonPretty(d.grounding))}</pre></div></div>`;
}

async function loadClaims(){
  const q=encodeURIComponent($('#claim-search').value.trim());
  const st=encodeURIComponent($('#claim-state-filter').value);
  const life=encodeURIComponent($('#claim-status-filter').value);
  const data=await api(`/api/claims?q=${q}&state=${st}&status=${life}`);
  state.claims=data.items; $('#claim-count').textContent=data.count;
  const list=$('#claim-list');
  list.innerHTML=data.items.length ? data.items.map(c=>`<div class="record ${state.selectedClaim===c.id?'selected':''}"><button data-claim-id="${esc(c.id)}"><div class="record-title">${esc(c.canonical_text)}</div><div class="record-meta">${badge(c.support_state,c.support_state)}${badge(c.status,c.status)}<span class="mono">${esc(short(c.id,21))}</span></div></button></div>`).join('') : '<div class="empty-state"><div class="empty-icon">◇</div><h3>No matching claims</h3><p>Create a typed proposition or broaden the filters.</p></div>';
  $$('[data-claim-id]',list).forEach(b=>b.addEventListener('click',()=>selectClaim(b.dataset.claimId)));
  if(state.selectedClaim && data.items.some(c=>c.id===state.selectedClaim)) await selectClaim(state.selectedClaim,false);
}

async function selectClaim(id,reloadList=true){
  state.selectedClaim=id; const d=await api(`/api/claims/${id}`); renderClaimDetail(d);
  if(reloadList) $$('.record','#claim-list').forEach(r=>r.classList.toggle('selected',r.querySelector('button')?.dataset.claimId===id));
}

function renderClaimDetail(d){
  const c=d.claim, s=d.support, rel=c.relation;
  const relationBlock=rel ? `<div class="kv-grid"><div class="kv"><label>Subject</label><div>${esc(rel.subject_name)} <span class="muted">(${esc(rel.subject_type)})</span></div></div><div class="kv"><label>Predicate</label><div class="mono">${esc(rel.predicate)}</div></div><div class="kv"><label>Object</label><div>${esc(rel.object.kind==='entity'?rel.object.name:JSON.stringify(rel.object.value))}${rel.object.unit?` ${esc(rel.object.unit)}`:''}</div></div><div class="kv"><label>Polarity / validity</label><div>${rel.polarity?'positive':'negative'} · ${esc(rel.valid_from||'unbounded')} → ${esc(rel.valid_to||'unbounded')}</div></div></div>` : '';
  const evidence=d.evidence.length ? d.evidence.map(e=>`<div class="evidence-card"><div class="record-meta">${badge(e.stance,e.stance==='support'?'supported':'unsupported')}${badge(e.source_kind)}${e.active?badge('active','supported'):badge('revoked','rejected')}</div><p><strong>${esc(e.source_uri)}</strong></p><p>${esc(e.excerpt)}</p><div class="idline">independence: ${esc(e.independence_key)} · hash: ${esc(short(e.content_hash,22))}</div>${e.active?`<button class="button danger small revoke-evidence" data-id="${esc(e.id)}">Revoke</button>`:''}</div>`).join('') : '<p class="muted">No evidence is attached.</p>';
  const proofs=d.proofs.length ? d.proofs.map(p=>`<div class="proof-card"><div class="record-meta">${badge(p.active?'active':'inactive',p.active?'supported':'rejected')}<span>${esc(p.rule)}</span></div><p>${p.premises.map(x=>`<span class="chip">${esc(x.canonical_text)}</span>`).join(' ')}</p><div class="idline">AND proof set ${esc(p.id)}</div></div>`).join('') : '<p class="muted">No derivation proofs are registered.</p>';
  const contradictions=d.contradictions.length ? d.contradictions.map(x=>`<div class="evidence-card"><div class="record-meta">${badge(x.active?'active contradiction':'inactive','challenged')}</div><p>${esc(x.claim_a_text)} ↔ ${esc(x.claim_b_text)}</p><p>${esc(x.rationale)}</p></div>`).join('') : '<p class="muted">No contradiction links.</p>';
  $('#claim-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Claim detail</p><h2 class="detail-title">${esc(c.canonical_text)}</h2><div class="idline">${esc(c.id)}</div></div><div>${badge(s.state,s.state)} ${badge(c.status,c.status)}</div></div>${relationBlock}<div class="detail-section"><div class="panel-head"><h3>Why this state?</h3><button class="button primary small" id="detail-add-evidence">Add evidence</button></div><ul class="reason-list">${s.reasons.map(r=>`<li>${esc(r)}</li>`).join('')}</ul></div><div class="detail-section"><h3>Evidence attestations (${d.evidence.length})</h3>${evidence}</div><div class="detail-section"><h3>Proof alternatives (${d.proofs.length})</h3>${proofs}</div><div class="detail-section"><h3>Contradictions (${d.contradictions.length})</h3>${contradictions}</div><div class="detail-section"><h3>Claim event history</h3>${eventTimeline(d.history.slice().reverse())}</div>`;
  $('#detail-add-evidence').addEventListener('click',()=>openEvidence(c.id));
  $$('.revoke-evidence','#claim-detail').forEach(b=>b.addEventListener('click',()=>revokeEvidence(b.dataset.id,c.id)));
}

function openClaimDialog(){ $('#claim-dialog').showModal(); }
function openEvidence(claimId){ $('#evidence-claim-id').value=claimId; $('#evidence-dialog').showModal(); }
async function revokeEvidence(id,claimId){
  const rationale=prompt('Why is this evidence being revoked?'); if(!rationale) return;
  await api(`/api/evidence/${id}/revoke`,{method:'POST',body:JSON.stringify({rationale,actor:'Derek Earnhart'})});
  flash('Evidence revoked and dependent support recomputed.'); await selectClaim(claimId); await loadClaims();
}

function analysisParameterFields(){
  const type=$('#analysis-type').value; let html='';
  if(type==='pearson_correlation') html='<label class="field">X column<input id="param-x" required placeholder="marker_a"></label><label class="field">Y column<input id="param-y" required placeholder="response"></label>';
  if(type==='group_mean_difference') html='<label class="field">Group column<input id="param-group" required placeholder="cohort"></label><label class="field">Value column<input id="param-value" required placeholder="response"></label>';
  if(type==='column_summary') html='<label class="field span-2">Numeric column<input id="param-columns" placeholder="marker_a"></label>';
  $('#analysis-parameter-fields').innerHTML=html;
}

async function loadAnalyses(){
  const d=await api('/api/analyses'); state.analyses=d.items; $('#analysis-count').textContent=d.count;
  $('#analysis-list').innerHTML=d.items.length ? d.items.map(r=>`<div class="record ${state.selectedAnalysis===r.id?'selected':''}"><button data-receipt-id="${esc(r.id)}"><div class="record-title">${esc(r.analysis_type)}</div><div class="record-meta">${badge(r.status,r.status)}${r.integrity_valid&&r.artifact_integrity_valid&&r.evidence_binding_valid?badge('integrity valid','supported'):badge('integrity issue','failed')}<span>${fmt(r.created_at)}</span></div></button></div>`).join('') : '<div class="empty-state"><div class="empty-icon">⌁</div><h3>No analysis receipts</h3><p>Upload a CSV and run one of the safe built-in analyzers.</p></div>';
  $$('[data-receipt-id]','#analysis-list').forEach(b=>b.addEventListener('click',()=>selectAnalysis(b.dataset.receiptId)));
}

async function selectAnalysis(id){state.selectedAnalysis=id;const r=await api(`/api/analyses/${id}`);renderAnalysis(r);$$('.record','#analysis-list').forEach(x=>x.classList.toggle('selected',x.querySelector('button')?.dataset.receiptId===id));}
function renderAnalysis(r){
  const integrity=[['Receipt',r.integrity_valid],['Artifacts',r.artifact_integrity_valid],['Evidence binding',r.evidence_binding_valid]].map(x=>`<span class="integrity-item ${isOk(x[1])?'ok':''}"><i></i>${x[0]} ${x[1]?'valid':'failed'}</span>`).join('');
  const assessments=(r.assessments||[]).length ? `<table class="summary-table"><thead><tr><th>Claim</th><th>Metric</th><th>Value</th><th>Outcome</th></tr></thead><tbody>${r.assessments.map(a=>`<tr><td class="mono">${esc(short(a.claim_id,24))}</td><td>${esc(a.metric_path)}</td><td>${esc(JSON.stringify(a.metric_value))}</td><td>${badge(a.outcome,a.outcome==='support'?'supported':a.outcome==='refute'?'unsupported':'unknown')}</td></tr>`).join('')}</tbody></table>`:'<p class="muted">No claim tests were declared.</p>';
  $('#analysis-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Analysis receipt</p><h2>${esc(r.analysis_type)}</h2><div class="idline">${esc(r.id)}</div></div><button class="button secondary small" id="reproduce-analysis">Reproduce</button></div><div class="integrity">${integrity}</div><div class="detail-section"><div class="kv-grid"><div class="kv"><label>Dataset hash</label><div class="mono">${esc(r.dataset_hash)}</div></div><div class="kv"><label>Code hash</label><div class="mono">${esc(r.code_hash)}</div></div><div class="kv"><label>Dataset</label><div>${esc(r.dataset_uri)}</div></div><div class="kv"><label>Status</label><div>${badge(r.status,r.status)}</div></div></div></div><div class="detail-section"><h3>Claim assessments</h3>${assessments}</div><div class="detail-section"><h3>Outputs</h3><pre class="json">${esc(jsonPretty(r.outputs))}</pre></div><div class="detail-section"><h3>Diagnostics</h3><pre class="json">${esc(jsonPretty(r.diagnostics))}</pre></div><div class="detail-section"><h3>Environment and parameters</h3><pre class="json">${esc(jsonPretty({environment:r.environment,parameters:r.parameters,preprocessing:r.preprocessing,comparison:r.comparison}))}</pre></div>`;
  $('#reproduce-analysis').addEventListener('click',()=>reproduceAnalysis(r.id));
}
async function reproduceAnalysis(id){const b=$('#reproduce-analysis');setBusy(b,true,'Reproducing…');try{const r=await api(`/api/analyses/${id}/reproduce`,{method:'POST',body:'{}'});flash(`Reproduction recorded: ${r.status}`);await loadAnalyses();await selectAnalysis(r.id);}catch(e){flash(e.message,'error');}finally{setBusy(b,false);}}

async function loadDiscoveries(){
  const d=await api('/api/discoveries');state.discoveries=d.items;$('#discovery-count').textContent=d.count;
  $('#discovery-list').innerHTML=d.items.length?d.items.map(inv=>`<div class="record ${state.selectedDiscovery===inv.id?'selected':''}"><button data-discovery-id="${esc(inv.id)}"><div class="record-title">${esc(inv.question)}</div><div class="record-meta">${badge(inv.status,inv.status)}<span>${inv.hypotheses.length} hypotheses</span><span>${fmt(inv.created_at)}</span></div></button></div>`).join(''):'<div class="empty-state"><div class="empty-icon">◎</div><h3>No investigations</h3><p>Upload a dataset and create a preregistered, counterexample-first investigation.</p></div>';
  $$('[data-discovery-id]','#discovery-list').forEach(b=>b.addEventListener('click',()=>selectDiscovery(b.dataset.discoveryId)));
  if(state.selectedDiscovery&&d.items.some(x=>x.id===state.selectedDiscovery))await selectDiscovery(state.selectedDiscovery,false);
}
async function selectDiscovery(id,mark=true){state.selectedDiscovery=id;const inv=await api(`/api/discoveries/${id}`);renderDiscovery(inv);if(mark)$$('.record','#discovery-list').forEach(x=>x.classList.toggle('selected',x.querySelector('button')?.dataset.discoveryId===id));}
function renderDiscovery(inv){
  const hypotheses=inv.hypotheses.length?`<table class="summary-table"><thead><tr><th>Claim</th><th>Discovery</th><th>Status</th><th>Confirmation / replication</th></tr></thead><tbody>${inv.hypotheses.map(h=>`<tr><td><span class="mono">${esc(short(h.claim_id,20))}</span><br>${esc(h.x_column)} ${esc(h.direction)} ${esc(h.y_column)}</td><td>r=${esc(Number(h.discovery_metrics.pearson_r||0).toFixed(3))}<br><span class="muted">non-warranting</span></td><td>${badge(h.status,h.status)}</td><td><pre class="json compact-json">${esc(jsonPretty({confirmation:h.confirmation_result,replication:h.replication_result}))}</pre></td></tr>`).join('')}</tbody></table>`:'<p class="muted">No candidates survived the discovery filter.</p>';
  let actions='';
  if(inv.status.includes('awaiting_')&&inv.status.includes('_approval'))actions='<button class="button primary small discovery-approve">Approve exact manifest</button>';
  if(inv.status.includes('awaiting_')&&inv.status.includes('_approval'))actions+='<button class="button secondary small discovery-advance">Check / advance</button>';
  if(['concluded','failed','budget_exhausted','no_candidates'].includes(inv.status))actions+='<button class="button secondary small discovery-report">Recompile report</button>';
  const runId=inv.resume_cursor?.[`${inv.current_phase}_run_id`]||'—';
  $('#discovery-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Governed investigation</p><h2>${esc(inv.question)}</h2><div class="idline">${esc(inv.id)}</div></div>${badge(inv.status,inv.status)}</div><div class="toolbar slim">${actions}</div><div class="kv-grid"><div class="kv"><label>Current phase</label><div>${esc(inv.current_phase)}</div></div><div class="kv"><label>Current execution</label><div class="mono">${esc(runId)}</div></div><div class="kv"><label>Dataset hash</label><div class="mono">${esc(short(inv.dataset_hash,30))}</div></div><div class="kv"><label>Replication hash</label><div class="mono">${esc(short(inv.replication_dataset_hash||'not supplied',30))}</div></div><div class="kv"><label>Budget used</label><div>${esc(jsonPretty(inv.budget_used))}</div></div><div class="kv"><label>Report integrity</label><div>${inv.report_integrity_valid===true?badge('valid','supported'):inv.report_integrity_valid===false?badge('failed','failed'):badge('not issued','unknown')}</div></div></div><div class="detail-section"><h3>Preregistered hypotheses</h3>${hypotheses}</div><div class="detail-section"><h3>Dataset profile</h3><pre class="json">${esc(jsonPretty(inv.profile))}</pre></div>${Object.keys(inv.report||{}).length?`<div class="detail-section"><h3>Investigation report</h3><pre class="json">${esc(jsonPretty(inv.report))}</pre></div>`:''}`;
  const approve=$('.discovery-approve','#discovery-detail');if(approve)approve.addEventListener('click',()=>{$('#discovery-review-id').value=inv.id;$('#discovery-review-dialog').showModal();});
  const advance=$('.discovery-advance','#discovery-detail');if(advance)advance.addEventListener('click',()=>advanceDiscovery(inv.id,advance));
  const report=$('.discovery-report','#discovery-detail');if(report)report.addEventListener('click',()=>compileDiscoveryReport(inv.id,report));
}
async function advanceDiscovery(id,button){setBusy(button,true,'Advancing…');try{const inv=await api(`/api/discoveries/${id}/advance`,{method:'POST',body:'{}'});flash(`Investigation state: ${inv.status}`);await loadDiscoveries();await selectDiscovery(id);}catch(e){flash(e.message,'error');}finally{setBusy(button,false);}}
async function compileDiscoveryReport(id,button){setBusy(button,true,'Compiling…');try{await api(`/api/discoveries/${id}/report`,{method:'POST',body:'{}'});flash('Investigation report compiled.');await selectDiscovery(id);}catch(e){flash(e.message,'error');}finally{setBusy(button,false);}}


async function loadEvaluations(){
  const d=await api('/api/evaluations');state.evaluations=d.items;$('#evaluation-count').textContent=d.count;
  $('#evaluation-list').innerHTML=d.items.length?d.items.map(suite=>`<div class="record ${state.selectedEvaluation===suite.id?'selected':''}"><button data-evaluation-id="${esc(suite.id)}"><div class="record-title">${esc(suite.name)}</div><div class="record-meta">${badge(suite.status,suite.status)}<span>${suite.runs.length} runs</span><span>v${esc(suite.version)}</span></div></button></div>`).join(''):'<div class="empty-state"><div class="empty-icon">◈</div><h3>No evaluation suites</h3><p>Create the sealed adversarial benchmark, then import empirical outputs or run labeled fixtures.</p></div>';
  $$('[data-evaluation-id]','#evaluation-list').forEach(b=>b.addEventListener('click',()=>selectEvaluation(b.dataset.evaluationId)));
  if(state.selectedEvaluation&&d.items.some(x=>x.id===state.selectedEvaluation))await selectEvaluation(state.selectedEvaluation,false);
}
async function selectEvaluation(id,mark=true){state.selectedEvaluation=id;const suite=await api(`/api/evaluations/${id}`);renderEvaluation(suite);if(mark)$$('.record','#evaluation-list').forEach(x=>x.classList.toggle('selected',x.querySelector('button')?.dataset.evaluationId===id));}
function pct(v){return v===null||v===undefined?'—':`${(100*Number(v)).toFixed(1)}%`;}
function renderEvaluation(suite){
  const runs=(suite.runs||[]).slice().sort((a,b)=>Number(b.metrics.overall_score)-Number(a.metrics.overall_score));
  const table=runs.length?`<table class="summary-table"><thead><tr><th>System</th><th>Mode</th><th>Overall</th><th>Unsupported</th><th>Recovery</th><th>False success</th><th>Audit</th></tr></thead><tbody>${runs.map(r=>`<tr><td><button class="link-button evaluation-run" data-run-id="${esc(r.id)}">${esc(r.system.name)}</button></td><td>${badge(r.system.evaluation_mode,r.system.evaluation_mode)}</td><td>${Number(r.metrics.overall_score).toFixed(3)}</td><td>${pct(r.metrics.rates.unsupported_commitment_rate)}</td><td>${pct(r.metrics.rates.contradiction_recovery_rate)}</td><td>${pct(r.metrics.rates.false_success_rate)}</td><td>${pct(r.metrics.rates.audit_completeness)}</td></tr>`).join('')}</tbody></table>`:'<p class="muted">No runs have been scored.</p>';
  $('#evaluation-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Sealed benchmark</p><h2>${esc(suite.name)}</h2><div class="idline">${esc(suite.id)} · ${esc(suite.suite_hash)}</div></div>${suite.integrity_valid?badge('integrity valid','supported'):badge('integrity failed','failed')}</div><div class="toolbar slim"><button class="button secondary small evaluation-fixture" data-profile="base_llm">Fixture: base LLM</button><button class="button secondary small evaluation-fixture" data-profile="rag">Fixture: RAG</button><button class="button secondary small evaluation-fixture" data-profile="final_answer_verifier">Fixture: verifier</button><button class="button secondary small evaluation-fixture" data-profile="orbita">Fixture: Orbita</button><button class="button primary small evaluation-report">Compile report</button></div><div class="kv-grid"><div class="kv"><label>Version</label><div>${esc(suite.version)}</div></div><div class="kv"><label>Tasks</label><div>${suite.tasks.length}</div></div><div class="kv"><label>Runs</label><div>${runs.length}</div></div><div class="kv"><label>Report</label><div>${suite.report_hash?badge('issued','supported'):badge('not issued','unknown')}</div></div></div><div class="detail-section"><h3>Comparative results</h3>${table}</div><div class="detail-section"><h3>Interpretation boundary</h3><p class="muted">Synthetic fixtures validate the scoring machinery only. Publishable performance claims require empirical outputs from real systems on a hidden task partition.</p></div>`;
  $$('.evaluation-fixture','#evaluation-detail').forEach(b=>b.addEventListener('click',()=>createEvaluationFixture(suite.id,b.dataset.profile,b)));
  $('.evaluation-report','#evaluation-detail').addEventListener('click',()=>compileEvaluationReport(suite.id,$('.evaluation-report','#evaluation-detail')));
  $$('.evaluation-run','#evaluation-detail').forEach(b=>b.addEventListener('click',()=>showEvaluationRun(b.dataset.runId)));
}
async function createEvaluationFixture(id,profile,button){setBusy(button,true,'Scoring…');try{await api(`/api/evaluations/${id}/fixture`,{method:'POST',body:JSON.stringify({profile})});flash(`Synthetic ${profile} fixture scored.`);await loadEvaluations();await selectEvaluation(id);}catch(e){flash(e.message,'error');}finally{setBusy(button,false);}}
async function compileEvaluationReport(id,button){setBusy(button,true,'Compiling…');try{await api(`/api/evaluations/${id}/report`,{method:'POST',body:'{}'});flash('Hash-verifiable comparative report compiled.');await loadEvaluations();await selectEvaluation(id);}catch(e){flash(e.message,'error');}finally{setBusy(button,false);}}
async function showEvaluationRun(id){const r=await api(`/api/evaluation-runs/${id}`);const rates=r.metrics.rates;const modal=`<div class="panel-head"><div><p class="eyebrow">Scored run</p><h2>${esc(r.system.name)}</h2><div class="idline">${esc(r.id)}</div></div>${badge(r.system.evaluation_mode,r.system.evaluation_mode)}</div><div class="kv-grid"><div class="kv"><label>Overall</label><div>${Number(r.metrics.overall_score).toFixed(3)}</div></div><div class="kv"><label>Unsupported</label><div>${pct(rates.unsupported_commitment_rate)}</div></div><div class="kv"><label>Recovery</label><div>${pct(rates.contradiction_recovery_rate)}</div></div><div class="kv"><label>Collapse accuracy</label><div>${pct(rates.evidence_collapse_accuracy)}</div></div><div class="kv"><label>False success</label><div>${pct(rates.false_success_rate)}</div></div><div class="kv"><label>Audit completeness</label><div>${pct(rates.audit_completeness)}</div></div></div><div class="detail-section"><h3>Task scores</h3><pre class="json">${esc(jsonPretty(r.metrics.task_scores))}</pre></div>`;$('#evaluation-detail').innerHTML=modal;}

async function loadExecutions(){
  const d=await api('/api/executions');state.executions=d.items;$('#execution-count').textContent=d.count;
  const engines=Object.entries(d.runtime.engines||{}).map(([name,ok])=>`<div class="kv"><label>${esc(name)}</label><div>${ok?badge('available','supported'):badge('unavailable','unknown')}</div></div>`).join('');
  $('#execution-runtime').innerHTML=`${engines}<div class="kv"><label>Network</label><div>${esc(d.runtime.network_policy)}</div></div><div class="kv"><label>Approval</label><div>${esc(d.runtime.approval_policy)}</div></div>`;
  $('#execution-list').innerHTML=d.items.length?d.items.map(r=>`<div class="record ${state.selectedExecution===r.id?'selected':''}"><button data-execution-id="${esc(r.id)}"><div class="record-title">${esc(r.name)}</div><div class="record-meta">${badge(r.status,r.status)}<span class="mono">${esc(short(r.manifest_hash,18))}</span><span>${fmt(r.created_at)}</span></div></button></div>`).join(''):'<div class="empty-state"><div class="empty-icon">▣</div><h3>No container runs</h3><p>Stage a digest-pinned manifest. Nothing executes until the exact hash is approved.</p></div>';
  $$('[data-execution-id]','#execution-list').forEach(b=>b.addEventListener('click',()=>selectExecution(b.dataset.executionId)));
  if(state.selectedExecution && d.items.some(x=>x.id===state.selectedExecution)) await selectExecution(state.selectedExecution,false);
}
async function selectExecution(id,mark=true){state.selectedExecution=id;const r=await api(`/api/executions/${id}`);renderExecution(r);if(mark)$$('.record','#execution-list').forEach(x=>x.classList.toggle('selected',x.querySelector('button')?.dataset.executionId===id));}
function renderExecution(r){
  const approval=r.approval||{};
  const integrity=[['Manifest',r.manifest_integrity_valid],['Artifacts',r.artifact_integrity_valid],['Receipt',r.receipt_integrity_valid]].map(x=>`<span class="integrity-item ${x[1]===true?'ok':''}"><i></i>${x[0]} ${x[1]===null?'not issued':x[1]?'valid':'failed'}</span>`).join('');
  const checks=(r.checks||[]).length?`<table class="summary-table"><thead><tr><th>Obligation</th><th>Path</th><th>Result</th><th>Detail</th></tr></thead><tbody>${r.checks.map(c=>`<tr><td>${esc(c.type)}</td><td class="mono">${esc(c.path||'—')}</td><td>${badge(c.ok?'pass':'fail',c.ok?'supported':'failed')}</td><td>${esc(c.detail||'')}</td></tr>`).join('')}</tbody></table>`:'<p class="muted">No postconditions have been evaluated yet.</p>';
  const assessments=(r.assessments||[]).length?`<table class="summary-table"><thead><tr><th>Claim</th><th>Metric</th><th>Value</th><th>Outcome</th></tr></thead><tbody>${r.assessments.map(a=>`<tr><td class="mono">${esc(short(a.claim_id,22))}</td><td>${esc(a.output_path)} · ${esc(a.metric_path)}</td><td>${esc(JSON.stringify(a.metric_value))}</td><td>${badge(a.outcome,a.outcome==='support'?'supported':a.outcome==='refute'?'unsupported':'unknown')}</td></tr>`).join('')}</tbody></table>`:'';
  let actions='';
  if(r.status==='waiting_approval') actions='<button class="button primary small execution-approve">Approve exact manifest</button><button class="button danger small execution-reject">Reject</button>';
  if(r.status==='approved') actions='<button class="button primary small execution-run">Run in OCI container</button>';
  if(r.status==='succeeded') actions='<button class="button secondary small execution-reproduce">Prepare reproduction</button>';
  $('#execution-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Container execution</p><h2>${esc(r.name)}</h2><div class="idline">${esc(r.id)} · manifest ${esc(r.manifest_hash)}</div></div><div>${badge(r.status,r.status)}</div></div><div class="integrity">${integrity}</div><div class="detail-section"><div class="toolbar slim">${actions}</div><div class="kv-grid"><div class="kv"><label>Image</label><div class="mono">${esc(r.image_ref)}</div></div><div class="kv"><label>Engine</label><div>${esc(r.engine_used||'not run')}</div></div><div class="kv"><label>Approval</label><div>${esc(approval.status||'none')} ${approval.reviewer?`· ${esc(approval.reviewer)}`:''}</div></div><div class="kv"><label>Exit / timeout</label><div>${r.exit_code===null?'—':esc(r.exit_code)} / ${r.timed_out?'yes':'no'}</div></div></div></div><div class="detail-section"><h3>Security-bound manifest</h3><pre class="json">${esc(jsonPretty(r.manifest))}</pre></div><div class="detail-section"><h3>Proof obligations</h3>${checks}</div>${assessments?`<div class="detail-section"><h3>Claim assessments</h3>${assessments}</div>`:''}<div class="detail-section"><h3>Standard output</h3><pre class="json">${esc(r.stdout||'')}</pre><h3>Standard error</h3><pre class="json">${esc(r.stderr||'')}</pre></div>`;
  const approve=$('.execution-approve','#execution-detail');if(approve)approve.addEventListener('click',()=>openExecutionReview(r.id,'approve'));
  const reject=$('.execution-reject','#execution-detail');if(reject)reject.addEventListener('click',()=>openExecutionReview(r.id,'reject'));
  const run=$('.execution-run','#execution-detail');if(run)run.addEventListener('click',()=>runExecution(r.id,run));
  const reproduce=$('.execution-reproduce','#execution-detail');if(reproduce)reproduce.addEventListener('click',()=>reproduceExecution(r.id,reproduce));
}
function openExecutionReview(id,decision){$('#execution-review-id').value=id;$('#execution-review-decision').value=decision;$('#execution-review-submit').textContent=decision==='approve'?'Approve exact manifest':'Reject execution';$('#execution-review-submit').className=decision==='approve'?'button primary':'button danger';$('#execution-review-dialog').showModal();}
async function runExecution(id,button){setBusy(button,true,'Running…');try{const r=await api(`/api/executions/${id}/run`,{method:'POST',body:'{}'});flash(`Execution finalized: ${r.status}`);await loadExecutions();await selectExecution(id);}catch(e){flash(e.message,'error');}finally{setBusy(button,false);}}
async function reproduceExecution(id,button){setBusy(button,true,'Staging…');try{const r=await api(`/api/executions/${id}/reproduce`,{method:'POST',body:'{}'});flash('Reproduction staged and awaiting a new approval.');await loadExecutions();await selectExecution(r.id);}catch(e){flash(e.message,'error');}finally{setBusy(button,false);}}


async function loadAutomation(){
  const [status,drafts,schedules]=await Promise.all([
    api('/api/integrations/status'),api('/api/integrations/drafts'),api('/api/schedules')
  ]);
  state.drafts=drafts.items||[]; state.schedules=schedules.items||[];
  $('#automation-draft-count').textContent=drafts.count||0;
  $('#automation-schedule-count').textContent=schedules.count||0;
  $('#automation-status').innerHTML=`<div class="kv"><label>Provider</label><div>${status.provider_bound?badge(status.provider_name||'bound','supported'):badge('not bound','unknown')}</div></div><div class="kv"><label>Drafts</label><div>${esc(status.drafts||0)}</div></div><div class="kv"><label>Receipts</label><div>${esc(status.receipts||0)}</div></div><div class="kv"><label>Windows apps</label><div>${esc(status.windows_apps||0)}</div></div><div class="kv"><label>Policy</label><div>draft-first · exact approval · local verification</div></div>`;
  $('#automation-drafts').innerHTML=state.drafts.length?state.drafts.map(d=>`<div class="record"><button type="button" data-automation-draft="${esc(d.id)}"><div class="record-title">${esc(d.draft_kind==='email'?(d.payload.subject||'(no subject)'):(d.payload.title||'(untitled event)'))}</div><div class="record-meta">${badge(d.draft_kind,d.draft_kind)}${badge(d.status,d.status)}<span>${fmt(d.created_at)}</span></div><div class="idline">${esc(d.id)} · ${esc(short(d.payload_hash,18))}</div></button></div>`).join(''):'<div class="empty-state"><div class="empty-icon">✉</div><h3>No drafts</h3><p>Create an email or calendar draft. Nothing is sent automatically.</p></div>';
  $('#automation-schedules').innerHTML=state.schedules.length?state.schedules.map(x=>`<div class="record"><button type="button" data-automation-schedule="${esc(x.id)}"><div class="record-title">${esc(x.name)}</div><div class="record-meta">${badge(x.status,x.status)}${badge(x.schedule_kind,x.schedule_kind)}<span>runs ${esc(x.run_count)}${x.max_runs?`/${esc(x.max_runs)}`:''}</span></div><p>${esc(x.goal_utterance)}</p><div class="idline">next: ${fmt(x.next_run_at)} · ${esc(x.id)}</div></button></div>`).join(''):'<div class="empty-state"><div class="empty-icon">⚙</div><h3>No scheduled tasks</h3><p>Create a one-time or interval task backed by the durable computer-plan engine.</p></div>';
  $$('[data-automation-draft]').forEach(b=>b.addEventListener('click',()=>showAutomationDraft(b.dataset.automationDraft)));
  $$('[data-automation-schedule]').forEach(b=>b.addEventListener('click',()=>showAutomationSchedule(b.dataset.automationSchedule)));
}
async function showAutomationDraft(id){
  const d=await api(`/api/integrations/drafts/${id}`);
  const exact=confirm(`Draft ${d.id}\n\n${jsonPretty(d.payload)}\n\nRequest exact-payload approval?`);
  if(!exact)return;
  const a=await api(`/api/integrations/drafts/${id}/request-approval`,{method:'POST',body:'{}'});
  const approve=confirm(`Approval ${a.id}\nPayload hash: ${a.payload_hash}\n\nApprove this exact draft?`);
  const reviewer=prompt('Reviewer name','Derek Earnhart')||'';
  const rationale=prompt('Rationale',approve?'Approved exact draft':'Rejected')||'';
  await api(`/api/integrations/approvals/${a.id}/decide`,{method:'POST',body:JSON.stringify({decision:approve?'approve':'reject',reviewer,rationale})});
  flash(approve?'Exact draft approved. Execute it through the authenticated OpenClaw CLI bridge.':'Draft rejected.');
  await loadAutomation();
}
async function showAutomationSchedule(id){
  const d=await api(`/api/schedules/${id}`);
  if(d.status==='blocked' && confirm(`Schedule is blocked. Resume plan ${d.runs?.at(-1)?.plan_id||''}?`)){
    await api(`/api/schedules/${id}/resume`,{method:'POST',body:JSON.stringify({worker:'local-ui'})});
    flash('Scheduled plan resumed.'); await loadAutomation();
  } else {
    alert(jsonPretty(d));
  }
}

async function loadProposals(){
  const status=encodeURIComponent($('#proposal-status-filter').value); const d=await api(`/api/proposals?status=${status}`);state.proposals=d.items;$('#proposal-count').textContent=d.count;
  $('#proposal-list').innerHTML=d.items.length?d.items.map(b=>`<div class="record ${state.selectedProposal===b.id?'selected':''}"><button data-batch-id="${esc(b.id)}"><div class="record-title">${esc(b.provider)} · ${esc(b.model_name)}</div><div class="record-meta">${badge(b.status,b.status)}<span>${b.items.length} items</span><span>${fmt(b.created_at)}</span></div></button></div>`).join(''):'<div class="empty-state"><div class="empty-icon">✦</div><h3>No model batches</h3><p>Import a schema-constrained response. Model content remains non-warranting.</p></div>';
  $$('[data-batch-id]','#proposal-list').forEach(b=>b.addEventListener('click',()=>selectProposal(b.dataset.batchId)));
}
async function selectProposal(id){state.selectedProposal=id;const b=await api(`/api/proposals/${id}`);renderProposal(b);$$('.record','#proposal-list').forEach(x=>x.classList.toggle('selected',x.querySelector('button')?.dataset.batchId===id));}
function renderProposal(b){
  const items=b.items.map(i=>`<div class="proposal-item"><div class="record-meta">${badge(i.item_type)}${badge(i.status,i.status)}${i.requires_human_review?badge('human review required','challenged'):''}</div><p><strong>${esc(i.local_id)}</strong> — ${esc(i.rationale)}</p><pre class="json">${esc(jsonPretty(i.payload))}</pre>${i.status==='quarantined'&&i.requires_human_review?`<button class="button secondary small review-item" data-id="${esc(i.id)}">Review item</button>`:''}</div>`).join('');
  $('#proposal-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">Model proposal batch</p><h2>${esc(b.provider)} · ${esc(b.model_name)}${b.model_version?` @ ${esc(b.model_version)}`:''}</h2><div class="idline">${esc(b.id)} · response hash ${esc(short(b.response_hash,26))}</div></div>${badge(b.status,b.status)}</div><div class="kv-grid"><div class="kv"><label>Created</label><div>${fmt(b.created_at)}</div></div><div class="kv"><label>Schema</label><div>${esc(b.schema_version)}</div></div><div class="kv"><label>System prompt hash</label><div class="mono">${esc(short(b.system_prompt_hash,28))}</div></div><div class="kv"><label>User prompt hash</label><div class="mono">${esc(short(b.user_prompt_hash,28))}</div></div></div><div class="detail-section"><h3>Proposal items (${b.items.length})</h3>${items||'<p class="muted">No items were created.</p>'}</div>${b.errors.length?`<div class="detail-section"><h3>Validation errors</h3><pre class="json">${esc(jsonPretty(b.errors))}</pre></div>`:''}<div class="detail-section"><h3>Exact prompts</h3><details><summary>System prompt</summary><pre class="json">${esc(b.system_prompt)}</pre></details><details><summary>User prompt</summary><pre class="json">${esc(b.user_prompt)}</pre></details></div>`;
  $$('.review-item','#proposal-detail').forEach(x=>x.addEventListener('click',()=>{ $('#review-item-id').value=x.dataset.id;$('#review-dialog').showModal();}));
}

async function loadGraphs(){
  const d=await api('/api/graphs');state.graphs=d;
  const snapOpts=d.snapshots.map(s=>`<option value="${esc(s.id)}">${esc(s.name)} · ${fmt(s.created_at)}</option>`).join('');
  $('#diff-before').innerHTML=`<option value="">Choose snapshot</option>${snapOpts}`;$('#diff-after').innerHTML=`<option value="">Choose snapshot</option>${snapOpts}`;
  $('#graph-selector').innerHTML=`<option value="">Choose a snapshot or diff</option><optgroup label="Snapshots">${d.snapshots.map(s=>`<option value="snapshot:${esc(s.id)}">${esc(s.name)}</option>`).join('')}</optgroup><optgroup label="Collapse diffs">${d.diffs.map(x=>`<option value="diff:${esc(x.id)}">${esc(x.name)}</option>`).join('')}</optgroup>`;
}
async function loadGraphSelection(value){if(!value)return;const [kind,id]=value.split(':');const d=await api(kind==='snapshot'?`/api/graphs/snapshots/${id}`:`/api/graphs/diffs/${id}`);renderGraph(d,kind);}
function renderGraph(d,kind){const obj=kind==='snapshot'?d.snapshot:d.diff;$('#graph-viewer-title').textContent=obj.name||'Epistemic graph';const summary=obj.summary||(d.snapshot?.summary)||{};$('#graph-summary').innerHTML=Object.entries(summary).filter(([,v])=>typeof v!=='object').map(([k,v])=>`<span class="chip">${esc(k.replaceAll('_',' '))}: ${esc(v)}</span>`).join('');$('#graph-viewer').innerHTML=d.svg||'<p>No SVG artifact.</p>';}

async function loadAudit(){const d=await api('/api/events?limit=300');$('#audit-events').innerHTML=eventTimeline(d.items);}

function wire(){
  $$('.nav-item').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
  $('#refresh-button').addEventListener('click',()=>switchView(state.view));
  $('#language-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Reasoning…');try{const d=await api('/api/language/ask',{method:'POST',body:JSON.stringify({utterance:$('#language-utterance').value})});renderLanguageResponse(d);flash(`Warranted response created: ${d.status}`);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#global-add-claim').addEventListener('click',openClaimDialog);$('#claims-add').addEventListener('click',openClaimDialog);
  let searchTimer;['claim-search','claim-state-filter','claim-status-filter'].forEach(id=>$('#'+id).addEventListener(id==='claim-search'?'input':'change',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(loadClaims,200);}));
  $('#claim-object-kind').addEventListener('change',()=>{const lit=$('#claim-object-kind').value==='literal';$('#claim-datatype-field').classList.toggle('hidden',!lit);$('#claim-unit-field').classList.toggle('hidden',!lit);$('#claim-object-type-field').classList.toggle('hidden',lit);});
  $('#claim-form').addEventListener('submit',async e=>{e.preventDefault();const button=e.submitter;setBusy(button,true,'Creating…');try{const payload={subject:$('#claim-subject').value,subject_type:$('#claim-subject-type').value,predicate:$('#claim-predicate').value,object_kind:$('#claim-object-kind').value,object_type:$('#claim-object-type').value,object_value:$('#claim-object-value').value,literal_datatype:$('#claim-datatype').value,unit:$('#claim-unit').value,polarity:$('#claim-polarity').checked,valid_from:$('#claim-valid-from').value,valid_to:$('#claim-valid-to').value,qualifiers:JSON.parse($('#claim-qualifiers').value||'{}'),actor:'Derek Earnhart'};const d=await api('/api/claims',{method:'POST',body:JSON.stringify(payload)});$('#claim-dialog').close();flash('Typed claim added as provisional.');await switchView('claims');await selectClaim(d.claim.id);}catch(err){flash(err.message,'error');}finally{setBusy(button,false);}});
  $('#evidence-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Attaching…');try{const claimId=$('#evidence-claim-id').value;const payload={stance:$('#evidence-stance').value,source_kind:$('#evidence-kind').value,source_uri:$('#evidence-uri').value,independence_key:$('#evidence-independence').value,excerpt:$('#evidence-excerpt').value,confidence:Number($('#evidence-confidence').value),actor:'Derek Earnhart'};await api(`/api/claims/${claimId}/evidence`,{method:'POST',body:JSON.stringify(payload)});$('#evidence-dialog').close();flash('Evidence attached and support recomputed.');await selectClaim(claimId);await loadClaims();}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#analysis-type').addEventListener('change',analysisParameterFields);analysisParameterFields();
  $('#analysis-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;const file=$('#analysis-file').files[0];if(!file){flash('Choose a CSV file.','error');return;}setBusy(b,true,'Analyzing…');try{const type=$('#analysis-type').value;let parameters={};if(type==='pearson_correlation')parameters={x:$('#param-x').value,y:$('#param-y').value};if(type==='group_mean_difference')parameters={group:$('#param-group').value,outcome:$('#param-value').value};if(type==='column_summary')parameters={column:$('#param-columns').value.split(',')[0].trim()};const claimTests=JSON.parse($('#analysis-claim-tests').value||'[]');const missing=['',...$('#analysis-missing').value.split(',').map(x=>x.trim()).filter(Boolean)];const r=await api('/api/analyses',{method:'POST',body:JSON.stringify({filename:file.name,csv_text:await file.text(),analysis_type:type,parameters,preprocessing:{missing_values:missing},claim_tests:claimTests})});flash(`Analysis receipt issued: ${r.status}`);await loadAnalyses();await selectAnalysis(r.id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#discovery-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;const file=$('#discovery-file').files[0];if(!file){flash('Choose a primary CSV.','error');return;}const rep=$('#discovery-replication-file').files[0];setBusy(b,true,'Mining and staging…');try{const payload={question:$('#discovery-question').value,filename:file.name,csv_text:await file.text(),replication_filename:rep?.name||null,replication_csv_text:rep?await rep.text():null,image:$('#discovery-image').value,min_rows:Number($('#discovery-min-rows').value),min_discovery_abs_r:Number($('#discovery-min-r').value),min_confirmation_abs_r:Number($('#discovery-confirm-r').value),permutation_trials:Number($('#discovery-permutations').value),bootstrap_trials:200};const inv=await api('/api/discoveries',{method:'POST',body:JSON.stringify(payload)});flash(`Investigation created: ${inv.status}`);await loadDiscoveries();await selectDiscovery(inv.id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#discovery-review-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Recording…');try{const id=$('#discovery-review-id').value;await api(`/api/discoveries/${id}/approve`,{method:'POST',body:JSON.stringify({reviewer:$('#discovery-reviewer').value,rationale:$('#discovery-rationale').value})});$('#discovery-review-dialog').close();flash('Exact discovery manifest approved.');await selectDiscovery(id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#evaluation-create-default').addEventListener('click',async e=>{const b=e.currentTarget;setBusy(b,true,'Creating…');try{const suite=await api('/api/evaluations/default',{method:'POST',body:'{}'});flash('Adversarial evaluation suite sealed.');await loadEvaluations();await selectEvaluation(suite.id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#execution-refresh').addEventListener('click',loadExecutions);
  $('#execution-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Staging…');try{const spec=JSON.parse($('#execution-spec').value);const r=await api('/api/executions',{method:'POST',body:JSON.stringify({spec})});flash('Manifest staged. Review the exact hash before approval.');await loadExecutions();await selectExecution(r.id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#execution-review-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Recording…');try{const id=$('#execution-review-id').value;const decision=$('#execution-review-decision').value;const payload={reviewer:$('#execution-reviewer').value,rationale:$('#execution-rationale').value};const r=await api(`/api/executions/${id}/${decision}`,{method:'POST',body:JSON.stringify(payload)});$('#execution-review-dialog').close();flash(`Execution ${decision==='approve'?'approved':'rejected'}: ${short(r.manifest_hash,18)}`);await loadExecutions();await selectExecution(id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#automation-draft-kind').addEventListener('change',()=>{const cal=$('#automation-draft-kind').value==='calendar';$$('.automation-calendar').forEach(x=>x.classList.toggle('hidden',!cal));});
  $('#automation-schedule-kind').addEventListener('change',()=>{const interval=$('#automation-schedule-kind').value==='interval';$$('.automation-once').forEach(x=>x.classList.toggle('hidden',interval));$$('.automation-interval').forEach(x=>x.classList.toggle('hidden',!interval));});
  $('#automation-draft-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Creating…');try{const kind=$('#automation-draft-kind').value;const target=$('#automation-draft-target').value.split(',').map(x=>x.trim()).filter(Boolean);const payload=kind==='email'?{kind,to:target,subject:$('#automation-draft-title').value,body:$('#automation-draft-body').value}:{kind,title:$('#automation-draft-title').value,description:$('#automation-draft-body').value,attendees:target,start:$('#automation-calendar-start').value,end:$('#automation-calendar-end').value,timezone:$('#automation-calendar-timezone').value,location:$('#automation-calendar-location').value};await api('/api/integrations/drafts',{method:'POST',body:JSON.stringify(payload)});flash('Local draft created. Nothing was sent.');await loadAutomation();}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#automation-schedule-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Scheduling…');try{const kind=$('#automation-schedule-kind').value;const payload={schedule_kind:kind,name:$('#automation-schedule-name').value,goal:$('#automation-schedule-goal').value,workspace:$('#automation-workspace').value||null,run_at:$('#automation-run-at').value,every_seconds:Number($('#automation-every-seconds').value),max_runs:$('#automation-max-runs').value?Number($('#automation-max-runs').value):null};await api('/api/schedules',{method:'POST',body:JSON.stringify(payload)});flash('Reboot-safe task scheduled.');await loadAutomation();}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#automation-tick').addEventListener('click',async e=>{const b=e.currentTarget;setBusy(b,true,'Running…');try{const r=await api('/api/scheduler/tick',{method:'POST',body:JSON.stringify({worker:'local-ui',max_jobs:10})});flash(`Processed ${r.length} due task(s).`);await loadAutomation();}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#proposal-status-filter').addEventListener('change',loadProposals);$('#proposal-import-button').addEventListener('click',()=>$('#proposal-import-dialog').showModal());
  $('#proposal-import-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Validating…');try{const raw=$('#proposal-raw').value;const r=await api('/api/proposals/ingest',{method:'POST',body:JSON.stringify({provider:$('#proposal-provider').value,model_name:$('#proposal-model').value,raw_response:raw})});$('#proposal-import-dialog').close();flash(`Proposal batch retained with status: ${r.status}`);await loadProposals();await selectProposal(r.id);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#review-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Recording…');try{await api(`/api/proposal-items/${$('#review-item-id').value}/review`,{method:'POST',body:JSON.stringify({decision:$('#review-decision').value,reviewer:$('#review-reviewer').value,rationale:$('#review-rationale').value})});$('#review-dialog').close();flash('Review decision recorded.');await loadProposals();if(state.selectedProposal)await selectProposal(state.selectedProposal);}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#snapshot-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Rendering…');try{const roots=$('#snapshot-roots').value.split(',').map(x=>x.trim()).filter(Boolean);const d=await api('/api/graphs/snapshots',{method:'POST',body:JSON.stringify({name:$('#snapshot-name').value,root_claim_ids:roots,include_descendants:$('#snapshot-descendants').checked})});flash('Epistemic graph snapshot captured.');renderGraph(d,'snapshot');await loadGraphs();$('#graph-selector').value=`snapshot:${d.snapshot.id}`;}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#diff-form').addEventListener('submit',async e=>{e.preventDefault();const b=e.submitter;setBusy(b,true,'Comparing…');try{const d=await api('/api/graphs/diffs',{method:'POST',body:JSON.stringify({name:$('#diff-name').value,before_snapshot_id:$('#diff-before').value,after_snapshot_id:$('#diff-after').value})});flash('Collapse diff generated.');renderGraph(d,'diff');await loadGraphs();$('#graph-selector').value=`diff:${d.diff.id}`;}catch(err){flash(err.message,'error');}finally{setBusy(b,false);}});
  $('#graph-selector').addEventListener('change',e=>loadGraphSelection(e.target.value).catch(err=>flash(err.message,'error')));
  $('#audit-refresh').addEventListener('click',loadAudit);
}

wire(); health(); switchView('overview');
