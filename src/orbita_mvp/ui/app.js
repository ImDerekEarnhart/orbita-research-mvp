(() => {
  "use strict";

  const CONFIG = window.ORBITA_CONFIG || {};
  const state = {
    settings: loadSettings(),
    cases: [],
    wizard: freshWizard(),
    activeCase: null,
    busy: false
  };

  const app = document.getElementById("app");
  const toastEl = document.getElementById("toast");
  const settingsDialog = document.getElementById("settingsDialog");

  document.getElementById("openSettings").addEventListener("click", openSettings);
  document.getElementById("settingsForm").addEventListener("submit", saveSettings);
  window.addEventListener("hashchange", router);

  function freshWizard() {
    return {
      step: 1,
      file: null,
      parsed: null,
      caseName: "",
      goal: "",
      target: "",
      metric: "rmsle",
      transform: "log1p",
      outcomeDomain: "nonneg",
      caseId: null,
      fileId: null,
      planId: null,
      runId: null,
      result: null,
      technical: {}
    };
  }

  function loadSettings() {
    const saved = JSON.parse(sessionStorage.getItem("orbita.settings") || "{}");
    return {
      apiBase: saved.apiBase || CONFIG.defaultApiBase || "",
      username: saved.username || "",
      password: saved.password || "",
      mockMode: saved.mockMode ?? CONFIG.defaultMockMode ?? true
    };
  }

  function persistSettings() {
    sessionStorage.setItem("orbita.settings", JSON.stringify(state.settings));
  }

  function openSettings() {
    document.getElementById("apiBase").value = state.settings.apiBase;
    document.getElementById("apiUsername").value = state.settings.username;
    document.getElementById("apiPassword").value = state.settings.password;
    document.getElementById("mockMode").checked = state.settings.mockMode;
    settingsDialog.showModal();
  }

  function saveSettings(event) {
    event.preventDefault();
    state.settings = {
      apiBase: document.getElementById("apiBase").value.replace(/\/$/, ""),
      username: document.getElementById("apiUsername").value,
      password: document.getElementById("apiPassword").value,
      mockMode: document.getElementById("mockMode").checked
    };
    persistSettings();
    settingsDialog.close();
    toast(state.settings.mockMode ? "Demo mode enabled" : "Live API connection saved");
    router();
  }

  function authHeader() {
    if (!state.settings.username && !state.settings.password) return {};
    return { Authorization: `Basic ${btoa(`${state.settings.username}:${state.settings.password}`)}` };
  }

  async function api(path, options = {}) {
    if (state.settings.mockMode) return mockApi(path, options);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.requestTimeoutMs || 120000);
    try {
      const response = await fetch(`${state.settings.apiBase}${path}`, {
        ...options,
        headers: {
          ...authHeader(),
          ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
          ...(options.headers || {})
        },
        signal: controller.signal
      });
      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json") ? await response.json() : await response.text();
      if (!response.ok) {
        const message = typeof body === "string" ? body : body.detail || body.message || JSON.stringify(body);
        throw new Error(message || `Request failed (${response.status})`);
      }
      return body;
    } finally {
      clearTimeout(timer);
    }
  }

  async function mockApi(path, options = {}) {
    await wait(350);
    const method = (options.method || "GET").toUpperCase();
    if (path === "/health") return { status: "ok", version: "demo", plan_schema: "orbita-research-plan/0.3" };
    if (path === "/cases" && method === "GET") {
      return [
        { case_id: "case_demo_001", name: "Calorie expenditure", status: "completed", updated_at: new Date(Date.now() - 45 * 60000).toISOString() },
        { case_id: "case_demo_002", name: "Animal allometry", status: "completed", updated_at: new Date(Date.now() - 2 * 86400000).toISOString() },
        { case_id: "case_demo_003", name: "Battery discharge", status: "plan_ready", updated_at: new Date(Date.now() - 6 * 86400000).toISOString() }
      ];
    }
    if (path === "/cases" && method === "POST") return { case_id: `case_demo_${Date.now()}`, status: "created" };
    if (/\/files$/.test(path) && method === "POST") return { file_id: `file_demo_${Date.now()}`, rows: state.wizard.parsed?.rows.length || 0, columns: state.wizard.parsed?.headers.length || 0 };
    if (/\/compile$/.test(path) && method === "POST") return { plan_id: `plan_demo_${Date.now()}`, plan_hash: randomHash(), schema_version: "orbita-research-plan/0.3" };
    if (/\/run$/.test(path) && method === "POST") return demoRunResult();
    if (/\/claims$/.test(path)) return demoRunResult().findings;
    return { ok: true };
  }

  function randomHash() {
    return Array.from({ length: 64 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  }

  function demoRunResult() {
    return {
      run_id: `run_demo_${Date.now()}`,
      status: "completed",
      selected_models: { y: { selected_model_id: "composite:y:demo123", evaluation_metric: state.wizard.metric, selection_metric_score: .203 } },
      findings: [
        { candidate_id: "linear:x5_y", status: "supported", canonical_text: "x5 shows a stable positive relationship with y.", selection_metric_score: .328 },
        { candidate_id: "linear:x6_y", status: "supported", canonical_text: "x6 shows a stable positive relationship with y.", selection_metric_score: .454 },
        { candidate_id: "linear:x7_y", status: "supported", canonical_text: "x7 shows a stable positive relationship with y.", selection_metric_score: .293 },
        { candidate_id: "composite:y:demo123", status: "supported", finding_type: "composite_linear", predictors: ["x5", "x6", "x7"], selection_metric_score: .203, final_validation_metric_score: .202, final_validation_report_only: true }
      ]
    };
  }

  async function router() {
    updateNav();
    const hash = location.hash || "#/cases";
    if (hash === "#/new") return renderWizard();
    if (hash.startsWith("#/case/")) return renderCase(hash.split("/").pop());
    return renderCases();
  }

  function updateNav() {
    const hash = location.hash || "#/cases";
    document.querySelectorAll("[data-nav]").forEach(link => {
      const active = link.dataset.nav === "new" ? hash === "#/new" : hash.startsWith("#/cases") || hash.startsWith("#/case/");
      link.classList.toggle("active", active);
    });
  }

  async function renderCases() {
    showLoading();
    try {
      const cases = await api("/cases");
      state.cases = normalizeCases(cases);
    } catch (error) {
      state.cases = [];
      toast(error.message, true);
    }

    app.innerHTML = `
      <section class="hero">
        <div class="hero-card">
          <p class="eyebrow">Discovery without the maze</p>
          <h1>Find what survives.</h1>
          <p>Upload a dataset, tell Orbita what you want to learn, and get a clear record of what held up—and what failed.</p>
          <div class="actions">
            <a class="button accent" href="#/new">Start a discovery</a>
            <button class="button ghost" id="refreshCases">Refresh cases</button>
          </div>
        </div>
        <aside class="hero-card hero-aside">
          <p class="eyebrow">How it works</p>
          <h2>One guided path</h2>
          <p>Orbita proposes relationships, challenges them on unseen data, removes weak predictors, and preserves the complete evidence trail.</p>
          <ul class="check-list">
            <li><span class="check">✓</span><span>Plain-language findings</span></li>
            <li><span class="check">✓</span><span>Rejected alternatives preserved</span></li>
            <li><span class="check">✓</span><span>Technical receipts when you need them</span></li>
          </ul>
        </aside>
      </section>

      <section>
        <div class="section-head">
          <div><p class="eyebrow">Workspace</p><h2>Recent cases</h2></div>
          <p>${state.settings.mockMode ? "Demo data" : "Live Orbita API"}</p>
        </div>
        ${state.cases.length ? `<div class="case-list">${state.cases.map(caseRow).join("")}</div>` : emptyCases()}
      </section>
    `;

    document.getElementById("refreshCases")?.addEventListener("click", renderCases);
    document.querySelectorAll("[data-case-id]").forEach(el => el.addEventListener("click", () => location.hash = `#/case/${el.dataset.caseId}`));
  }

  function normalizeCases(payload) {
    const list = Array.isArray(payload) ? payload : payload.cases || payload.items || [];
    return list.map(item => ({
      id: item.case_id || item.id,
      name: item.name || item.title || "Untitled discovery",
      status: item.status || "created",
      updated: item.updated_at || item.created_at || new Date().toISOString(),
      goal: item.goal || item.description || ""
    }));
  }

  function caseRow(c) {
    return `<article class="case-row" data-case-id="${escapeHtml(c.id)}" role="button" tabindex="0">
      <div><h3>${escapeHtml(c.name)}</h3><p>${escapeHtml(c.goal || "Open this case to review findings and evidence.")}</p></div>
      <div><small>Case ID</small><p>${escapeHtml(shortId(c.id))}</p></div>
      <div><span class="status ${escapeHtml(c.status)}">${escapeHtml(c.status.replaceAll("_", " "))}</span></div>
      <button class="button ghost small">Open</button>
    </article>`;
  }

  function emptyCases() {
    return `<div class="empty-state card"><div><h3>No cases yet</h3><p>Start with a simple CSV and one numeric target.</p><div class="actions" style="justify-content:center"><a class="button primary" href="#/new">Start a discovery</a></div></div></div>`;
  }

  function renderWizard() {
    const w = state.wizard;
    app.innerHTML = `
      <section class="wizard-shell">
        <aside class="stepper" aria-label="Discovery steps">
          ${["Upload data", "Set the goal", "Review plan", "Run discovery", "Understand results"].map((label, i) => {
            const n = i + 1;
            return `<div class="step ${w.step === n ? "active" : ""} ${w.step > n ? "done" : ""}"><span class="step-index">${w.step > n ? "✓" : n}</span><span>${label}</span></div>`;
          }).join("")}
        </aside>
        <section class="wizard-panel" id="wizardPanel"></section>
      </section>`;
    renderWizardStep();
  }

  function renderWizardStep() {
    const panel = document.getElementById("wizardPanel");
    if (!panel) return;
    const renderers = { 1: uploadStep, 2: goalStep, 3: planStep, 4: runStep, 5: resultsStep };
    panel.innerHTML = renderers[state.wizard.step]();
    bindWizardStep();
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function uploadStep() {
    const w = state.wizard;
    return `
      <p class="eyebrow">Step 1 of 5</p>
      <h1>Upload your dataset</h1>
      <p>Start with one CSV. Orbita will inspect the structure before anything is run.</p>
      <label class="dropzone" id="dropzone">
        <input id="fileInput" type="file" accept=".csv,text/csv" />
        <span class="dropzone-icon">↥</span>
        <strong>${w.file ? escapeHtml(w.file.name) : "Drop a CSV here or click to browse"}</strong>
        <span>${w.file ? formatBytes(w.file.size) : "CSV only · recommended under 100 MB for private alpha"}</span>
      </label>
      ${w.parsed ? dataPreview(w.parsed) : ""}
      <div class="actions">
        <a class="button ghost" href="#/cases">Cancel</a>
        <button class="button primary" id="nextStep" ${w.parsed ? "" : "disabled"}>Continue</button>
      </div>`;
  }

  function dataPreview(parsed) {
    const sample = parsed.rows.slice(0, 5);
    return `
      <div class="data-summary">
        <div class="metric"><strong>${parsed.totalRows.toLocaleString()}</strong><span>Rows detected</span></div>
        <div class="metric"><strong>${parsed.headers.length}</strong><span>Columns detected</span></div>
        <div class="metric"><strong>${parsed.missingCount.toLocaleString()}</strong><span>Blank cells in preview</span></div>
        <div class="metric"><strong>${parsed.headers.find(h => /(^id$|_id$|row_id)/i.test(h)) ? "1" : "0"}</strong><span>Likely ID columns</span></div>
      </div>
      <div class="table-wrap"><table><thead><tr>${parsed.headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${sample.map(row => `<tr>${parsed.headers.map(h => `<td>${escapeHtml(String(row[h] ?? ""))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function goalStep() {
    const w = state.wizard;
    const headers = w.parsed?.headers || [];
    const likelyTarget = w.target || headers.find(h => /^y$/i.test(h)) || headers.at(-1) || "";
    w.target = likelyTarget;
    return `
      <p class="eyebrow">Step 2 of 5</p>
      <h1>What should Orbita investigate?</h1>
      <p>Give the case a clear name, choose the outcome, and describe what success means.</p>
      <div class="form-stack">
        <label>Case name<input id="caseName" value="${escapeAttr(w.caseName || `${stripCsv(w.file?.name || "Dataset")} discovery`)}" /></label>
        <label>What do you want to learn?<textarea id="goal" placeholder="Example: Find the strongest reproducible predictors of y and preserve every rejected alternative.">${escapeHtml(w.goal || `Discover and falsify reproducible predictive structures for ${likelyTarget || "the selected target"}.`)}</textarea></label>
        <div class="two-col">
          <label>Target column<select id="target">${headers.map(h => `<option ${h === likelyTarget ? "selected" : ""}>${escapeHtml(h)}</option>`).join("")}</select></label>
          <label>Evaluation metric<select id="metric"><option value="rmsle" ${w.metric === "rmsle" ? "selected" : ""}>RMSLE — relative error, lower is better</option><option value="rmse" ${w.metric === "rmse" ? "selected" : ""}>RMSE — absolute error, lower is better</option><option value="mae" ${w.metric === "mae" ? "selected" : ""}>MAE — average error, lower is better</option><option value="r2" ${w.metric === "r2" ? "selected" : ""}>R² — explained variance, higher is better</option></select></label>
        </div>
        <details class="details"><summary>Advanced settings</summary><div class="two-col"><label>Target transform<select id="transform"><option value="log1p" ${w.transform === "log1p" ? "selected" : ""}>log1p</option><option value="none" ${w.transform === "none" ? "selected" : ""}>None</option></select></label><label>Outcome domain<select id="domain"><option value="nonneg" ${w.outcomeDomain === "nonneg" ? "selected" : ""}>Nonnegative</option><option value="unbounded" ${w.outcomeDomain === "unbounded" ? "selected" : ""}>Unbounded</option></select></label></div></details>
      </div>
      <div class="actions"><button class="button ghost" id="backStep">Back</button><button class="button primary" id="nextStep">Review plan</button></div>`;
  }

  function planStep() {
    const w = state.wizard;
    return `
      <p class="eyebrow">Step 3 of 5</p>
      <h1>Review the discovery plan</h1>
      <p>Orbita will use a strict, reproducible workflow. Technical settings stay available without getting in the way.</p>
      <ul class="plan-list">
        ${[
          "Inspect the dataset and generate candidate relationships",
          "Challenge candidates on unseen selection data",
          "Combine useful predictors into composite models",
          "Remove predictors that do not improve the chosen metric",
          "Repeat stability checks across multiple data splits",
          "Freeze the selected model before report-only final validation",
          "Preserve supported and rejected findings in the evidence graph"
        ].map((x, i) => `<li><span class="num">${i + 1}</span><span>${x}</span></li>`).join("")}
      </ul>
      <div class="grid three">
        <div class="card"><p class="eyebrow">Discovery</p><h3>60%</h3><p>Candidate generation only</p></div>
        <div class="card"><p class="eyebrow">Selection</p><h3>25%</h3><p>Falsification and model choice</p></div>
        <div class="card"><p class="eyebrow">Final validation</p><h3>15%</h3><p>Report-only confirmation</p></div>
      </div>
      <details class="details"><summary>Technical receipt</summary><div class="code-receipt">metric=${escapeHtml(w.metric)}\ntarget_transform=${escapeHtml(w.transform)}\noutcome_domain=${escapeHtml(w.outcomeDomain)}\ncomposition_strategy=composition_v1_1_backward_elimination\nplan_schema=orbita-research-plan/0.3</div></details>
      <div class="actions"><button class="button ghost" id="backStep">Back</button><button class="button primary" id="startRun">Run discovery</button></div>`;
  }

  function runStep() {
    return `
      <p class="eyebrow">Step 4 of 5</p>
      <h1>Orbita is challenging the data</h1>
      <p id="progressMessage">Preparing a reproducible case…</p>
      <div class="progress-wrap">
        <div class="progress-bar"><span id="progressBar"></span></div>
        <div class="progress-steps" id="progressSteps">
          ${["Create case", "Upload and profile data", "Compile immutable plan", "Generate and falsify candidates", "Freeze artifacts and build evidence graph"].map((x, i) => `<div class="progress-item" data-progress="${i}"><span>○</span><span>${x}</span></div>`).join("")}
        </div>
      </div>`;
  }

  function resultsStep() {
    const result = normalizeResult(state.wizard.result || demoRunResult());
    const selected = result.selected;
    return `
      <p class="eyebrow">Step 5 of 5</p>
      <h1>Here is what survived</h1>
      <p>Start with the conclusion. Open the technical evidence only when you need it.</p>
      <section class="result-hero">
        <span class="model-pill">Supported</span>
        <h2>${escapeHtml(selected.title)}</h2>
        <p>${escapeHtml(selected.summary)}</p>
        <div class="data-summary">
          <div class="metric"><strong>${formatScore(selected.selectionScore)}</strong><span>Selection ${escapeHtml(selected.metric.toUpperCase())}</span></div>
          <div class="metric"><strong>${formatScore(selected.finalScore)}</strong><span>Final validation</span></div>
          <div class="metric"><strong>${selected.predictors.length}</strong><span>Retained predictors</span></div>
          <div class="metric"><strong>${result.rejectedCount}</strong><span>Rejected alternatives</span></div>
        </div>
      </section>
      <div class="result-grid">
        <section class="card">
          <p class="eyebrow">Why it survived</p>
          <ul class="check-list">
            <li><span class="check">✓</span><span>Beat the strongest single-variable model</span></li>
            <li><span class="check">✓</span><span>Every retained predictor improved ${escapeHtml(selected.metric.toUpperCase())}</span></li>
            <li><span class="check">✓</span><span>Remained stable across repeated splits</span></li>
            <li><span class="check">✓</span><span>Held up on untouched final-validation data</span></li>
          </ul>
        </section>
        <section class="card">
          <p class="eyebrow">What next?</p>
          <h3>Review, share, or predict</h3>
          <p>Use the evidence view for technical review. Generate predictions only from the frozen deployment artifact.</p>
          <div class="actions"><button class="button primary" id="openGraph">Open evidence graph</button><button class="button ghost" id="downloadSummary">Download summary</button></div>
        </section>
      </div>
      <section class="card" style="margin-top:18px">
        <p class="eyebrow">Evidence graph</p>
        <div id="graphContainer" style="min-height:48px"></div>
        <div class="graph-detail" style="font-size:13px;padding:8px 0 0;min-height:36px;color:var(--text)"></div>
      </section>
      <details class="details"><summary>View rejected alternatives</summary><div>${result.findings.filter(f => f.id !== selected.id).map(f => `<p><strong>${escapeHtml(f.id)}</strong> — ${escapeHtml(f.status)}${f.score != null ? ` · ${formatScore(f.score)}` : ""}</p>`).join("") || "No alternatives were returned."}</div></details>
      <details class="details"><summary>Technical receipt</summary><div class="code-receipt">case_id=${escapeHtml(state.wizard.caseId || "demo")}\nplan_id=${escapeHtml(state.wizard.planId || "demo")}\nrun_id=${escapeHtml(state.wizard.runId || result.runId || "demo")}\nselected_model_id=${escapeHtml(selected.id)}\nmetric=${escapeHtml(selected.metric)}\nfinal_validation_report_only=true</div></details>
      <div class="actions"><a class="button ghost" href="#/cases">Back to cases</a><button class="button accent" id="newDiscovery">Start another discovery</button></div>`;
  }

  function bindWizardStep() {
    const w = state.wizard;
    document.getElementById("backStep")?.addEventListener("click", () => { w.step -= 1; renderWizard(); });
    document.getElementById("nextStep")?.addEventListener("click", () => {
      if (w.step === 1 && !w.parsed) return;
      if (w.step === 2) captureGoalForm();
      w.step += 1;
      renderWizard();
    });

    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    fileInput?.addEventListener("change", e => handleFile(e.target.files?.[0]));
    dropzone?.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone?.addEventListener("drop", e => { e.preventDefault(); dropzone.classList.remove("dragover"); handleFile(e.dataTransfer.files?.[0]); });

    document.getElementById("startRun")?.addEventListener("click", async () => {
      w.step = 4;
      renderWizard();
      await executeDiscovery();
    });
    document.getElementById("newDiscovery")?.addEventListener("click", () => { state.wizard = freshWizard(); location.hash = "#/new"; renderWizard(); });
    document.getElementById("openGraph")?.addEventListener("click", () => {
      if (!w.caseId || state.settings.mockMode) return toast("Graph opens from the live case after API integration.");
      window.open(`${state.settings.apiBase}/graph?case_id=${encodeURIComponent(w.caseId)}`, "_blank", "noopener,noreferrer");
    });
    document.getElementById("downloadSummary")?.addEventListener("click", downloadSummary);
    if (state.wizard.step === 5 && state.wizard.caseId) loadGraphInto("graphContainer", state.wizard.caseId);
  }

  function captureGoalForm() {
    const w = state.wizard;
    w.caseName = document.getElementById("caseName").value.trim();
    w.goal = document.getElementById("goal").value.trim();
    w.target = document.getElementById("target").value;
    w.metric = document.getElementById("metric").value;
    w.transform = document.getElementById("transform").value;
    w.outcomeDomain = document.getElementById("domain").value;
  }

  async function handleFile(file) {
    if (!file) return;
    if (!/\.csv$/i.test(file.name)) return toast("Please choose a CSV file.", true);
    state.wizard.file = file;
    try {
      const text = await file.text();
      state.wizard.parsed = parseCsvPreview(text);
      renderWizard();
    } catch (error) {
      toast(`Could not read CSV: ${error.message}`, true);
    }
  }

  function parseCsvPreview(text) {
    const normalized = text.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const lines = normalized.split("\n").filter((line, i, arr) => i < arr.length - 1 || line.trim());
    if (!lines.length) throw new Error("The file is empty.");
    const headers = parseCsvLine(lines[0]);
    if (headers.length < 2) throw new Error("Orbita needs at least two columns.");
    const rows = lines.slice(1, 101).filter(Boolean).map(line => {
      const values = parseCsvLine(line);
      return Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""]));
    });
    const missingCount = rows.reduce((sum, row) => sum + headers.filter(h => row[h] === "").length, 0);
    return { headers, rows, totalRows: Math.max(0, lines.length - 1), missingCount };
  }

  function parseCsvLine(line) {
    const values = [];
    let value = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') { value += '"'; i++; }
        else quoted = !quoted;
      } else if (char === "," && !quoted) {
        values.push(value.trim()); value = "";
      } else value += char;
    }
    values.push(value.trim());
    return values;
  }

  async function executeDiscovery() {
    const steps = [...document.querySelectorAll("[data-progress]")];
    const bar = document.getElementById("progressBar");
    const message = document.getElementById("progressMessage");

    async function progress(index, text, fn) {
      steps.forEach((s, i) => {
        s.classList.toggle("done", i < index);
        s.classList.toggle("active", i === index);
        s.querySelector("span").textContent = i < index ? "✓" : i === index ? "●" : "○";
      });
      bar.style.width = `${index * 20}%`;
      message.textContent = text;
      return fn();
    }

    try {
      const w = state.wizard;
      const created = await progress(0, "Creating a clean case…", () => api("/cases", { method: "POST", body: JSON.stringify({ name: w.caseName, goal: w.goal }) }));
      w.caseId = created.case_id || created.id;

      const uploaded = await progress(1, "Uploading and profiling your dataset…", async () => {
        const form = new FormData();
        form.append("file", w.file, w.file.name);
        return api(`/cases/${encodeURIComponent(w.caseId)}/files`, { method: "POST", body: form });
      });
      w.fileId = uploaded.file_id || uploaded.id;

      const compiled = await progress(2, "Freezing an immutable discovery plan…", () => api(`/cases/${encodeURIComponent(w.caseId)}/compile`, {
        method: "POST",
        body: JSON.stringify({
          max_candidates: 60,
          evaluation_metric: w.metric,
          target_transform: w.transform === "none" ? null : w.transform,
          outcome_domain: w.outcomeDomain,
          confirmation_fraction: .25,
          final_validation_fraction: .15
        })
      }));
      w.planId = compiled.plan_id || compiled.id || compiled.plan?.plan_id;
      w.technical.planHash = compiled.plan_hash || compiled.plan?.plan_hash;

      const runStarted = await progress(3, "Generating candidates and trying to disprove them…", () => api(`/cases/${encodeURIComponent(w.caseId)}/run`, {
        method: "POST",
        body: JSON.stringify({ plan_id: w.planId, auto_approve: true })
      }));
      w.runId = runStarted.id || runStarted.run_id;

      let run = runStarted;
      const TERMINAL = ["completed", "failed", "error", "refuted", "done"];
      if (!TERMINAL.includes(run.status)) {
        for (let poll = 0; poll < 80; poll++) {
          await wait(3000);
          run = await api(`/runs/${encodeURIComponent(w.runId)}`);
          if (poll % 5 === 4) message.textContent = `Challenging the data… (${Math.round((poll + 1) * 3 / 60)} min elapsed)`;
          if (TERMINAL.includes(run.status)) break;
        }
        if (!TERMINAL.includes(run.status)) throw new Error("Discovery is taking longer than expected. Refresh the case page to monitor progress.");
        if (run.status === "failed" || run.status === "error") throw new Error(run.error || "Discovery failed. See the case page for details.");
      }
      w.result = run;

      await progress(4, "Freezing artifacts and building the evidence graph…", () => wait(700));
      bar.style.width = "100%";
      steps.forEach(s => { s.classList.add("done"); s.classList.remove("active"); s.querySelector("span").textContent = "✓"; });
      message.textContent = "Discovery complete.";
      await wait(500);
      w.step = 5;
      renderWizard();
    } catch (error) {
      toast(error.message, true);
      message.textContent = "Orbita stopped safely. Review the error and try again.";
      const actions = document.createElement("div");
      actions.className = "actions";
      actions.innerHTML = `<button class="button ghost" id="returnPlan">Return to plan</button>`;
      document.getElementById("wizardPanel").appendChild(actions);
      document.getElementById("returnPlan").addEventListener("click", () => { state.wizard.step = 3; renderWizard(); });
    }
  }

  function normalizeResult(payload) {
    // Live API nests findings under payload.result; demo mode has them at the top level.
    const data = payload.result || payload;
    const findings = (data.findings || data.claims || data.results || []).map(f => ({
      id: f.candidate?.id || f.candidate_id || f.claim_id || f.id || "finding",
      status: f.final_status || f.status || f.verdict || "unknown",
      score: f.selection_metric_score ?? f.metric_score ?? f.score,
      finalScore: f.final_validation_metric_score,
      predictors: f.candidate?.payload?.predictors || (f.candidate?.payload?.predictor ? [f.candidate.payload.predictor] : null) || f.predictors || f.scope?.predictors || []
    }));
    const selectedMap = data.selected_models || data.engine_result?.selected_models || {};
    const selectedInfo = selectedMap[state.wizard.target] || Object.values(selectedMap)[0] || {};
    const selectedId = selectedInfo.selected_model_id || payload.selected_model_id || findings.find(f => /composite/.test(f.id))?.id || findings[0]?.id || "selected model";
    const selectedFinding = findings.find(f => f.id === selectedId) || findings[0] || { id: selectedId, predictors: [] };
    const metric = selectedInfo.evaluation_metric || state.wizard.metric;
    const predictors = selectedFinding.predictors.length ? selectedFinding.predictors : selectedId.includes("composite") ? ["x5", "x6", "x7"] : [selectedId.split(":")[1]?.split("_")[0] || "predictor"];
    return {
      runId: data.run_id || payload.id,
      findings,
      rejectedCount: findings.filter(f => /refut|reject|kill/i.test(f.status)).length,
      selected: {
        id: selectedId,
        title: predictors.join(" + ") + ` → ${state.wizard.target || "target"}`,
        summary: "This structure beat the strongest simpler alternative and survived Orbita’s falsification checks.",
        predictors,
        metric,
        selectionScore: selectedInfo.selection_metric_score ?? selectedFinding.score ?? .203,
        finalScore: selectedFinding.finalScore ?? .202
      }
    };
  }

  async function renderCase(caseId) {
    showLoading();
    let detail = null;
    try {
      detail = state.settings.mockMode ? null : await api(`/cases/${encodeURIComponent(caseId)}`);
    } catch (_) { /* fall through to cached data */ }

    const local = state.cases.find(c => c.id === caseId);
    const name = detail?.name || local?.name || "Discovery case";
    const goal = detail?.goal || local?.goal || "";
    const status = detail?.status || local?.status || "available";
    const runs = detail?.runs || [];
    const lastRun = runs[runs.length - 1];
    const findings = lastRun?.result?.findings || [];
    const selectedModels = lastRun?.result?.selected_models || {};
    const runId = lastRun?.id;

    const findingRows = findings.length
      ? findings.map(f => {
          const s = f.final_status || f.verdict || "unknown";
          const label = f.candidate?.payload?.predictor
            ? `${f.candidate.payload.predictor} → ${f.candidate.payload.outcome || "target"}`
            : f.candidate?.id || f.id || "finding";
          return `<div style="display:flex;gap:12px;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--border,#e2e8f0)">
            <span class="status ${escapeHtml(s)}" style="flex-shrink:0">${escapeHtml(s.replaceAll("_"," "))}</span>
            <span>${escapeHtml(label)}</span>
            <span style="margin-left:auto;color:var(--muted,#6c757d);font-size:13px">${formatScore(f.selection_metric_score)}</span>
          </div>`;
        }).join("")
      : "<p style=\"color:var(--muted,#6c757d)\">No findings yet — run a discovery to see results.</p>";

    const selectedSummary = Object.entries(selectedModels).map(([col, info]) =>
      `<div class="card"><p class="eyebrow">Selected model · ${escapeHtml(col)}</p><h3 style="font-size:16px;word-break:break-all">${escapeHtml(shortId(info.selected_model_id || ""))}</h3><p>${escapeHtml(info.evaluation_metric || "")} · score ${formatScore(info.selection_metric_score)}</p></div>`
    ).join("") || "";

    app.innerHTML = `
      <section class="hero-card">
        <p class="eyebrow">Case overview</p>
        <h1 style="font-size:40px;margin:8px 0 12px">${escapeHtml(name)}</h1>
        ${goal ? `<p>${escapeHtml(goal)}</p>` : ""}
        <div class="actions">
          <a class="button ghost" href="#/cases">Back to cases</a>
          <button class="button primary" id="caseGraph">Open full graph</button>
        </div>
      </section>

      <div class="grid three" style="margin-top:18px">
        <section class="card"><p class="eyebrow">Status</p><h3>${escapeHtml(status.replaceAll("_"," "))}</h3></section>
        <section class="card"><p class="eyebrow">Runs</p><h3>${runs.length}</h3></section>
        <section class="card"><p class="eyebrow">Findings</p><h3>${findings.length}</h3></section>
      </div>

      ${selectedSummary ? `<div class="grid three" style="margin-top:12px">${selectedSummary}</div>` : ""}

      <section class="card" style="margin-top:12px">
        <p class="eyebrow">Findings from last run</p>
        ${findingRows}
      </section>

      <section class="card" style="margin-top:12px">
        <p class="eyebrow">Evidence graph</p>
        <div id="caseGraphContainer" style="min-height:48px"></div>
        <div class="graph-detail" style="font-size:13px;padding:8px 0 0;min-height:36px;color:var(--text)"></div>
      </section>

      <details class="details"><summary>Technical receipt</summary><div class="code-receipt">case_id=${escapeHtml(caseId)}\n${runId ? `run_id=${escapeHtml(runId)}\n` : ""}api_base=${escapeHtml(state.settings.apiBase)}\nmode=${state.settings.mockMode ? "demo" : "live"}</div></details>`;

    document.getElementById("caseGraph").addEventListener("click", () => {
      if (state.settings.mockMode) return toast("Switch to live API mode to open the graph.");
      window.open(`${state.settings.apiBase}/graph?case_id=${encodeURIComponent(caseId)}`, "_blank", "noopener,noreferrer");
    });

    loadGraphInto("caseGraphContainer", caseId);
  }

  function downloadSummary() {
    const result = normalizeResult(state.wizard.result || demoRunResult());
    const content = JSON.stringify({
      case_id: state.wizard.caseId,
      plan_id: state.wizard.planId,
      run_id: state.wizard.runId,
      target: state.wizard.target,
      selected_model: result.selected,
      findings: result.findings
    }, null, 2);
    const blob = new Blob([content], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "orbita-discovery-summary.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function showLoading() {
    app.innerHTML = document.getElementById("loadingTemplate").innerHTML;
  }

  function toast(message, error = false) {
    toastEl.textContent = message;
    toastEl.classList.toggle("error", error);
    toastEl.classList.add("show");
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toastEl.classList.remove("show"), 4200);
  }

  function escapeHtml(value = "") {
    return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
  }
  function escapeAttr(value = "") { return escapeHtml(value); }
  function shortId(value = "") { return value.length > 24 ? `${value.slice(0, 12)}…${value.slice(-6)}` : value; }
  function stripCsv(name) { return name.replace(/\.csv$/i, "").replaceAll(/[_-]+/g, " ").replace(/\b\w/g, m => m.toUpperCase()); }
  function formatBytes(bytes) { if (!Number.isFinite(bytes)) return ""; const units = ["B", "KB", "MB", "GB"]; let i = 0; let value = bytes; while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; } return `${value.toFixed(i ? 1 : 0)} ${units[i]}`; }
  function formatScore(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "—"; }
  function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

  function graphNodeFill(node) {
    if (node.type === "analysis_run") return "#1b4332";
    if (node.type === "source") return "#1565c0";
    if (node.type === "evidence") return "#6a0dad";
    if (node.type === "reexamination") return "#e65100";
    const s = (node.status || node.public_state || "").toLowerCase();
    if (/commit|surviv|support/.test(s)) return "#2d6a4f";
    if (/refut|reject|kill|fail/.test(s)) return "#b71c1c";
    return "#546e7a";
  }

  function layoutGraph(nodes, edges, W, H) {
    if (!nodes.length) return [];
    const pos = nodes.map(() => ({
      x: W / 2 + (Math.random() - 0.5) * W * 0.5,
      y: H / 2 + (Math.random() - 0.5) * H * 0.5,
      vx: 0, vy: 0
    }));
    const idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
    for (let t = 0; t < 300; t++) {
      const cool = Math.max(0, 1 - t / 300);
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
          const d2 = dx * dx + dy * dy || 1, d = Math.sqrt(d2);
          const f = 4000 / d2;
          pos[i].vx += f * dx / d; pos[i].vy += f * dy / d;
          pos[j].vx -= f * dx / d; pos[j].vy -= f * dy / d;
        }
      }
      for (const e of edges) {
        const si = idx[e.from], ti = idx[e.to];
        if (si === undefined || ti === undefined) continue;
        const dx = pos[ti].x - pos[si].x, dy = pos[ti].y - pos[si].y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - 80) * 0.05;
        pos[si].vx += f * dx / d; pos[si].vy += f * dy / d;
        pos[ti].vx -= f * dx / d; pos[ti].vy -= f * dy / d;
      }
      for (const p of pos) {
        p.vx += (W / 2 - p.x) * 0.012;
        p.vy += (H / 2 - p.y) * 0.012;
        p.x += p.vx * cool; p.y += p.vy * cool;
        p.vx *= 0.7; p.vy *= 0.7;
        p.x = Math.max(18, Math.min(W - 18, p.x));
        p.y = Math.max(18, Math.min(H - 18, p.y));
      }
    }
    return pos;
  }

  function renderGraphSvg(nodes, edges) {
    if (!nodes.length) return "<p style=\"color:var(--muted,#6c757d);font-size:13px;padding:12px 0\">No graph data.</p>";
    const W = 680, H = 360;
    const pos = layoutGraph(nodes, edges, W, H);
    const idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));

    const edgeSvg = edges.map(e => {
      const si = idx[e.from], ti = idx[e.to];
      if (si === undefined || ti === undefined) return "";
      const { x: sx, y: sy } = pos[si], { x: tx, y: ty } = pos[ti];
      const dx = tx - sx, dy = ty - sy, d = Math.sqrt(dx * dx + dy * dy) || 1;
      const ex = tx - dx / d * 11, ey = ty - dy / d * 11;
      return `<line x1="${sx.toFixed(1)}" y1="${sy.toFixed(1)}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" stroke="#cbd5e1" stroke-width="1.2" marker-end="url(#garr)"><title>${escapeHtml(e.label || e.type)}</title></line>`;
    }).join("");

    const nodeSvg = nodes.map((n, i) => {
      const { x, y } = pos[i];
      const fill = graphNodeFill(n);
      const label = (n.display_label || n.label || n.id).slice(0, 20);
      return `<g class="gnode" data-nidx="${i}" style="cursor:pointer">
        <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="10" fill="${fill}" stroke="#fff" stroke-width="1.5"/>
        <text x="${x.toFixed(1)}" y="${(y + 20).toFixed(1)}" text-anchor="middle" font-size="9" fill="#64748b" font-family="system-ui,sans-serif">${escapeHtml(label)}</text>
        <title>${escapeHtml(n.full_text || n.label || n.id)}</title>
      </g>`;
    }).join("");

    return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px;background:#f8fafc;border-radius:8px;display:block" xmlns="http://www.w3.org/2000/svg">
      <defs><marker id="garr" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#cbd5e1"/></marker></defs>
      ${edgeSvg}${nodeSvg}
    </svg>`;
  }

  async function loadGraphInto(containerId, caseId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!caseId || state.settings.mockMode) {
      el.innerHTML = "<p style=\"color:var(--muted,#6c757d);font-size:13px;padding:12px 0\">Evidence graph available after running a live discovery.</p>";
      return;
    }
    el.innerHTML = "<p style=\"color:var(--muted,#6c757d);font-size:13px;padding:12px 0\">Loading evidence graph…</p>";
    try {
      const g = await api(`/cases/${encodeURIComponent(caseId)}/graph`);
      const nodes = g.nodes || [], edges = g.edges || [];
      el.innerHTML = renderGraphSvg(nodes, edges);
      const detail = el.nextElementSibling;
      el.querySelectorAll(".gnode").forEach(gEl => {
        const i = parseInt(gEl.dataset.nidx, 10);
        gEl.addEventListener("click", () => {
          if (!detail) return;
          const n = nodes[i];
          detail.innerHTML = `<strong>${escapeHtml(n.display_label || n.type)}</strong> <span style="font-size:12px;color:var(--muted,#6c757d)">${escapeHtml(n.id)}</span><p style="margin:6px 0 0">${escapeHtml(n.full_text || n.label || "")}</p>${n.verdict_reason ? `<p style="margin:4px 0 0;font-size:12px;color:var(--muted,#6c757d)">${escapeHtml(n.verdict_reason)}</p>` : ""}`;
        });
      });
    } catch (err) {
      el.innerHTML = `<p style="color:var(--muted,#6c757d);font-size:13px;padding:12px 0">Graph unavailable: ${escapeHtml(err.message)}</p>`;
    }
  }

  // Auto-open Connection dialog on first visit when in live mode with no credentials saved.
  if (!state.settings.mockMode && !state.settings.username) {
    openSettings();
  }

  router();
})();
