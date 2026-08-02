/* Research Control Plane SPA — Phase 12.2 (CFD Data Acquisition) */
const PAGES = [
  ["overview", "Overview"],
  ["new-run", "New Run"],
  ["data-acquisition", "Data Acquisition"],
  ["source-target", "Source vs Target"],
  ["data-import", "Data Import"],
  ["datasets", "Datasets"],
  ["dataset-detail", "Dataset Detail"],
  ["runs", "Runs"],
  ["run-detail", "Run Detail"],
  ["alpha-miner", "Alpha Miner"],
  ["wfo", "WFO"],
  ["registry", "Candidate Registry"],
  ["portfolio", "Portfolio"],
  ["vault", "Vault"],
  ["runtime", "Shadow / Paper"],
  ["data-quality", "Data Quality"],
  ["health", "System Health"],
  ["artifacts", "Artifacts"],
  ["logs", "Logs"],
];

const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"]);
const ACTIVE = new Set(["CREATED", "QUEUED", "STARTING", "RUNNING", "CANCEL_REQUESTED"]);

const state = {
  page: location.hash.replace(/^#\/?/, "").split("?")[0] || "overview",
  selectedRunId: localStorage.getItem("cp_run_id") || "",
  selectedDatasetId: localStorage.getItem("cp_dataset_id") || "",
  runFilter: localStorage.getItem("cp_run_filter") || "DEFAULT",
  sortKey: "candidate_id",
  sortDir: 1,
  filter: "",
  sse: null,
  catalog: [],
  constraints: null,
  alphaRefreshTimer: null,
  alphaRefreshInFlight: false,
  alphaRunActive: false,
  continueSearchDrafts: {},
};

const CS_DRAFT_DEFAULTS = { runtime: 10800, generated: 0, full_wfo: 0 };
const CS_DRAFT_STORAGE_KEY = "cp_continue_search_drafts";

function _loadContinueSearchDraftsFromStorage() {
  try {
    const raw = sessionStorage.getItem(CS_DRAFT_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}
function _persistContinueSearchDrafts() {
  try {
    sessionStorage.setItem(CS_DRAFT_STORAGE_KEY, JSON.stringify(state.continueSearchDrafts || {}));
  } catch {}
}
state.continueSearchDrafts = _loadContinueSearchDraftsFromStorage();

function getContinueSearchDraft(runId) {
  if (!runId) {
    return { ...CS_DRAFT_DEFAULTS };
  }
  const existing = (state.continueSearchDrafts || {})[runId];
  if (existing && typeof existing === "object") {
    return {
      runtime: Number.isFinite(Number(existing.runtime)) ? Number(existing.runtime) : CS_DRAFT_DEFAULTS.runtime,
      generated: Number.isFinite(Number(existing.generated)) ? Number(existing.generated) : CS_DRAFT_DEFAULTS.generated,
      full_wfo: Number.isFinite(Number(existing.full_wfo)) ? Number(existing.full_wfo) : CS_DRAFT_DEFAULTS.full_wfo,
    };
  }
  return { ...CS_DRAFT_DEFAULTS };
}
function saveContinueSearchDraft(runId, patch) {
  if (!runId) return getContinueSearchDraft(runId);
  const cur = getContinueSearchDraft(runId);
  const next = {
    runtime: patch.runtime != null ? Number(patch.runtime) : cur.runtime,
    generated: patch.generated != null ? Number(patch.generated) : cur.generated,
    full_wfo: patch.full_wfo != null ? Number(patch.full_wfo) : cur.full_wfo,
  };
  if (!Number.isFinite(next.runtime) || next.runtime < 0) next.runtime = cur.runtime;
  if (!Number.isFinite(next.generated) || next.generated < 0) next.generated = cur.generated;
  if (!Number.isFinite(next.full_wfo) || next.full_wfo < 0) next.full_wfo = cur.full_wfo;
  state.continueSearchDrafts = { ...(state.continueSearchDrafts || {}), [runId]: next };
  _persistContinueSearchDrafts();
  return next;
}
function wireContinueSearchDraftInputs(runId) {
  const runtime = $("#cs_runtime");
  const gen = $("#cs_gen");
  const wfo = $("#cs_wfo");
  if (!runtime && !gen && !wfo) return;
  const saveFromInputs = () => {
    saveContinueSearchDraft(runId, {
      runtime: runtime ? runtime.value : undefined,
      generated: gen ? gen.value : undefined,
      full_wfo: wfo ? wfo.value : undefined,
    });
  };
  for (const el of [runtime, gen, wfo]) {
    if (!el) continue;
    el.addEventListener("input", saveFromInputs);
    el.addEventListener("change", saveFromInputs);
  }
}

function $(sel, el = document) { return el.querySelector(sel); }
function statusText(v, fallback = "NOT_EVALUATED") {
  if (v == null || v === "" || v === "-") return fallback;
  return String(v);
}
function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return statusText(null, "NOT_AVAILABLE");
  const n = Number(v);
  const s = `${n.toFixed(2)}%`;
  return n < 0 ? `<span class="neg">${s}</span>` : `<span class="pos">${s}</span>`;
}
function fmtVal(v) {
  if (v == null || v === "" || v === "-") return "NOT_EVALUATED";
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(4) : "NOT_EVALUATED";
  return String(v);
}
function fmtMetric(v, digits = 4) {
  if (v == null || v === "" || v === "-") return "NOT_AVAILABLE";
  const n = Number(v);
  if (!Number.isFinite(n)) return statusText(v, "NOT_AVAILABLE");
  if (n === 0) return Number(0).toFixed(digits);
  return Math.abs(n) < 10 ** (-digits) ? n.toExponential(3) : n.toFixed(digits);
}
function fmtPercentMetric(v, digits = 3) {
  if (v == null || v === "" || v === "-") return "NOT_AVAILABLE";
  const n = Number(v);
  return Number.isFinite(n) ? `${fmtMetric(n, digits)}%` : statusText(v, "NOT_AVAILABLE");
}
async function api(path, opts) {
  const r = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(opts && opts.headers) },
    ...opts,
  });
  const text = await r.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || r.statusText));
  return data;
}
function setRun(id) {
  state.selectedRunId = id || "";
  localStorage.setItem("cp_run_id", state.selectedRunId);
}
function setDataset(id) {
  state.selectedDatasetId = id || "";
  localStorage.setItem("cp_dataset_id", state.selectedDatasetId);
}
function setFilter(f) {
  state.runFilter = f;
  localStorage.setItem("cp_run_filter", f);
}
function datasetLabel(d) {
  const range = (d.start_timestamp && d.end_timestamp)
    ? `${String(d.start_timestamp).slice(0, 10)} → ${String(d.end_timestamp).slice(0, 10)}`
    : "n/a";
  const sym = (d.tradable_contracts && d.tradable_contracts[0]) || (d.symbols && d.symbols[0]) || "";
  const smoke = d.smoke_test_only ? " · SMOKE TEST ONLY" : "";
  return `${d.display_name} | ${d.provider_id} | ${d.asset_class} | ${sym} | ${d.timeframe} | ${range} | rows=${d.row_count} | ${d.quality_status} | v=${d.dataset_version}${smoke}`;
}
function nav() {
  const el = $("#nav");
  el.innerHTML = `<div class="brand">Quant Research</div>` + PAGES.map(([id, label]) =>
    `<a href="#/${id}" class="${state.page === id ? "active" : ""}">${label}</a>`
  ).join("");
}
function stopSSE() {
  if (state.sse) { state.sse.close(); state.sse = null; }
}

async function renderOverview() {
  const o = await api("/api/overview");
  return `
    <h1>Overview</h1>
    <p class="sub">System ${o.system_version} · research control plane</p>
    <div class="warn-box">${o.banner}</div>
    <div class="grid">
      <div class="stat"><div class="label">Active</div><div class="value">${o.active}</div></div>
      <div class="stat"><div class="label">Queued</div><div class="value">${o.queued}</div></div>
      <div class="stat"><div class="label">Completed</div><div class="value">${o.completed}</div></div>
      <div class="stat"><div class="label">Failed</div><div class="value">${o.failed}</div></div>
      <div class="stat"><div class="label">Datasets</div><div class="value">${o.dataset_count ?? 0}</div></div>
      <div class="stat"><div class="label">Latest discovery</div><div class="value mono" style="font-size:0.85rem">${o.latest_discovery_run || "NONE"}</div></div>
      <div class="stat"><div class="label">Latest portfolio</div><div class="value mono" style="font-size:0.85rem">${o.latest_portfolio || "NONE"}</div></div>
      <div class="stat"><div class="label">Latest Vault</div><div class="value mono" style="font-size:0.85rem">${o.latest_vault || "NONE"}</div></div>
      <div class="stat"><div class="label">Latest Paper</div><div class="value mono" style="font-size:0.85rem">${o.latest_paper || "NONE"}</div></div>
      <div class="stat"><div class="label">Workers</div><div class="value">${o.health.max_workers}</div></div>
      <div class="stat"><div class="label">Live trading</div><div class="value neg">OFF</div></div>
    </div>`;
}

function newRunForm(datasets) {
  const opts = (datasets || []).map(d =>
    `<option value="${d.dataset_id}">${datasetLabel(d)}</option>`
  ).join("") || `<option value="synthetic_demo">synthetic_demo — SMOKE TEST ONLY</option>`;
  return `
    <h1>New Run</h1>
    <p class="sub">Dataset Catalog drives symbols, timeframe, and compatible models. synthetic_demo is SMOKE TEST ONLY.</p>
    <div class="panel">
      <div class="row">
        <div>
          <label>Run type</label>
          <select id="run_type">
            <option>RESEARCH_DEMO</option>
            <option>ALPHA_MINER</option>
            <option>WFO_ONLY</option>
            <option>PORTFOLIO_BUILD</option>
            <option>VAULT_EVALUATION</option>
            <option>SHADOW_RUNTIME</option>
            <option>PAPER_RUNTIME</option>
          </select>
        </div>
        <div>
          <label>Dataset (from Catalog)</label>
          <select id="dataset">${opts}</select>
        </div>
      </div>
      <div id="dataset_hint" class="warn-box" style="margin-top:0.5rem"></div>
      <div class="row">
        <div><label>Symbol / contract</label><select id="symbols"></select></div>
        <div><label>Timeframe</label><select id="timeframe"></select></div>
      </div>
      <div class="row">
        <div>
          <label>Run mode</label>
          <select id="run_mode">
            <option value="single">Single Family</option>
            <option value="multi">Multi-Family Generated</option>
          </select>
        </div>
        <div><label>Random seed</label><input id="seed" type="number" value="42" /></div>
      </div>
      <div id="single_family_row" class="row">
        <div><label>Strategy family</label><select id="family"></select></div>
        <div><label>Asset class (locked)</label><input id="asset_class" readonly /></div>
      </div>
      <div id="multi_family_panel" class="panel" style="display:none;margin-top:0.5rem">
        <h3>Strategy Family Generator</h3>
        <p class="sub">Hypothesis-driven families with distinct DSL grammars. strategy_family is forced to <span class="mono">multi_family_generated</span>.</p>
        <div class="row">
          <div><label>Family count</label><input id="mf_family_count" type="number" min="2" max="8" value="8" /></div>
          <div><label>Initial candidates per family</label><input id="mf_per_family" type="number" min="1" max="100" value="6" /></div>
          <div><label>Total candidate budget</label><input id="mf_total_budget" type="number" min="1" max="10000" value="96" /></div>
        </div>
        <label>Family blueprints (multiselect — hold Ctrl/Cmd)</label>
        <select id="mf_blueprints" multiple size="8" style="width:100%;min-height:9rem"></select>
        <div class="confirm" style="margin-top:0.5rem">
          <input type="checkbox" id="mf_adaptive" checked />
          <span>Adaptive WFO allocation after early folds</span>
        </div>
        <div class="confirm" style="margin-top:0.5rem">
          <input type="checkbox" id="mf_family_local_evolution" checked />
          <span>Enable family-local evolution</span>
        </div>
        <div class="row">
          <div><label>Evolution generations</label><input id="mf_evolution_generations" type="number" min="1" max="50" value="3" /></div>
          <div><label>Stagnation generations</label><input id="mf_stagnation_generations" type="number" min="1" max="99" value="1" /></div>
          <div><label>Minimum improvement</label><input id="mf_minimum_improvement" type="number" min="0" step="0.0001" value="0.0001" /></div>
        </div>
        <label>Stress scenarios (multiselect — hold Ctrl/Cmd)</label>
        <select id="mf_stress_scenarios" multiple size="6" style="width:100%;min-height:7rem">
          <option value="base_costs" selected>base_costs</option>
          <option value="costs_2x" selected>costs_2x</option>
          <option value="wider_spread" selected>wider_spread</option>
          <option value="worse_slippage" selected>worse_slippage</option>
          <option value="delayed_execution" selected>delayed_execution</option>
          <option value="removed_best_day" selected>removed_best_day</option>
        </select>
        <div class="row">
          <div><label>Max Stress evaluations</label><input id="mf_max_stress_evaluations" type="number" min="0" max="500" value="24" /></div>
          <div><label>Max Stress scenarios / candidate</label><input id="mf_max_stress_scenarios_per_candidate" type="number" min="1" max="50" value="6" /></div>
          <div><label>Minimum Stress pass rate</label><input id="mf_min_stress_pass_rate" type="number" min="0" max="1" step="0.05" value="0.5" /></div>
        </div>
        <div class="row">
          <div><label>Max Robustness candidates</label><input id="mf_max_robustness_candidates" type="number" min="0" max="50" value="3" /></div>
          <div><label>Max Robustness evaluations</label><input id="mf_max_robustness_evaluations" type="number" min="0" max="500" value="18" /></div>
        </div>
        <div class="row">
          <div><label>Max parameters / candidate</label><input id="mf_max_parameters_per_candidate" type="number" min="1" max="20" value="2" /></div>
          <div><label>Max points / parameter</label><input id="mf_max_points_per_parameter" type="number" min="1" max="20" value="3" /></div>
          <div><label>Min valid neighborhood points</label><input id="mf_min_valid_neighborhood_points" type="number" min="1" max="50" value="3" /></div>
        </div>
        <div class="row">
          <div><label>Minimum DSR</label><input id="mf_min_dsr" type="number" min="0" max="5" step="0.01" value="0.95" /></div>
          <div><label>Maximum PBO</label><input id="mf_max_pbo" type="number" min="0" max="1" step="0.01" value="0.5" /></div>
          <div><label>PBO splits</label><input id="mf_pbo_n_splits" type="number" min="2" max="20" value="4" /></div>
        </div>
        <div class="row">
          <div><label>Min DSR OOS observations</label><input id="mf_min_oos_observations_for_dsr" type="number" min="1" max="1000" value="20" /></div>
          <div><label>Behavioral similarity threshold</label><input id="mf_behavioral_similarity_threshold" type="number" min="0" max="1" step="0.01" value="0.85" /></div>
        </div>
        <p class="sub" id="mf_hint">Select at least 2 blueprints. Launch fails if fewer than 2 distinct grammar fingerprints.</p>
      </div>
      <div class="row">
        <div><label>Cost model</label><select id="cost"></select></div>
        <div><label>Risk profile</label><select id="risk"></select></div>
      </div>
      <div class="row">
        <div><label>Feature set</label><select id="features"></select></div>
        <div id="asset_class_multi_wrap"><label>Asset class (locked)</label><input id="asset_class_mirror" readonly /></div>
      </div>
      <label>Search budget JSON</label>
      <textarea id="budget" rows="6">{"max_generated_candidates":200,"max_evaluated_candidates":200,"max_full_wfo_evaluations":200,"population_size":4,"max_runtime_seconds":45,"stagnation_generations":8,"minimum_generations_before_stagnation":4,"max_candidates_per_family":200,"max_candidates_per_complexity_tier":200,"max_candidates_per_feature_family":200,"max_oos_drawdown":0.2}</textarea>
      <label>WFO JSON</label>
      <textarea id="wfo" rows="2">{"train_window_days":5,"validation_window_days":2,"step_forward_days":5,"max_folds":4}</textarea>
      <div class="confirm"><input type="checkbox" id="c_miner" /><span>Confirm Alpha Miner</span></div>
      <div class="confirm"><input type="checkbox" id="c_smoke" /><span>Smoke test only (required for synthetic_demo with Alpha Miner)</span></div>
      <div class="confirm"><input type="checkbox" id="c_vault" /><span>Confirm Vault evaluation</span></div>
      <div class="confirm"><input type="checkbox" id="c_paper" /><span>Confirm Paper runtime</span></div>
      <div id="backend_box" class="panel" style="margin-top:0.75rem"></div>
      <div class="panel" id="preview" style="margin-top:0.75rem"></div>
      <button id="start_btn">Start run</button>
      <pre class="logs" id="create_msg"></pre>
    </div>`;
}

function fillSelect(el, values, selected) {
  const vals = values && values.length ? values : [selected || ""];
  el.innerHTML = vals.map(v => `<option value="${v}" ${v === selected ? "selected" : ""}>${v}</option>`).join("");
}

function resolveBackendPreview(c) {
  const smoke = $("#c_smoke")?.checked === true;
  const family = $("#family")?.value || (c.compatible_strategy_families || [])[0];
  const tf = $("#timeframe")?.value || c.default_timeframe || c.timeframe;
  if (c.smoke_test_only || smoke) {
    return {
      evaluation_backend: "synthetic_oos_probe",
      ok: true,
      banners: ["SYNTHETIC OOS PROBE", "SMOKE TEST ONLY", "VAULT / PAPER / LIVE DISABLED"],
      block_reason: null,
    };
  }
  if (c.research_eligible) {
    if (tf === "tick" && family === "mean_reversion_vwap_bb") {
      return {
        evaluation_backend: null,
        ok: false,
        banners: [],
        block_reason: "REAL_DATA_BACKEND_UNAVAILABLE: mean_reversion_vwap_bb rejects raw tick — select 1m or 5m",
      };
    }
    return {
      evaluation_backend: "event_driven_wfo",
      ok: true,
      banners: c.execution_banners && c.execution_banners.length ? c.execution_banners : [
        "REAL EVENT-DRIVEN WFO",
        "GENUINE DUKASCOPY DATA",
        "OBSERVED BID/ASK SPREAD",
        "INTRADAY ONLY",
        "VAULT / PAPER / LIVE DISABLED",
      ],
      block_reason: null,
    };
  }
  return {
    evaluation_backend: null,
    ok: false,
    banners: [],
    block_reason: "Dataset is not RESEARCH_ELIGIBLE and smoke_test is false",
  };
}

function renderBackendBox(c) {
  const box = $("#backend_box");
  if (!box) return;
  const prev = resolveBackendPreview(c);
  const banners = (prev.banners || []).map(b => `<span class="pill" style="margin-right:0.35rem;display:inline-block;padding:0.15rem 0.45rem;border:1px solid var(--accent);font-size:0.75rem">${b}</span>`).join("");
  if (!prev.ok) {
    box.innerHTML = `<strong style="color:#c0392b">BLOCKED — genuine dataset would not use a real backend</strong>
      <p class="sub">${prev.block_reason || "REAL_DATA_BACKEND_UNAVAILABLE"}</p>
      <p class="sub">Resolved backend: ${prev.evaluation_backend || "NONE"}</p>`;
    const btn = $("#start_btn");
    if (btn && $("#run_type")?.value === "ALPHA_MINER") btn.disabled = true;
    return;
  }
  const real = prev.evaluation_backend === "event_driven_wfo";
  box.innerHTML = `<strong>Resolved execution backend before launch:</strong>
    <div class="mono" style="margin:0.35rem 0">${prev.evaluation_backend}</div>
    <div>${banners}</div>
    ${real ? "" : `<p class="sub">Smoke / synthetic path — FINALIST promotion disabled.</p>`}`;
  const btn = $("#start_btn");
  if (btn) btn.disabled = false;
}

async function applyDatasetConstraints() {
  const id = $("#dataset").value;
  const c = await api(`/api/datasets/${encodeURIComponent(id)}/constraints`);
  state.constraints = c;
  const hint = $("#dataset_hint");
  if (c.smoke_test_only) {
    hint.innerHTML = `<strong>synthetic_demo — SMOKE TEST ONLY</strong>. Institutional Alpha Miner requires research_eligible data, or check Smoke test only.`;
  } else if (!c.research_eligible) {
    hint.innerHTML = `<strong>NOT RESEARCH ELIGIBLE</strong> — Alpha Miner will reject this dataset.`;
  } else {
    hint.innerHTML = `Research eligible · ${c.asset_class} · constraints locked to catalog membership.`;
  }
  const symOpts = [...new Set([...(c.tradable_contracts || []), ...(c.symbols || [])])];
  fillSelect($("#symbols"), symOpts, symOpts[0]);
  const tfs = (c.compatible_timeframes && c.compatible_timeframes.length)
    ? c.compatible_timeframes
    : [c.timeframe];
  const defaultTf = c.default_timeframe || (tfs.includes("1m") ? "1m" : tfs[0]);
  fillSelect($("#timeframe"), tfs, defaultTf);
  fillSelect($("#family"), c.compatible_strategy_families, c.compatible_strategy_families[0]);
  fillSelect($("#cost"), c.compatible_cost_models, c.compatible_cost_models[0]);
  fillSelect($("#risk"), c.compatible_risk_profiles, c.compatible_risk_profiles[0]);
  fillSelect($("#features"), c.compatible_feature_sets, c.compatible_feature_sets[0]);
  const ac = c.asset_class || "";
  if ($("#asset_class")) $("#asset_class").value = ac;
  if ($("#asset_class_mirror")) $("#asset_class_mirror").value = ac;
  if (c.smoke_test_only) $("#c_smoke").checked = true;
  else if (c.research_eligible) $("#c_smoke").checked = false;
  renderBackendBox(c);
}

function selectedBlueprintIds() {
  const sel = $("#mf_blueprints");
  if (!sel) return [];
  return Array.from(sel.selectedOptions || []).map(o => o.value).filter(Boolean);
}

function selectedStressScenarios() {
  const sel = $("#mf_stress_scenarios");
  if (!sel) {
    return ["base_costs", "costs_2x", "wider_spread", "worse_slippage", "delayed_execution", "removed_best_day"];
  }
  const picked = Array.from(sel.selectedOptions || []).map(o => o.value).filter(Boolean);
  return picked.length
    ? picked
    : ["base_costs", "costs_2x", "wider_spread", "worse_slippage", "delayed_execution", "removed_best_day"];
}

function mfNum(id, fallback) {
  const el = $(`#${id}`);
  const n = Number(el && el.value);
  return Number.isFinite(n) ? n : fallback;
}

function isMultiFamilyMode() {
  return ($("#run_mode") && $("#run_mode").value === "multi");
}

function syncRunModePanels() {
  const multi = isMultiFamilyMode();
  const singleRow = $("#single_family_row");
  const multiPanel = $("#multi_family_panel");
  if (singleRow) singleRow.style.display = multi ? "none" : "";
  if (multiPanel) multiPanel.style.display = multi ? "" : "none";
  const hint = $("#mf_hint");
  if (hint && multi) {
    const n = selectedBlueprintIds().length;
    hint.textContent = n < 2
      ? "Select at least 2 blueprints. Launch fails if fewer than 2 distinct grammar fingerprints."
      : `${n} blueprints selected · strategy_family → multi_family_generated`;
  }
}

async function loadFamilyBlueprints() {
  const seed = Number(($("#seed") && $("#seed").value) || 42);
  const data = await api(`/api/strategy_families/blueprints?seed=${encodeURIComponent(seed)}`);
  const sel = $("#mf_blueprints");
  if (!sel) return data;
  const prev = new Set(selectedBlueprintIds());
  const rows = data.blueprints || [];
  sel.innerHTML = rows.map(b => {
    const selected = prev.size ? prev.has(b.family_id) : true;
    return `<option value="${b.family_id}" ${selected ? "selected" : ""} title="${b.hypothesis}">${b.family_id} · fp ${String(b.effective_grammar_fingerprint || "").slice(0, 10)}</option>`;
  }).join("");
  if (!prev.size) {
    // Default: select first N matching family count (at least 2).
    const want = Math.max(2, Number(($("#mf_family_count") && $("#mf_family_count").value) || 8));
    Array.from(sel.options).forEach((o, i) => { o.selected = i < want; });
  }
  syncRunModePanels();
  return data;
}

async function fetchMultiFamilyPreview(body) {
  if (!body.multi_family || !body.multi_family.enabled) return null;
  try {
    return await api("/api/strategy_families/preview", {
      method: "POST",
      body: JSON.stringify({
        seed: body.multi_family.seed ?? body.random_seed,
        requested_family_count: body.multi_family.requested_family_count,
        family_ids: body.multi_family.family_ids,
      }),
    });
  } catch (e) {
    return { error: String(e.message || e) };
  }
}

function resolveEffectiveBudget(budget) {
  const maxGen = Number(budget.max_generated_candidates ?? 12);
  const pick = (key) => (budget[key] == null ? maxGen : Number(budget[key]));
  const effective = {
    ...budget,
    max_candidates_per_family: pick("max_candidates_per_family"),
    max_candidates_per_complexity_tier: pick("max_candidates_per_complexity_tier"),
    max_candidates_per_feature_family: pick("max_candidates_per_feature_family"),
  };
  const requested = {
    max_candidates_per_family: budget.max_candidates_per_family ?? null,
    max_candidates_per_complexity_tier: budget.max_candidates_per_complexity_tier ?? null,
    max_candidates_per_feature_family: budget.max_candidates_per_feature_family ?? null,
  };
  return { requested, effective, max_generated_candidates: maxGen };
}

async function wireNewRun() {
  await applyDatasetConstraints();
  await loadFamilyBlueprints();
  syncRunModePanels();
  let previewTimer = null;
  const preview = async () => {
    syncRunModePanels();
    const body = collectRunBody();
    if (state.constraints) renderBackendBox(state.constraints);
    const prev = state.constraints ? resolveBackendPreview(state.constraints) : {};
    let mfBlock = "";
    let mfSpecs = null;
    const mfBudgetErr = multiFamilyBudgetError(body);
    if (mfBudgetErr) {
      mfBlock = `<div class="warn-box" style="border-color:#c0392b;color:#c0392b"><strong>Multi-Family:</strong> ${mfBudgetErr}</div>`;
    } else if (body.multi_family && body.multi_family.enabled) {
      mfSpecs = await fetchMultiFamilyPreview(body);
      if (mfSpecs && mfSpecs.error) {
        mfBlock = `<div class="warn-box" style="border-color:#c0392b;color:#c0392b"><strong>Multi-Family:</strong> ${mfSpecs.error}</div>`;
      } else if (mfSpecs) {
        body.multi_family = {
          ...body.multi_family,
          preview_family_specs: mfSpecs.families,
          distinct_grammar_fingerprints: mfSpecs.distinct_grammar_fingerprints,
          grammar_fingerprints: mfSpecs.grammar_fingerprints,
        };
        mfBlock = `<div class="panel" style="margin-top:0.5rem"><strong>Generated family specs</strong>
          <p class="sub">${mfSpecs.family_count} families · ${mfSpecs.distinct_grammar_fingerprints} distinct grammar fingerprints</p>
          <pre class="mono">${JSON.stringify(mfSpecs.families, null, 2)}</pre></div>`;
      }
    }
    const block = prev && prev.ok === false
      ? `<div class="warn-box" style="border-color:#c0392b;color:#c0392b"><strong>Cannot launch:</strong> ${prev.block_reason}</div>`
      : "";
    const caps = resolveEffectiveBudget(body.search_budget || {});
    $("#preview").innerHTML = `${block}${mfBlock}<strong>Frozen config preview</strong><pre class="mono">${JSON.stringify(body, null, 2)}</pre>
      <div class="panel" style="margin-top:0.5rem"><strong>Bucket caps (requested → effective)</strong>
        <pre class="mono">${JSON.stringify({
          max_generated_candidates: caps.max_generated_candidates,
          max_candidates_per_family: {
            requested: caps.requested.max_candidates_per_family,
            effective: caps.effective.max_candidates_per_family,
          },
          max_candidates_per_complexity_tier: {
            requested: caps.requested.max_candidates_per_complexity_tier,
            effective: caps.effective.max_candidates_per_complexity_tier,
          },
          max_candidates_per_feature_family: {
            requested: caps.requested.max_candidates_per_feature_family,
            effective: caps.effective.max_candidates_per_feature_family,
          },
        }, null, 2)}</pre>
        <p class="sub">Omitted family/tier/feature-family caps default to max_generated_candidates (${caps.max_generated_candidates}), not a hidden 20.</p>
      </div>
      <p class="sub">Resolved backend: ${prev.evaluation_backend || "?"} · research-only · no live orders</p>`;
  };
  const schedulePreview = () => {
    if (previewTimer) clearTimeout(previewTimer);
    previewTimer = setTimeout(() => { preview().catch(() => {}); }, 120);
  };
  ["run_type","dataset","symbols","timeframe","family","seed","cost","risk","features","budget","wfo","c_miner","c_smoke","c_vault","c_paper","run_mode","mf_family_count","mf_per_family","mf_total_budget","mf_blueprints","mf_adaptive","mf_family_local_evolution","mf_evolution_generations","mf_stagnation_generations","mf_minimum_improvement","mf_stress_scenarios","mf_max_stress_evaluations","mf_max_stress_scenarios_per_candidate","mf_min_stress_pass_rate","mf_max_robustness_candidates","mf_max_robustness_evaluations","mf_max_parameters_per_candidate","mf_max_points_per_parameter","mf_min_valid_neighborhood_points","mf_min_dsr","mf_max_pbo","mf_pbo_n_splits","mf_min_oos_observations_for_dsr","mf_behavioral_similarity_threshold"]
    .forEach(id => {
      const el = $(`#${id}`);
      if (!el) return;
      el.addEventListener("input", schedulePreview);
      el.addEventListener("change", schedulePreview);
    });
  $("#run_mode").addEventListener("change", () => {
    syncRunModePanels();
    schedulePreview();
  });
  $("#seed").addEventListener("change", async () => {
    if (isMultiFamilyMode()) await loadFamilyBlueprints();
    schedulePreview();
  });
  $("#dataset").addEventListener("change", async () => {
    await applyDatasetConstraints();
    schedulePreview();
  });
  await preview();
  $("#start_btn").onclick = async () => {
    try {
      if (state.constraints) {
        const prev = resolveBackendPreview(state.constraints);
        if ($("#run_type").value === "ALPHA_MINER" && prev.ok === false) {
          $("#create_msg").textContent = prev.block_reason || "REAL_DATA_BACKEND_UNAVAILABLE";
          return;
        }
      }
      const body = collectRunBody();
      const mfBudgetErr = multiFamilyBudgetError(body);
      if (mfBudgetErr) {
        $("#create_msg").textContent = mfBudgetErr;
        return;
      }
      if (body.multi_family && body.multi_family.enabled) {
        const mfPrev = await fetchMultiFamilyPreview(body);
        if (mfPrev && mfPrev.error) {
          $("#create_msg").textContent = mfPrev.error;
          return;
        }
        if (!mfPrev || Number(mfPrev.distinct_grammar_fingerprints || 0) < 2) {
          $("#create_msg").textContent =
            "MULTI_FAMILY_TOO_FEW: fewer than 2 distinct grammar fingerprints";
          return;
        }
      }
      const run = await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
      setRun(run.run_id);
      $("#create_msg").textContent = `Started ${run.run_id} · state=${run.state}`;
      location.hash = run.run_type === "ALPHA_MINER" ? "#/alpha-miner" : "#/run-detail";
    } catch (e) {
      $("#create_msg").textContent = String(e.message || e);
    }
  };
}
function multiFamilyBudgetError(body) {
  if (!body || !body.multi_family || !body.multi_family.enabled) return null;
  const mf = body.multi_family;
  const familyCount = Number(mf.requested_family_count || 0);
  const perFamily = Number(mf.min_candidates_per_family || 0);
  const totalBudget = Number(mf.total_candidate_budget || 0);
  const initialPop = familyCount * perFamily;
  if (!(totalBudget >= initialPop)) {
    return "MULTI_FAMILY_INVALID: total candidate budget is smaller than the initial population";
  }
  const runtime = Number(mf.max_runtime_seconds);
  if (!Number.isFinite(runtime) || !(runtime > 0)) {
    return "MULTI_FAMILY_INVALID: max_runtime_seconds must be a positive finite number";
  }
  return null;
}
function collectRunBody() {
  let budget = {}, wfo = {};
  try { budget = JSON.parse($("#budget").value || "{}"); } catch {}
  try { wfo = JSON.parse($("#wfo").value || "{}"); } catch {}
  const multi = isMultiFamilyMode();
  const seed = Number($("#seed").value);
  const perFamily = Math.max(1, Number(($("#mf_per_family") && $("#mf_per_family").value) || 6));
  const familyCount = Math.max(2, Number(($("#mf_family_count") && $("#mf_family_count").value) || 8));
  // Independent of initial population (familyCount * perFamily); do not auto-derive.
  const totalBudget = Math.max(1, Number(($("#mf_total_budget") && $("#mf_total_budget").value) || 96));
  const familyIds = selectedBlueprintIds();
  const body = {
    run_type: $("#run_type").value,
    dataset: $("#dataset").value,
    symbols: [$("#symbols").value].filter(Boolean),
    timeframe: $("#timeframe").value,
    strategy_family: multi ? "multi_family_generated" : $("#family").value,
    random_seed: seed,
    cost_model_version: $("#cost").value,
    risk_profile: $("#risk").value,
    feature_set_version: $("#features").value,
    search_budget: budget,
    wfo,
    smoke_test: $("#c_smoke").checked,
    confirm_alpha_miner: $("#c_miner").checked,
    confirm_vault: $("#c_vault").checked,
    confirm_paper: $("#c_paper").checked,
  };
  if (multi) {
    const rawMaxGen = Number(budget.max_generated_candidates);
    // Explicitly synchronize generated cap upward when total budget exceeds it.
    const maxGenerated = Number.isFinite(rawMaxGen)
      ? Math.max(rawMaxGen, totalBudget)
      : totalBudget;
    body.search_budget = {
      ...budget,
      max_generated_candidates: maxGenerated,
      max_evaluated_candidates: totalBudget,
      max_full_wfo_evaluations: Math.max(1, Math.floor(totalBudget / 3)),
    };
    body.multi_family = {
      enabled: true,
      requested_family_count: familyCount,
      min_candidates_per_family: perFamily,
      total_candidate_budget: totalBudget,
      max_full_wfo: Number(body.search_budget.max_full_wfo_evaluations),
      max_runtime_seconds: Number(budget.max_runtime_seconds ?? 7200),
      adaptive_reallocation: !!( $("#mf_adaptive") && $("#mf_adaptive").checked ),
      family_ids: familyIds,
      seed,
      max_oos_drawdown: Number(budget.max_oos_drawdown ?? 0.2),
      family_local_evolution: !!( $("#mf_family_local_evolution") && $("#mf_family_local_evolution").checked ),
      evolution_generations: mfNum("mf_evolution_generations", 3),
      stagnation_generations: mfNum("mf_stagnation_generations", 1),
      minimum_improvement: mfNum("mf_minimum_improvement", 0.0001),
      allow_cross_family_crossover: false,
      stress_scenarios: selectedStressScenarios(),
      max_stress_evaluations: mfNum("mf_max_stress_evaluations", 24),
      max_stress_scenarios_per_candidate: mfNum("mf_max_stress_scenarios_per_candidate", 6),
      min_stress_pass_rate: mfNum("mf_min_stress_pass_rate", 0.5),
      fail_closed_unsupported_stress: true,
      max_robustness_candidates: mfNum("mf_max_robustness_candidates", 3),
      max_robustness_evaluations: mfNum("mf_max_robustness_evaluations", 18),
      max_parameters_per_candidate: mfNum("mf_max_parameters_per_candidate", 2),
      max_points_per_parameter: mfNum("mf_max_points_per_parameter", 3),
      min_valid_neighborhood_points: mfNum("mf_min_valid_neighborhood_points", 3),
      allow_one_sided_neighborhood: false,
      min_dsr: mfNum("mf_min_dsr", 0.95),
      max_pbo: mfNum("mf_max_pbo", 0.5),
      pbo_n_splits: mfNum("mf_pbo_n_splits", 4),
      min_oos_observations_for_dsr: mfNum("mf_min_oos_observations_for_dsr", 20),
      behavioral_similarity_threshold: mfNum("mf_behavioral_similarity_threshold", 0.85),
    };
  }
  return body;
}

const RUN_TABS = ["DEFAULT", "ALL", "ACTIVE", "QUEUED", "COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"];

async function renderRunsPage(mode) {
  const filter = mode === "active-only" ? "ACTIVE" : state.runFilter;
  const data = await api(`/api/runs?filter=${encodeURIComponent(filter)}`);
  const runs = data.runs || [];
  const tabs = RUN_TABS.map(t =>
    `<button class="secondary ${filter === t ? "active-tab" : ""}" data-filter="${t}" style="margin-right:0.35rem;${filter===t?"outline:1px solid var(--accent)":""}">${t}</button>`
  ).join("");
  const rows = runs.map(r => `
    <tr data-id="${r.run_id}">
      <td class="mono">${r.run_id}</td>
      <td>${r.run_type}</td>
      <td>${r.state}</td>
      <td>${r.run_type === "ALPHA_MINER" ? statusText(r.summary?.discovery_result || r.profitability_result, "SEE_SUMMARY") : fmtPct(r.total_return_pct)}</td>
      <td>${(r.progress_pct||0).toFixed(0)}%</td>
      <td>${r.current_stage || statusText(null, "IDLE")}</td>
      <td>${r.finalist_count ?? 0}</td>
    </tr>`).join("") || `<tr><td colspan="7">No runs in this filter</td></tr>`;
  return `
    <h1>Runs</h1>
    <p class="sub">Default view: active runs first, then recent terminal history. History persists across restarts.</p>
    <div class="panel" id="run_tabs">${tabs}</div>
    <div class="panel"><table>
      <thead><tr><th>Run ID</th><th>Type</th><th>State</th><th>Result</th><th>Progress</th><th>Stage</th><th>Finalists</th></tr></thead>
      <tbody id="run_rows">${rows}</tbody>
    </table></div>
    <div class="panel" id="detail"></div>`;
}

async function wireRuns() {
  document.querySelectorAll("#run_tabs [data-filter]").forEach(btn => {
    btn.onclick = () => { setFilter(btn.dataset.filter); route(); };
  });
  $("#run_rows").onclick = async (ev) => {
    const tr = ev.target.closest("tr[data-id]");
    if (!tr) return;
    setRun(tr.dataset.id);
    await showDetail(tr.dataset.id);
  };
  if (state.selectedRunId) await showDetail(state.selectedRunId);
}

async function showDetail(id) {
  const run = await api(`/api/runs/${id}`);
  state.alphaRunActive = ACTIVE.has(run.state);
  const summary = await api(`/api/runs/${id}/summary`);
  const el = $("#detail");
  if (!el) return;
  const cancellable = !!run.cancellable && ACTIVE.has(run.state);
  el.innerHTML = `
    <h2 class="mono">${run.run_id}</h2>
    <div class="grid">
      <div class="stat"><div class="label">State</div><div class="value" style="font-size:1rem">${run.state}</div></div>
      <div class="stat"><div class="label">Software</div><div class="value" style="font-size:0.95rem">${statusText(summary.software_execution_status)}</div></div>
      <div class="stat"><div class="label">Discovery</div><div class="value" style="font-size:0.85rem">${statusText(summary.discovery_result, "NOT_APPLICABLE")}</div></div>
      <div class="stat"><div class="label">Profitability</div><div class="value" style="font-size:0.85rem">${statusText(summary.profitability_result, "NOT_AVAILABLE")}</div></div>
      <div class="stat"><div class="label">Total return</div><div class="value">${run.total_return_pct == null ? statusText(summary.total_return_status, "NOT_AVAILABLE") : fmtPct(summary.total_return_pct)}</div></div>
      <div class="stat"><div class="label">Statistical</div><div class="value" style="font-size:0.85rem">${statusText(summary.statistical_result)}</div></div>
      <div class="stat"><div class="label">Vault</div><div class="value" style="font-size:0.85rem">${statusText(summary.vault_result, "NOT_SUBMITTED")}</div></div>
      <div class="stat"><div class="label">Terminal reason</div><div class="value" style="font-size:0.8rem">${statusText(run.terminal_reason, "NONE")}</div></div>
    </div>
    ${cancellable ? `<button class="danger" id="cancel_btn">Cancel</button>` : `<button class="secondary" id="cancel_btn" disabled title="Terminal runs cannot be cancelled">Cancel unavailable</button>`}
    <button class="secondary" id="refresh_btn">Refresh</button>
    ${run.run_type === "ALPHA_MINER" ? `<button class="secondary" id="goto_miner">Open Alpha Miner</button>` : ""}
    <pre class="logs">${JSON.stringify({run, summary}, null, 2)}</pre>`;
  const cancelBtn = $("#cancel_btn");
  if (cancellable) {
    cancelBtn.onclick = async () => {
      try {
        await api(`/api/runs/${id}/cancel`, { method: "POST" });
      } catch (e) {
        alert(String(e.message || e));
      }
      await showDetail(id);
    };
  }
  $("#refresh_btn").onclick = () => showDetail(id);
  const gm = $("#goto_miner");
  if (gm) gm.onclick = () => { location.hash = "#/alpha-miner"; };
}

async function renderAlpha() {
  const all = await api("/api/runs?filter=ALL");
  const miners = (all.runs || []).filter(r => r.run_type === "ALPHA_MINER");
  const id = state.selectedRunId && miners.some(m => m.run_id === state.selectedRunId)
    ? state.selectedRunId
    : (miners[0] && miners[0].run_id) || "";
  if (id && id !== state.selectedRunId) setRun(id);

  const picker = `<label>Alpha Miner run</label>
    <select id="miner_run">${miners.map(m =>
      `<option value="${m.run_id}" ${m.run_id===id?"selected":""}>${m.run_id} · ${m.state} · finalists=${m.finalist_count||0}</option>`
    ).join("") || `<option value="">No ALPHA_MINER runs</option>`}</select>`;

  if (!id) {
    return `<h1>Alpha Miner</h1><div class="panel">${picker}</div>
      <div class="warn-box">No Alpha Miner runs yet. Start one from New Run.</div>`;
  }

  const run = await api(`/api/runs/${id}`);
  let summary = {};
  try {
    summary = await api(`/api/runs/${id}/summary`);
  } catch {
    summary = run.summary || {};
  }
  let report = {};
  try { report = await api(`/api/runs/${id}/alpha_miner`); } catch { report = {}; }
  let cands = {candidates: report.candidates || [], tables: report.tables || {}, registry_total: (report.candidates || []).length, rejected_visible: true};
  if (!Array.isArray(report.candidates) || report.candidates.length === 0) {
    try { cands = await api(`/api/runs/${id}/candidates`); } catch {}
  }
  let rows = cands.candidates || [];
  if (state.filter) {
    const f = state.filter.toLowerCase();
    rows = rows.filter(c => JSON.stringify(c).toLowerCase().includes(f));
  }
  rows = rows.slice().sort((a,b) => {
    const av = a[state.sortKey], bv = b[state.sortKey];
    if (av == bv) return 0;
    return (av > bv ? 1 : -1) * state.sortDir;
  });
  const tables = report.tables || cands.tables || {};
  const finalists = tables.finalists || rows.filter(c => c.promotion_label === "FINALIST");
  const shortlist = tables.unvalidated_shortlist || rows.filter(c => ["SHORTLISTED","TOP_RANKED_UNVALIDATED","RESEARCH_SHORTLISTED"].includes(c.promotion_label));
  const rejected = tables.rejected || rows.filter(c => c.rejected && c.evaluation_stage !== "DUPLICATE");
  const duplicates = tables.duplicates || rows.filter(c => c.evaluation_stage === "DUPLICATE");
  const failures = tables.evaluation_failures || rows.filter(c => ["EVALUATION_ERROR","WFO_FAILED","INVALID"].includes(c.evaluation_stage));
  const noFinalists = Number(report.finalist_count ?? (typeof report.finalists === "number" ? report.finalists : 0)) === 0;
  const isFailed = run.state === "FAILED" || report.software_execution_status === "SOFTWARE_FAILURE" || report.discovery_result === "SOFTWARE_FAILURE" || report.software_failure_banner === true;
  const provenance =
    report.runtime_provenance ||
    summary.runtime_provenance ||
    run.summary?.runtime_provenance ||
    {};
  const signalSource =
    report.signal_source ||
    summary.signal_source ||
    run.summary?.signal_source ||
    "unknown";
  const softwareErr = statusText(report.terminal_reason || run.terminal_reason || report.error, "SOFTWARE_FAILURE");
  const fpDiff = report.fingerprint_diff || run.summary?.fingerprint_diff || null;
  const fpMismatchHtml = fpDiff ? `
    <div class="sub" style="margin-top:0.5rem">
      <strong>Fingerprint mismatch:</strong> ${(fpDiff.changed_reason_codes || []).join(", ") || "INCOMPATIBLE_SEARCH_FINGERPRINT"}
      <pre class="mono" style="margin-top:0.35rem;white-space:pre-wrap">${JSON.stringify({
        changed_components: fpDiff.changed_components,
        changed_reason_codes: fpDiff.changed_reason_codes,
        stored_dataset_hash: fpDiff.stored_dataset_hash,
        incoming_dataset_hash: fpDiff.incoming_dataset_hash,
        stored_wfo_hash: fpDiff.stored_wfo_hash,
        incoming_wfo_hash: fpDiff.incoming_wfo_hash,
        stored_git_commit_sha: fpDiff.stored_git_commit_sha,
        incoming_git_commit_sha: fpDiff.incoming_git_commit_sha,
        stored_components: fpDiff.stored_components,
        incoming_components: fpDiff.incoming_components,
      }, null, 2)}</pre>
    </div>` : "";
  const budget = report.search_budget_consumed || {};
  const dist = report.rejection_reason_distribution || {};
  state.alphaRunActive = ACTIVE.has(run.state);
  const csDraft = getContinueSearchDraft(id);

  return `
    <h1>Alpha Miner</h1>
    <p class="sub">Live funnel + terminal summary. Software success ≠ discovery success.</p>
    <div class="panel">${picker}
      <p class="sub" id="live_status">State: ${run.state} · progress ${(run.progress_pct||0).toFixed(0)}% · stage ${run.current_stage || "IDLE"}</p>
    </div>
    ${isFailed ? `<div class="warn-box" style="border-color:#c0392b;background:rgba(192,57,43,0.12)"><strong style="color:#c0392b">SOFTWARE FAILURE</strong><br/>${softwareErr}${fpMismatchHtml}<br/><span class="sub">Backend: ${statusText(report.evaluation_backend || run.summary?.evaluation_backend, "unknown")} · path: ${statusText(report.evaluation_path, report.evaluation_backend)}</span></div>` : (noFinalists && run.state === "COMPLETED" ? `<div class="warn-box"><strong>No candidate passed all mandatory Alpha Miner gates.</strong> Status: ${statusText(report.qualified_candidate_status, report.discovery_result || "NO_QUALIFIED_CANDIDATE")}<br/><span class="sub">${statusText(report.generation_cap_explanation, "")}</span></div>` : "")}
    ${(run.terminal_reason === "RUNTIME_EXHAUSTED" || report.terminal_reason === "RUNTIME_EXHAUSTED") ? `
    <div class="panel" id="continue_search_panel">
      <h3>Continue Search</h3>
      <p class="sub">Resume the persistent search program without regenerating Generation 0 or repeating completed evaluations. Session runtime is the value below (not added to the prior cap).</p>
      <div class="grid" style="grid-template-columns:repeat(3,minmax(0,1fr));gap:0.5rem">
        <label>Session runtime (sec)<input id="cs_runtime" type="number" min="0" value="${csDraft.runtime}"/></label>
        <label>Additional generated budget<input id="cs_gen" type="number" min="0" value="${csDraft.generated}"/></label>
        <label>Additional Full-WFO budget<input id="cs_wfo" type="number" min="0" value="${csDraft.full_wfo}"/></label>
      </div>
      <p class="sub">search_program_id=${statusText((run.config_snapshot||{}).search_program_id || (run.config_snapshot||{}).multi_family?.search_program_id || report.budget_allocation?.search_program_id)} · pending by gate: ${JSON.stringify(report.budget_allocation?.pending_by_gate || {})}</p>
      <button id="cs_continue">Continue Search</button>
    </div>` : ""}
    <p class="sub">Backend: ${statusText(report.evaluation_backend, "unknown")} · path: ${statusText(report.evaluation_path, report.evaluation_backend)} · signal_source=${statusText(signalSource)} · is_full_event_wfo=${report.is_full_event_wfo === true} · terminal: ${statusText(report.terminal_reason)} · proxy_metric_used=${report.proxy_metric_used === true}</p>
    ${(report.budget_allocation && (report.budget_allocation.search_program_id || report.budget_allocation.cumulative_totals)) ? `
    <div class="panel">
      <h3>Search program totals</h3>
      <pre class="mono">${JSON.stringify({
        search_program_id: report.budget_allocation.search_program_id,
        search_mode: report.budget_allocation.search_mode,
        compatibility_fingerprint: report.budget_allocation.compatibility_fingerprint,
        source_run_id: report.budget_allocation.source_run_id,
        resumed_from_run_id: report.budget_allocation.resumed_from_run_id,
        cumulative: report.budget_allocation.cumulative_totals,
        session: report.budget_allocation.session_totals,
        pending_by_gate: report.budget_allocation.pending_by_gate,
        evaluation_cache_hits: report.budget_allocation.evaluation_cache_hits,
        evaluation_cache_misses: report.budget_allocation.evaluation_cache_misses,
      }, null, 2)}</pre>
    </div>` : ""}
    ${provenance && provenance.git_commit_sha ? `<div class="panel"><h3>Runtime provenance</h3><pre class="mono">${JSON.stringify({
      git_commit_sha: provenance.git_commit_sha,
      python_executable: provenance.python_executable,
      cwd: provenance.cwd,
      module_paths: provenance.module_paths,
      signal_source: signalSource,
      pid: provenance.pid,
    }, null, 2)}</pre></div>` : ""}
    ${Array.isArray(report.family_funnel) && report.family_funnel.length ? `
    <div class="panel">
      <h3>Strategy Family Generator</h3>
      <p class="sub">Hypothesis-driven families above Alpha Miner — distinct DSL search spaces.</p>
      <h4>Budget allocation</h4>
      <pre class="mono">${JSON.stringify(report.budget_allocation || {}, null, 2)}</pre>
      <h4>Family funnel</h4>
      <table>
        <thead><tr>
          <th>Family</th><th>Alloc gen</th><th>Alloc WFO</th><th>Generated</th><th>Evaluated</th>
          <th>Full WFO</th><th>Score qual</th><th>Stress</th><th>Robust</th><th>Stats</th><th>Shortlist</th>
          <th>Best fitness</th>
          <th>Med OOS Exp</th><th>Med PF</th>
        </tr></thead>
        <tbody>${report.family_funnel.map(f => `<tr>
          <td class="mono">${f.family_id}</td>
          <td>${fmtVal(f.allocation_generated)}</td>
          <td>${fmtVal(f.allocation_wfo)}</td>
          <td>${fmtVal(f.generated)}</td>
          <td>${fmtVal(f.evaluated)}</td>
          <td>${fmtVal(f.full_wfo)}</td>
          <td>${fmtVal(f.score_qualified)}</td>
          <td>${fmtVal(f.stress_passed)}</td>
          <td>${fmtVal(f.robustness_passed)}</td>
          <td>${fmtVal(f.statistically_passed)}</td>
          <td>${fmtVal(f.research_shortlisted)}</td>
          <td>${fmtMetric(f.best_fitness, 4)}</td>
          <td>${fmtMetric(f.median_oos_expectancy, 6)}</td>
          <td>${fmtMetric(f.median_pf, 4)}</td>
        </tr>`).join("")}</tbody>
      </table>
      <h4>Hypothesis &amp; constraints</h4>
      ${report.family_funnel.map(f => `<div style="margin-bottom:0.75rem">
        <strong class="mono">${f.family_id}</strong> — ${statusText(f.hypothesis, "")}
        <pre class="mono" style="max-height:10rem;overflow:auto">${JSON.stringify({
          grammar_fingerprint: f.grammar_fingerprint,
          best_candidate_ids: f.best_candidate_ids,
          rejection_reasons: f.rejection_reasons,
          constraints: f.constraints,
        }, null, 2)}</pre>
      </div>`).join("")}
      <h4>Best candidates per family</h4>
      <pre class="mono">${JSON.stringify(report.best_candidates_per_family || {}, null, 2)}</pre>
    </div>` : ""}
    ${Array.isArray(report.clusters) && report.clusters.length ? `
    <div class="panel">
      <h3>Behavioral clusters (count=${report.clusters.length})</h3>
      <table>
        <thead><tr><th>Cluster</th><th>Members</th><th>Representative</th></tr></thead>
        <tbody>${report.clusters.map(c => `<tr>
          <td class="mono">${c.cluster_id}</td>
          <td class="mono">${(c.member_ids || []).join(", ")}</td>
          <td class="mono">${c.representative_id || ""}</td>
        </tr>`).join("")}</tbody>
      </table>
      ${report.clustering_accounting ? `<pre class="mono">${JSON.stringify(report.clustering_accounting, null, 2)}</pre>` : ""}
    </div>` : ""}
    ${Array.isArray(report.research_shortlist) && report.research_shortlist.length ? `
    <div class="panel">
      <h3>Research shortlist (count=${report.research_shortlist.length})</h3>
      <p class="sub">Vault / Paper / Live remain blocked. Shortlist ≠ Finalist.</p>
      <table>
        <thead><tr>
          <th>ID</th><th>Family</th><th>Cluster</th><th>Fitness</th><th>DSR</th><th>PBO</th><th>Gates</th>
        </tr></thead>
        <tbody>${report.research_shortlist.map(e => `<tr>
          <td class="mono">${e.candidate_id}</td>
          <td>${e.family_id || ""}</td>
          <td class="mono">${e.cluster_id || ""}</td>
          <td>${fmtVal(e.fitness)}</td>
          <td>${fmtVal(e.dsr_value)}</td>
          <td>${fmtVal(e.pbo_value)}</td>
          <td class="mono">${(e.gates_passed || []).join(" → ")}</td>
        </tr>`).join("")}</tbody>
      </table>
    </div>` : ""}
    ${(report.candidate_statistics_summaries && report.candidate_statistics_summaries.length) || report.statistics_accounting || report.population_stats ? `
    <div class="panel">
      <h3>Statistical artifacts (DSR / PBO)</h3>
      ${report.statistics_accounting ? `<pre class="mono">${JSON.stringify(report.statistics_accounting, null, 2)}</pre>` : ""}
      ${report.population_stats ? `<pre class="mono">${JSON.stringify(report.population_stats, null, 2)}</pre>` : ""}
      ${Array.isArray(report.candidate_statistics_summaries) && report.candidate_statistics_summaries.length ? `
      <table>
        <thead><tr>
          <th>ID</th><th>Family</th><th>Obs Sharpe</th><th>N obs</th>
          <th>DSR</th><th>DSR status</th><th>PBO</th><th>PBO status</th><th>Decision</th><th>Reason</th>
        </tr></thead>
        <tbody>${report.candidate_statistics_summaries.map(s => `<tr>
          <td class="mono">${s.candidate_id}</td>
          <td>${s.family_id || ""}</td>
          <td>${fmtVal(s.observed_sharpe)}</td>
          <td>${fmtVal(s.n_observations)}</td>
          <td>${fmtVal(s.dsr_value)}</td>
          <td>${fmtVal(s.dsr_status)}</td>
          <td>${fmtVal(s.pbo_value)}</td>
          <td>${fmtVal(s.pbo_status)}</td>
          <td>${fmtVal(s.final_decision)}</td>
          <td class="mono">${fmtVal(s.final_reason || s.dsr_reason || s.pbo_reason)}</td>
        </tr>`).join("")}</tbody>
      </table>` : ""}
      ${Array.isArray(report.shortlist_rejects) && report.shortlist_rejects.length ? `
      <h4>Shortlist rejects</h4>
      <table>
        <thead><tr><th>ID</th><th>Cluster</th><th>Reason</th><th>Missing gates</th></tr></thead>
        <tbody>${report.shortlist_rejects.map(r => `<tr>
          <td class="mono">${r.candidate_id}</td>
          <td class="mono">${r.cluster_id || ""}</td>
          <td class="mono">${fmtVal(r.reason)}</td>
          <td class="mono">${(r.missing_gates || []).join(", ")}</td>
        </tr>`).join("")}</tbody>
      </table>` : ""}
    </div>` : ""}
    ${Array.isArray(report.candidate_status_history) && report.candidate_status_history.length ? `
    <div class="panel">
      <h3>Candidate status history (seq=${report.candidate_status_history.length})</h3>
      <table>
        <thead><tr>
          <th>Seq</th><th>ID</th><th>Family</th><th>Gen</th><th>Prior</th><th>New</th><th>Reason</th>
        </tr></thead>
        <tbody>${report.candidate_status_history.slice(-80).map(e => `<tr>
          <td>${e.sequence}</td>
          <td class="mono">${e.candidate_id}</td>
          <td>${e.family_id || ""}</td>
          <td>${fmtVal(e.generation)}</td>
          <td class="mono">${fmtVal(e.prior_status)}</td>
          <td class="mono">${fmtVal(e.new_status)}</td>
          <td class="mono">${fmtVal(e.reason)}</td>
        </tr>`).join("")}</tbody>
      </table>
    </div>` : ""}
    ${(report.evaluation_backend === "synthetic_oos_probe") ? `<div class="warn-box">Synthetic OOS probe — not a real-data institutional evaluation.</div>` : ""}
    ${report.silver_resolution ? `<div class="panel"><h3>Silver artifacts</h3><pre class="mono">${JSON.stringify(report.silver_resolution, null, 2)}</pre></div>` : ""}
    ${report.wfo_summary ? `<div class="panel"><h3>WFO</h3><pre class="mono">${JSON.stringify(report.wfo_summary, null, 2)}</pre></div>` : ""}
    <div class="grid">
      <div class="stat"><div class="label">Generated</div><div class="value">${report.generation_accounting?.generated_total ?? report.budget_allocation?.generated_total ?? report.budget_allocation?.campaign_generated ?? report.generated_candidates ?? run.generated_count ?? 0}</div></div>
      <div class="stat"><div class="label">Invalid</div><div class="value">${report.invalid_candidates ?? report.invalid ?? 0}</div></div>
      <div class="stat"><div class="label">Duplicates</div><div class="value">${report.duplicate_candidates ?? 0}</div></div>
      <div class="stat"><div class="label">Precheck rejected</div><div class="value">${report.precheck_rejected ?? 0}</div></div>
      <div class="stat"><div class="label">Evaluated</div><div class="value">${report.evaluated_candidates ?? run.evaluated_count ?? 0}</div></div>
      <div class="stat"><div class="label">Full WFO</div><div class="value">${Number(report.full_wfo_evaluations ?? report.full_wfo_evaluated ?? 0)}</div></div>
      <div class="stat"><div class="label">Score qualified</div><div class="value">${Number(report.score_qualified ?? 0)}</div></div>
      <div class="stat"><div class="label">MaxDD limit</div><div class="value" style="font-size:0.85rem">${(() => {
        const lim = report.search_budget_consumed?.max_oos_drawdown
          ?? report.frozen_search_budget?.effective?.max_oos_drawdown
          ?? report.frozen_search_budget?.requested?.max_oos_drawdown;
        return lim == null ? "—" : `${(Number(lim)*100).toFixed(1)}%`;
      })()}</div></div>
      <div class="stat"><div class="label">Stress passed</div><div class="value">${Number(report.stress_passed ?? 0)}</div></div>
      <div class="stat"><div class="label">Clusters</div><div class="value">${Number(report.behavioral_clusters ?? (report.clusters || []).length ?? 0)}</div></div>
      <div class="stat"><div class="label">Shortlisted</div><div class="value">${Number(report.shortlisted ?? (report.research_shortlist || []).length ?? 0)}</div></div>
      <div class="stat"><div class="label">Finalists</div><div class="value">${Number(report.finalist_count ?? (typeof report.finalists === "number" ? report.finalists : 0) ?? run.finalist_count ?? 0)}</div></div>
      <div class="stat"><div class="label">Duplicates</div><div class="value">${Number(report.duplicate_candidates ?? report.duplicates ?? 0)}</div></div>
      <div class="stat"><div class="label">Elapsed s</div><div class="value">${Number(report.elapsed_time ?? run.elapsed_seconds ?? 0).toFixed(2)}</div></div>
      <div class="stat"><div class="label">Throughput /s</div><div class="value">${Number(report.throughput_per_second ?? 0).toFixed(2)}</div></div>
      <div class="stat"><div class="label">Budget gen</div><div class="value" style="font-size:0.9rem">${budget.generated ?? 0}/${budget.generated_cap ?? "?"}</div></div>
      <div class="stat"><div class="label">Family cap</div><div class="value" style="font-size:0.85rem">${budget.max_candidates_per_family_requested ?? "∅"}→${budget.max_candidates_per_family_effective ?? budget.bucket_caps_effective?.max_candidates_per_family ?? "?"}</div></div>
      <div class="stat"><div class="label">Complexity tier cap</div><div class="value" style="font-size:0.85rem">${budget.max_candidates_per_complexity_tier_requested ?? "∅"}→${budget.max_candidates_per_complexity_tier_effective ?? budget.bucket_caps_effective?.max_candidates_per_complexity_tier ?? "?"}</div></div>
      <div class="stat"><div class="label">Feature family cap</div><div class="value" style="font-size:0.85rem">${budget.max_candidates_per_feature_family_requested ?? "∅"}→${budget.max_candidates_per_feature_family_effective ?? budget.bucket_caps_effective?.max_candidates_per_feature_family ?? "?"}</div></div>
      <div class="stat"><div class="label">Terminal reason</div><div class="value" style="font-size:0.75rem">${statusText(report.terminal_reason || run.terminal_reason, "NONE")}</div></div>
      <div class="stat"><div class="label">Discovery</div><div class="value" style="font-size:0.8rem">${statusText(report.discovery_result, "NOT_APPLICABLE")}</div></div>
      <div class="stat"><div class="label">Statistical</div><div class="value" style="font-size:0.8rem">${statusText(report.statistical_result)}</div></div>
    </div>
    <div class="panel">
      <h3>Qualified finalists (count=${finalists.length})</h3>
      <table><thead><tr><th>ID</th><th>Family</th><th>Fitness</th><th>OOS Exp</th><th>PF</th><th>MaxDD</th><th>Stress</th><th>DSR</th><th>PBO</th><th>Cluster</th><th>Next gate</th></tr></thead>
      <tbody>${finalists.map(c => `<tr>
        <td class="mono">${c.candidate_id}</td><td>${c.family||c.strategy_family}</td>
        <td>${fmtVal(c.fitness)}</td><td>${fmtVal(c.median_oos_expectancy)}</td>
        <td>${fmtVal(c.profit_factor)}</td><td>${fmtVal(c.max_drawdown)}</td>
        <td>${fmtVal(c.stress_status)}</td><td>${fmtVal(c.dsr)}</td><td>${fmtVal(c.pbo)}</td>
        <td>${fmtVal(c.behavioral_cluster)}</td><td>${fmtVal(c.next_missing_gate)}</td>
      </tr>`).join("") || `<tr><td colspan="11">NO_QUALIFIED_CANDIDATE</td></tr>`}</tbody></table>
    </div>
    <div class="panel">
      <h3>Unvalidated shortlist (count=${shortlist.length})</h3>
      <table><thead><tr><th>ID</th><th>Label</th><th>Stage</th><th>Backend</th><th>Missing gate</th><th>Fitness</th></tr></thead>
      <tbody>${shortlist.map(c => `<tr>
        <td class="mono">${c.candidate_id}</td><td>${fmtVal(c.promotion_label)}</td>
        <td>${fmtVal(c.evaluation_stage)}</td><td>${fmtVal(c.backend_kind)}</td>
        <td>${fmtVal(c.next_missing_gate)}</td><td>${fmtVal(c.fitness)}</td>
      </tr>`).join("") || `<tr><td colspan="6">None</td></tr>`}</tbody></table>
    </div>
    <div class="panel">
      <h3>Duplicate candidates (count=${duplicates.length})</h3>
      <table><thead><tr><th>ID</th><th>Reason</th><th>Stage</th></tr></thead>
      <tbody>${duplicates.map(c => `<tr><td class="mono">${c.candidate_id}</td><td>${fmtVal(c.rejection_reason)}</td><td>${fmtVal(c.evaluation_stage)}</td></tr>`).join("") || `<tr><td colspan="3">None</td></tr>`}</tbody></table>
    </div>
    <div class="panel">
      <h3>Evaluation failures (count=${failures.length})</h3>
      <table><thead><tr><th>ID</th><th>Stage</th><th>Reason</th></tr></thead>
      <tbody>${failures.map(c => `<tr><td class="mono">${c.candidate_id}</td><td>${fmtVal(c.evaluation_stage)}</td><td>${fmtVal(c.evaluation_error_reason || c.rejection_reason)}</td></tr>`).join("") || `<tr><td colspan="3">None</td></tr>`}</tbody></table>
    </div>
    <div class="panel">
      <h3>Rejected candidates</h3>
      <input id="cand_filter" placeholder="Filter…" value="${state.filter}" />
      <table>
        <thead><tr>
          <th data-k="candidate_id">Candidate</th>
          <th data-k="family">Family</th>
          <th data-k="generation">Gen</th>
          <th data-k="fitness">Fitness</th>
          <th data-k="dsr">DSR</th>
          <th data-k="pbo">PBO</th>
          <th data-k="stress_status">Stress</th>
          <th data-k="rejection_reason">Rejection</th>
          <th data-k="evaluation_stage">Stage</th>
        </tr></thead>
        <tbody>${(state.filter ? rows : rejected).filter(c => c.rejected).map(c => `<tr>
          <td class="mono">${c.candidate_id}</td>
          <td>${c.family||c.strategy_family||""}</td>
          <td>${fmtVal(c.generation)}</td>
          <td>${fmtVal(c.fitness)}</td>
          <td>${fmtVal(c.dsr)}</td>
          <td>${fmtVal(c.pbo)}</td>
          <td>${fmtVal(c.stress_status)}</td>
          <td>${fmtVal(c.rejection_reason)}</td>
          <td>${fmtVal(c.evaluation_stage)}</td>
        </tr>`).join("") || `<tr><td colspan="9">No rejected candidates</td></tr>`}</tbody>
      </table>
    </div>
    <div class="panel">
      <h3>All candidates</h3>
      <table>
        <thead><tr>
          <th>ID</th><th>Family</th><th>Parents</th><th>Complexity</th><th>Fitness</th>
          <th>OOS Exp $</th><th>PF</th><th>MaxDD %</th><th>Calmar</th>
          <th>Entry signals</th><th>Fills</th><th>OOS trades</th>
          <th>DSR</th><th>PBO</th><th>Cluster</th><th>Stage</th><th>Error / reject</th>
        </tr></thead>
        <tbody>${rows.map(c => `<tr>
          <td class="mono">${c.candidate_id}</td>
          <td>${c.family||c.strategy_family||""}</td>
          <td class="mono">${(c.parent_ids||[]).join(",") || "NONE"}</td>
          <td>${fmtMetric(c.complexity, 4)}</td>
          <td>${fmtMetric(c.fitness, 4)}</td>
          <td>${fmtMetric(c.median_oos_expectancy, 6)}</td>
          <td>${fmtMetric(c.profit_factor, 4)}</td>
          <td>${fmtPercentMetric(c.max_drawdown_pct, 4)}</td>
          <td>${fmtMetric(c.calmar ?? c.calmar_mar, 4)}</td>
          <td>${fmtVal(c.entry_true_count)}</td>
          <td>${fmtVal(c.fills)}</td>
          <td>${fmtVal(c.total_oos_trades)}</td>
          <td>${fmtVal(c.dsr)}</td>
          <td>${fmtVal(c.pbo)}</td>
          <td>${fmtVal(c.behavioral_cluster)}</td>
          <td>${fmtVal(c.evaluation_stage)}</td>
          <td>${c.evaluation_stage === "EVALUATION_ERROR"
            ? fmtVal(c.evaluation_error_reason || c.rejection_reason)
            : (c.rejected ? fmtVal(c.rejection_reason) : "—")}</td>
        </tr>`).join("") || `<tr><td colspan="17">No candidates</td></tr>`}</tbody>
      </table>
      <p class="sub">Registry total ${cands.registry_total} · rejected visible=${cands.rejected_visible}</p>
    </div>
    <div class="row">
      <div class="panel"><h3>Rejection distribution</h3>
        <p class="sub">SCORE_QUALIFIED requires expectancy&gt;0, PF&gt;1, OOS trades≥min, |MaxDD|≤limit. Rejects: NEGATIVE_EXPECTANCY / PF_BELOW_ONE / INSUFFICIENT_OOS_TRADES / MAX_DRAWDOWN_EXCEEDED (no Stress).</p>
        <pre class="logs">${JSON.stringify(dist, null, 2)}</pre>
      </div>
      <div class="panel"><h3>Frozen search budget / bucket caps</h3><pre class="logs">${JSON.stringify({
        frozen: report.frozen_search_budget,
        consumed_caps: {
          family: {
            requested: budget.max_candidates_per_family_requested ?? null,
            effective: budget.max_candidates_per_family_effective ?? budget.bucket_caps_effective?.max_candidates_per_family,
          },
          complexity_tier: {
            requested: budget.max_candidates_per_complexity_tier_requested ?? null,
            effective: budget.max_candidates_per_complexity_tier_effective ?? budget.bucket_caps_effective?.max_candidates_per_complexity_tier,
          },
          feature_family: {
            requested: budget.max_candidates_per_feature_family_requested ?? null,
            effective: budget.max_candidates_per_feature_family_effective ?? budget.bucket_caps_effective?.max_candidates_per_feature_family,
          },
        },
      }, null, 2)}</pre></div>
      <div class="panel"><h3>WFO / Stress / Clusters</h3><pre class="logs">${JSON.stringify({
        wfo: report.wfo_summary, stress: report.stress_summary, clusters: report.behavioral_cluster_summary
      }, null, 2)}</pre></div>
    </div>
    <div class="panel"><h3>Event stream</h3><div class="logs" id="miner_events">Connecting…</div></div>`;
}

function scheduleAlphaRefresh(delay = 900) {
  if (!["alpha-miner", "registry"].includes(state.page) || state.alphaRefreshTimer || state.alphaRefreshInFlight) return;
  // Terminal Alpha Miner pages stay stable — no automatic background refresh loop.
  if (!state.alphaRunActive) return;
  state.alphaRefreshTimer = setTimeout(async () => {
    state.alphaRefreshTimer = null; state.alphaRefreshInFlight = true;
    try { await route({background:true, preserveScroll:true}); } finally { state.alphaRefreshInFlight = false; }
  }, delay);
}

function wireAlpha() {
  const sel = $("#miner_run");
  if (sel) sel.onchange = () => { setRun(sel.value); route(); };
  const f = $("#cand_filter");
  if (f) f.oninput = () => { state.filter = f.value; route(); };
  const id = state.selectedRunId;
  wireContinueSearchDraftInputs(id);
  const cs = $("#cs_continue");
  if (cs) {
    cs.onclick = async () => {
      if (!id) return;
      // Persist visible values immediately before submit.
      const draft = saveContinueSearchDraft(id, {
        runtime: $("#cs_runtime")?.value,
        generated: $("#cs_gen")?.value,
        full_wfo: $("#cs_wfo")?.value,
      });
      cs.disabled = true;
      try {
        const body = {
          search_mode: "EXTEND_BUDGET",
          additional_runtime_seconds: Number(draft.runtime),
          additional_generated_budget: Number(draft.generated),
          additional_full_wfo_budget: Number(draft.full_wfo),
        };
        const run = await api(`/api/runs/${id}/continue_search`, {
          method: "POST",
          body: JSON.stringify(body),
        });
        setRun(run.run_id);
        await route();
      } catch (err) {
        alert(String(err.message || err));
        cs.disabled = false;
      }
    };
  }
  document.querySelectorAll("th[data-k]").forEach(th => {
    th.onclick = () => {
      const k = th.dataset.k;
      if (state.sortKey === k) state.sortDir *= -1;
      else { state.sortKey = k; state.sortDir = 1; }
      route();
    };
  });
  // SSE live updates + reconnect with after_seq — only while the run is active.
  const box = $("#miner_events");
  if (!id || !box) return;
  stopSSE();
  let after = 0;
  const connect = () => {
    if (!state.alphaRunActive) return;
    const es = new EventSource(`/api/runs/${id}/events/stream?after_seq=${after}`);
    state.sse = es;
    es.onmessage = (ev) => {
      try {
        const e = JSON.parse(ev.data);
        after = Math.max(after, e.seq || 0);
        box.textContent = `${e.seq} [${e.event_type}] ${e.message}\n` + box.textContent;
        const live = $("#live_status");
        if (live && e.progress != null) {
          live.textContent = `State: live update · progress ${Number(e.progress).toFixed(0)}% · ${e.event_type}`;
        }
      } catch {}
      if (state.alphaRunActive) scheduleAlphaRefresh(650);
    };
    es.addEventListener("end", () => {
      es.close();
      // Do not schedule another automatic Alpha refresh for terminal runs.
      if (state.alphaRunActive) scheduleAlphaRefresh(50);
    });
    es.onerror = () => {
      es.close();
      // Pull missed events, refresh, then reconnect only while active.
      api(`/api/runs/${id}/events?after_seq=${after}`).then(data => {
        for (const e of (data.events || [])) {
          after = Math.max(after, e.seq || 0);
          box.textContent = `${e.seq} [${e.event_type}] ${e.message}\n` + box.textContent;
        }
      }).catch(() => {}).finally(() => {
        if (state.alphaRunActive) {
          scheduleAlphaRefresh(100);
          setTimeout(connect, 1000);
        }
      });
    };
  };
  // Seed from persistence first
  api(`/api/runs/${id}/events`).then(data => {
    const events = data.events || [];
    box.textContent = events.map(e => `${e.seq} [${e.event_type}] ${e.message}`).join("\n") || "No events";
    after = events.reduce((m, e) => Math.max(m, e.seq || 0), 0);
    if (state.alphaRunActive) {
      connect();
      scheduleAlphaRefresh(1400);
    }
  }).catch(() => { box.textContent = "Failed to load events"; });
}

async function renderWfo() {
  const id = state.selectedRunId;
  if (!id) return `<h1>WFO</h1><p class="sub">Select a run first.</p>`;
  const w = await api(`/api/runs/${id}/wfo`);
  const folds = w.fold_oos || [];
  return `
    <h1>WFO</h1>
    <p class="sub">Training metrics are never labeled as OOS. Ranking source: ${w.ranking_source}</p>
    <div class="warn-box">training_metrics_labeled_as_oos = ${w.training_metrics_labeled_as_oos}</div>
    <div class="grid">
      <div class="stat"><div class="label">Folds</div><div class="value">${w.folds ?? folds.length}</div></div>
      <div class="stat"><div class="label">Agg OOS metric</div><div class="value">${w.aggregated_validation_metric != null ? Number(w.aggregated_validation_metric).toFixed(4) : "NOT_AVAILABLE"}</div></div>
    </div>
    <div class="panel"><table>
      <thead><tr><th>Fold</th><th>Phase</th><th>OOS metric</th><th>Params</th><th>Is training?</th></tr></thead>
      <tbody>${folds.map(f => `<tr>
        <td>${f.fold}</td><td>${f.phase}</td><td>${f.metric}</td>
        <td class="mono">${JSON.stringify(f.params)}</td><td>${f.is_training}</td>
      </tr>`).join("") || `<tr><td colspan="5">No fold rows</td></tr>`}</tbody>
    </table></div>`;
}

async function renderRegistry() { return renderAlpha(); }

async function renderPortfolio() {
  const id = state.selectedRunId;
  if (!id) return `<h1>Portfolio</h1><p class="sub">Select a run first.</p>`;
  const p = await api(`/api/runs/${id}/portfolio`);
  return `<h1>Portfolio</h1><div class="panel"><pre class="logs">${JSON.stringify(p, null, 2)}</pre></div>`;
}

async function renderVault() {
  const id = state.selectedRunId;
  if (!id) return `<h1>Vault</h1><p class="sub">Select a run first.</p>`;
  const v = await api(`/api/runs/${id}/vault`);
  return `
    <h1>Vault</h1>
    <div class="warn-box">Raw Vault bars, features, timestamps, and returns are never exposed.</div>
    <div class="panel"><pre class="logs">${JSON.stringify(v, null, 2)}</pre></div>`;
}

async function renderRuntime() {
  const runs = await api("/api/runs?filter=ALL");
  const rt = (runs.runs || []).filter(r => r.run_type === "SHADOW_RUNTIME" || r.run_type === "PAPER_RUNTIME");
  let detail = "";
  if (state.selectedRunId) {
    const s = await api(`/api/runs/${state.selectedRunId}/summary`);
    detail = `<div class="panel"><pre class="logs">${JSON.stringify(s, null, 2)}</pre></div>`;
  }
  return `
    <h1>Shadow / Paper</h1>
    <div class="warn-box">No live-trading activation button. PAPER remains paper-only.</div>
    <div class="panel"><table>
      <thead><tr><th>Run</th><th>Type</th><th>State</th><th>Paper eligible</th></tr></thead>
      <tbody>${rt.map(r => `<tr data-id="${r.run_id}"><td class="mono">${r.run_id}</td><td>${r.run_type}</td><td>${r.state}</td><td>${r.paper_eligible}</td></tr>`).join("") || `<tr><td colspan="4">None</td></tr>`}</tbody>
    </table></div>${detail}`;
}

async function renderDataQuality() {
  const data = await api("/api/datasets");
  const rows = (data.datasets || []).map(d =>
    `<tr><td class="mono">${d.dataset_id}</td><td>${d.quality_status}</td>
     <td>${d.missing_bar_count}</td><td>${d.duplicate_count}</td>
     <td>${d.research_eligible ? "YES" : "NO"}</td>
     <td>${(d.rejection_reasons||[]).join("; ")}</td></tr>`
  ).join("");
  return `<h1>Data Quality</h1>
    <p class="sub">Catalog quality status for registered datasets.</p>
    <div class="panel"><table>
      <thead><tr><th>Dataset</th><th>Status</th><th>Missing</th><th>Dupes</th><th>Eligible</th><th>Reasons</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="6">No datasets</td></tr>`}</tbody>
    </table></div>`;
}

async function renderDatasets() {
  const data = await api("/api/datasets");
  const rows = (data.datasets || []).map(d => `
    <tr data-id="${d.dataset_id}">
      <td class="mono">${d.dataset_id}</td>
      <td>${d.display_name}</td>
      <td>${d.provider_id}</td>
      <td>${d.asset_class}</td>
      <td>${(d.tradable_contracts||[]).join(",") || (d.symbols||[]).join(",")}</td>
      <td>${d.timeframe}</td>
      <td>${d.row_count}</td>
      <td>${d.quality_status}</td>
      <td>${d.research_eligible ? "YES" : "NO"}</td>
      <td>${d.smoke_test_only ? "SMOKE" : ""}</td>
    </tr>`).join("") || `<tr><td colspan="10">No datasets</td></tr>`;
  return `
    <h1>Dataset Catalog</h1>
    <p class="sub">Import root: <span class="mono">${data.import_root || ""}</span></p>
    <div class="panel"><table>
      <thead><tr>
        <th>ID</th><th>Name</th><th>Provider</th><th>Class</th><th>Contract</th>
        <th>TF</th><th>Rows</th><th>Quality</th><th>Eligible</th><th>Smoke</th>
      </tr></thead>
      <tbody id="ds_rows">${rows}</tbody>
    </table></div>`;
}

async function wireDatasets() {
  const tb = $("#ds_rows");
  if (!tb) return;
  tb.onclick = (ev) => {
    const tr = ev.target.closest("tr[data-id]");
    if (!tr) return;
    setDataset(tr.dataset.id);
    location.hash = "#/dataset-detail";
  };
}

async function renderDatasetDetail() {
  const id = state.selectedDatasetId;
  if (!id) return `<h1>Dataset Detail</h1><p class="sub">Select a dataset from Datasets.</p>`;
  const d = await api(`/api/datasets/${encodeURIComponent(id)}`);
  return `
    <h1>Dataset Detail</h1>
    <p class="sub">${d.display_name} · ${d.research_eligible ? "RESEARCH ELIGIBLE" : "NOT RESEARCH ELIGIBLE"}
      ${d.smoke_test_only ? " · SMOKE TEST ONLY" : ""}</p>
    <div class="grid">
      <div class="stat"><div class="label">Quality</div><div class="value" style="font-size:1rem">${d.quality_status}</div></div>
      <div class="stat"><div class="label">Rows</div><div class="value">${d.row_count}</div></div>
      <div class="stat"><div class="label">Missing bars</div><div class="value">${d.missing_bar_count}</div></div>
      <div class="stat"><div class="label">Duplicates</div><div class="value">${d.duplicate_count}</div></div>
    </div>
    <div class="panel">
      <h3>Manifest / provenance / hashes / schema / coverage</h3>
      <pre class="logs">${JSON.stringify({
        dataset_id: d.dataset_id,
        dataset_version: d.dataset_version,
        provider_id: d.provider_id,
        provider_version: d.provider_version,
        asset_class: d.asset_class,
        symbols: d.symbols,
        tradable_contracts: d.tradable_contracts,
        timeframe: d.timeframe,
        start_timestamp: d.start_timestamp,
        end_timestamp: d.end_timestamp,
        source_timezone: d.source_timezone,
        exchange_or_broker: d.exchange_or_broker,
        volume_type: d.volume_type,
        raw_hash: d.raw_hash,
        normalized_hash: d.normalized_hash,
        correction_version: d.correction_version,
        immutable: d.immutable,
        rejection_reasons: d.rejection_reasons,
        schema: d.schema,
        provenance: d.provenance,
        futures_meta: d.futures_meta,
        cfd_meta: d.cfd_meta,
        continuous_series: d.continuous_series,
        rollover_metadata: d.rollover_metadata,
        quality_report: d.quality_report,
        dependent_run_ids: d.dependent_run_ids,
      }, null, 2)}</pre>
    </div>`;
}

async function renderDataImport() {
  const files = await api("/api/import/files");
  const opts = (files.files || []).map(f =>
    `<option value="${f.relative_path}">${f.relative_path} (${f.size_bytes} B)</option>`
  ).join("") || `<option value="">(no CSV/Parquet under import root)</option>`;
  return `
    <h1>Data Import</h1>
    <p class="sub">Safe root: <span class="mono">${files.import_root}</span> — place files here, then select. No absolute paths, URLs, or pickles.</p>
    <div class="panel">
      <div class="row">
        <div><label>File</label><select id="imp_file">${opts}</select></div>
        <div><label>Asset class</label>
          <select id="imp_ac"><option>FUTURES</option><option>CFD</option><option>EQUITY</option><option>FX</option></select>
        </div>
      </div>
      <h3>Column mapping (explicit)</h3>
      <div class="row">
        <div><label>timestamp</label><input id="m_ts" value="timestamp" /></div>
        <div><label>open</label><input id="m_o" value="open" /></div>
        <div><label>high</label><input id="m_h" value="high" /></div>
        <div><label>low</label><input id="m_l" value="low" /></div>
      </div>
      <div class="row">
        <div><label>close</label><input id="m_c" value="close" /></div>
        <div><label>volume</label><input id="m_v" value="volume" /></div>
        <div><label>symbol col (opt)</label><input id="m_sym" value="" /></div>
        <div><label>contract col (opt)</label><input id="m_ctr" value="" /></div>
      </div>
      <h3>Identity (never guessed)</h3>
      <div class="row">
        <div><label>provider_id</label><input id="imp_prov" value="" placeholder="e.g. local_csv" /></div>
        <div><label>provider_version</label><input id="imp_prov_v" value="1.0.0" /></div>
        <div><label>source_timezone</label><input id="imp_tz" value="America/Chicago" /></div>
        <div><label>exchange_or_broker</label><input id="imp_xch" value="CME" /></div>
      </div>
      <div class="row">
        <div><label>timeframe</label><input id="imp_tf" value="5min" /></div>
        <div><label>volume_type</label>
          <select id="imp_vol">
            <option>EXCHANGE_EXECUTED_VOLUME</option>
            <option>BROKER_REPORTED_VOLUME</option>
            <option>TICK_ACTIVITY_PROXY</option>
            <option>UNKNOWN</option>
          </select>
        </div>
        <div><label>root_symbol</label><input id="imp_root" value="ES" /></div>
        <div><label>tradable_contract</label><input id="imp_contract" value="ESH24" /></div>
      </div>
      <div class="row" id="fut_extra">
        <div><label>multiplier</label><input id="imp_mult" type="number" value="50" /></div>
        <div><label>tick_size</label><input id="imp_tick" type="number" step="0.01" value="0.25" /></div>
        <div><label>tick_value</label><input id="imp_tickv" type="number" step="0.01" value="12.5" /></div>
        <div><label>expiration (opt)</label><input id="imp_exp" value="" /></div>
      </div>
      <div class="row" id="cfd_extra" style="display:none">
        <div><label>broker_symbol</label><input id="imp_bsym" value="" /></div>
        <div><label>spread_available</label><select id="imp_spread"><option value="">(required)</option><option>true</option><option>false</option></select></div>
        <div><label>financing_rate_available</label><select id="imp_fin"><option value="">(required)</option><option>true</option><option>false</option></select></div>
      </div>
      <div class="confirm"><input type="checkbox" id="imp_cont" /><span>Continuous series (requires rollover_metadata JSON)</span></div>
      <label>rollover_metadata JSON</label>
      <textarea id="imp_roll" rows="2">{}</textarea>
      <label>display_name</label>
      <input id="imp_name" value="" />
      <div style="margin-top:0.75rem">
        <button type="button" id="btn_preview" class="secondary">Preview</button>
        <button type="button" id="btn_validate" class="secondary">Validate</button>
        <button type="button" id="btn_commit">Confirm &amp; commit (Bronze→Silver→Catalog)</button>
      </div>
      <div class="confirm"><input type="checkbox" id="imp_confirm" /><span>I confirm writing an immutable Bronze dataset</span></div>
      <pre class="logs" id="imp_out"></pre>
    </div>`;
}

function collectImportBody(confirm) {
  let roll = {};
  try { roll = JSON.parse($("#imp_roll").value || "{}"); } catch {}
  const ac = $("#imp_ac").value;
  const body = {
    relative_path: $("#imp_file").value,
    asset_class: ac,
    column_mapping: {
      timestamp: $("#m_ts").value,
      open: $("#m_o").value,
      high: $("#m_h").value,
      low: $("#m_l").value,
      close: $("#m_c").value,
      volume: $("#m_v").value,
      symbol: $("#m_sym").value || null,
      tradable_contract: $("#m_ctr").value || null,
    },
    provider_id: $("#imp_prov").value,
    provider_version: $("#imp_prov_v").value,
    source_timezone: $("#imp_tz").value,
    exchange_or_broker: $("#imp_xch").value,
    timeframe: $("#imp_tf").value,
    volume_type: $("#imp_vol").value,
    root_symbol: $("#imp_root").value,
    tradable_contract: $("#imp_contract").value || null,
    continuous_series: $("#imp_cont").checked,
    rollover_metadata: roll,
    display_name: $("#imp_name").value || null,
    confirm: !!confirm,
  };
  if (ac === "FUTURES") {
    body.multiplier = Number($("#imp_mult").value);
    body.tick_size = Number($("#imp_tick").value);
    body.tick_value = Number($("#imp_tickv").value);
    body.contract_expiration = $("#imp_exp").value || null;
  }
  if (ac === "CFD") {
    body.broker_symbol = $("#imp_bsym").value || null;
    body.spread_available = $("#imp_spread").value === "" ? null : $("#imp_spread").value === "true";
    body.financing_rate_available = $("#imp_fin").value === "" ? null : $("#imp_fin").value === "true";
  }
  return body;
}

async function wireDataImport() {
  const toggle = () => {
    const ac = $("#imp_ac").value;
    $("#fut_extra").style.display = ac === "FUTURES" ? "" : "none";
    $("#cfd_extra").style.display = ac === "CFD" ? "" : "none";
    if (ac === "CFD") $("#imp_vol").value = "BROKER_REPORTED_VOLUME";
  };
  $("#imp_ac").onchange = toggle;
  toggle();
  $("#btn_preview").onclick = async () => {
    try {
      const body = collectImportBody(false);
      const r = await api("/api/import/preview", { method: "POST", body: JSON.stringify(body) });
      $("#imp_out").textContent = JSON.stringify(r, null, 2);
    } catch (e) { $("#imp_out").textContent = String(e.message || e); }
  };
  $("#btn_validate").onclick = async () => {
    try {
      const r = await api("/api/import/validate", { method: "POST", body: JSON.stringify(collectImportBody(false)) });
      $("#imp_out").textContent = JSON.stringify(r, null, 2);
    } catch (e) { $("#imp_out").textContent = String(e.message || e); }
  };
  $("#btn_commit").onclick = async () => {
    try {
      if (!$("#imp_confirm").checked) throw new Error("Check confirmation before commit");
      const entry = await api("/api/import/commit", { method: "POST", body: JSON.stringify(collectImportBody(true)) });
      setDataset(entry.dataset_id);
      $("#imp_out").textContent = JSON.stringify(entry, null, 2);
    } catch (e) { $("#imp_out").textContent = String(e.message || e); }
  };
}

const CFD_BANNER = "DUKASCOPY USA500 DATA IS BROKER CFD DATA. IT IS NOT CME ES FUTURES DATA.";

function acqJobRow(j) {
  return `<tr data-job="${j.download_job_id}">
    <td class="mono">${j.download_job_id}</td>
    <td>${j.state}</td>
    <td>${j.stage}</td>
    <td>${j.acquisition_mode}</td>
    <td class="mono">${j.source_symbol}</td>
    <td>${j.start_date} → ${j.end_date}</td>
    <td>${j.requested_event_type} / ${j.requested_granularity}</td>
    <td class="mono">${j.current_chunk || "-"}</td>
    <td>${j.completed_chunks}/${j.total_chunks} (${Number(j.progress_pct).toFixed(1)}%)</td>
    <td>${j.failed_chunks}</td>
    <td>${j.retries}</td>
    <td>${j.downloaded_bytes}</td>
    <td>${j.registered_dataset_id || "-"}</td>
  </tr>`;
}

async function renderDataAcquisition() {
  const d = await api("/api/acquisition/source");
  const s = d.source;
  const auto = d.automated_download;
  const symOpts = (d.verified_symbols || [])
    .map(v => `<option value="${v.source_symbol}">${v.source_symbol} — ${v.display_symbol}</option>`)
    .join("");
  const modeOpts = (d.acquisition_modes || [])
    .map(m => `<option value="${m}"${m === "MANUAL_EXPORT_IMPORT" ? " selected" : ""}>${m}</option>`)
    .join("");
  const granOpts = (d.granularities || []).map(g => `<option value="${g}">${g}</option>`).join("");
  const jobRows = (d.jobs || []).map(acqJobRow).join("")
    || `<tr><td colspan="13">No download jobs yet</td></tr>`;

  return `
    <h1>Data Acquisition</h1>
    <div class="warn-box"><strong>${CFD_BANNER}</strong></div>
    <p class="sub">Staged acquisition: probe one day, validate one month, then expand. Multi-year spans are never started implicitly.</p>

    <div class="panel">
      <h3>Historical source</h3>
      <div class="grid">
        <div class="stat"><div class="label">Source</div><div class="value" style="font-size:0.95rem">${s.broker_or_venue}</div></div>
        <div class="stat"><div class="label">Instrument</div><div class="value" style="font-size:0.95rem">${s.instrument}</div></div>
        <div class="stat"><div class="label">Exact source symbol</div><div class="value" style="font-size:0.95rem">${s.exact_source_symbol}</div></div>
        <div class="stat"><div class="label">Display symbol</div><div class="value" style="font-size:0.95rem">${s.source_display_symbol}</div></div>
        <div class="stat"><div class="label">Asset class</div><div class="value" style="font-size:0.95rem">${s.asset_class}</div></div>
        <div class="stat"><div class="label">Source timezone</div><div class="value" style="font-size:0.95rem">${s.source_timezone}</div></div>
        <div class="stat"><div class="label">Price convention</div><div class="value" style="font-size:0.95rem">${s.price_convention}</div></div>
        <div class="stat"><div class="label">Volume type</div><div class="value" style="font-size:0.95rem">${s.volume_type}</div></div>
        <div class="stat"><div class="label">Exchange volume</div><div class="value" style="font-size:0.95rem">${s.centralized_exchange_volume ? "YES" : "NO"}</div></div>
        <div class="stat"><div class="label">Futures rollover</div><div class="value" style="font-size:0.95rem">${s.futures_rollover}</div></div>
      </div>
      <p class="sub" style="margin-top:0.75rem">Event types: <span class="mono">${(d.event_types || []).join(", ")}</span> ·
        Availability: <span class="mono">${JSON.stringify(s.historical_availability)}</span></p>
    </div>

    <div class="panel">
      <h3>Acquisition mode</h3>
      <p class="sub">Automated public download enabled: <strong>${auto.enabled ? "YES" : "NO"}</strong>
        (opt-in flag <span class="mono">${auto.network_opt_in_flag}</span>).
        Local <span class="mono">.bi5</span> archive configured: <strong>${d.archive_root_configured ? "YES" : "NO"}</strong>
        (<span class="mono">${d.archive_root_env}</span>).
        Manual export/import is always available under <span class="mono">${d.import_root}</span>.</p>
      <div class="row">
        <div><label>Source symbol</label><select id="acq_symbol">${symOpts}</select></div>
        <div><label>Acquisition mode</label><select id="acq_mode">${modeOpts}</select></div>
      </div>
      <div class="row">
        <div><label>Start date</label><input id="acq_start" value="2024-03-04" /></div>
        <div><label>End date</label><input id="acq_end" value="2024-03-04" /></div>
      </div>
      <div class="row">
        <div><label>Granularity</label><select id="acq_gran">${granOpts}</select></div>
        <div><label>Manual export (optional, relative to import root)</label><input id="acq_file" placeholder="usa500_ticks.csv" /></div>
      </div>
      <div class="row">
        <div><button id="acq_probe">Probe One Day</button></div>
        <div><button id="acq_create">Create Download Job</button></div>
      </div>
    </div>

    <div class="panel">
      <h3>Probe result — sample rows, quality, estimates</h3>
      <pre class="logs" id="acq_probe_out">Run "Probe One Day" to parse a single trading day. Nothing is stored until you register.</pre>
    </div>

    <div class="panel">
      <h3>Download jobs</h3>
      <table>
        <thead><tr>
          <th>Job</th><th>State</th><th>Stage</th><th>Mode</th><th>Symbol</th><th>Range</th>
          <th>Event/Gran</th><th>Chunk</th><th>Progress</th><th>Failed</th><th>Retries</th><th>Bytes</th><th>Dataset</th>
        </tr></thead>
        <tbody id="acq_jobs">${jobRows}</tbody>
      </table>
      <div class="row" style="margin-top:0.75rem">
        <div><label>Selected job id</label><input id="acq_job_id" placeholder="click a row above" /></div>
        <div><label>&nbsp;</label>
          <button id="acq_run">Run / Resume Chunks</button>
        </div>
      </div>
      <div class="row">
        <div><button id="acq_pause">Pause</button></div>
        <div><button id="acq_resume">Resume</button></div>
      </div>
      <div class="row">
        <div><button id="acq_cancel">Cancel</button></div>
        <div><button id="acq_register">Register Dataset</button></div>
      </div>
      <div class="confirm">
        <input type="checkbox" id="acq_confirm" />
        <label for="acq_confirm">I confirm writing an immutable Bronze dataset and registering it in the catalog.</label>
      </div>
      <pre class="logs" id="acq_out">No action yet.</pre>
    </div>`;
}

async function wireDataAcquisition() {
  const out = $("#acq_out");
  const probeOut = $("#acq_probe_out");
  const show = (el, v) => { el.textContent = typeof v === "string" ? v : JSON.stringify(v, null, 2); };
  const jobId = () => ($("#acq_job_id").value || "").trim();

  const jobs = $("#acq_jobs");
  if (jobs) {
    jobs.onclick = (ev) => {
      const tr = ev.target.closest("tr[data-job]");
      if (!tr) return;
      $("#acq_job_id").value = tr.dataset.job;
    };
  }

  $("#acq_probe").onclick = async () => {
    show(probeOut, "Probing…");
    try {
      const body = {
        symbol: $("#acq_symbol").value,
        day: $("#acq_start").value,
      };
      const file = ($("#acq_file").value || "").trim();
      if (file) body.relative_path = file;
      const id = jobId();
      if (id) body.job_id = id;
      const r = await api("/api/acquisition/probe", { method: "POST", body: JSON.stringify(body) });
      show(probeOut, {
        exact_source_symbol: r.source_symbol,
        asset_class: r.asset_class,
        source_timezone: r.source_timezone,
        price_convention: r.price_convention,
        volume_type: r.volume_type,
        centralized_exchange_volume: r.centralized_exchange_volume,
        schema: r.schema,
        estimated_row_count: r.estimated_rows_per_day,
        estimated_storage_bytes: r.estimated_storage_bytes,
        quality_status: r.quality.quality_status,
        research_eligible: r.quality.research_eligible,
        rejection_reasons: r.quality.rejection_reasons,
        warnings: r.quality.warnings,
        crossed_quotes: r.quality.crossed_quote_count,
        spread_distribution: r.quality.spread_distribution,
        approval_required: r.approval_required,
        sample_rows: r.sample_rows,
      });
    } catch (e) { show(probeOut, `ERROR: ${e.message}`); }
  };

  $("#acq_create").onclick = async () => {
    show(out, "Creating job…");
    try {
      const r = await api("/api/acquisition/jobs", {
        method: "POST",
        body: JSON.stringify({
          symbol: $("#acq_symbol").value,
          start: $("#acq_start").value,
          end: $("#acq_end").value,
          granularity: $("#acq_gran").value,
          acquisition_mode: $("#acq_mode").value,
        }),
      });
      $("#acq_job_id").value = r.download_job_id;
      show(out, r);
    } catch (e) { show(out, `ERROR: ${e.message}`); }
  };

  const jobAction = (path, label) => async () => {
    const id = jobId();
    if (!id) { show(out, "Select a job id first."); return; }
    show(out, `${label}…`);
    try {
      show(out, await api(`/api/acquisition/jobs/${encodeURIComponent(id)}/${path}`, { method: "POST" }));
    } catch (e) { show(out, `ERROR: ${e.message}`); }
  };
  $("#acq_run").onclick = jobAction("run", "Acquiring chunks");
  $("#acq_pause").onclick = jobAction("pause", "Pausing");
  $("#acq_resume").onclick = jobAction("resume", "Resuming");
  $("#acq_cancel").onclick = jobAction("cancel", "Cancelling");

  $("#acq_register").onclick = async () => {
    show(out, "Registering…");
    try {
      const body = { confirm: $("#acq_confirm").checked };
      const id = jobId();
      if (id) body.job_id = id;
      const file = ($("#acq_file").value || "").trim();
      if (file) body.relative_path = file;
      body.symbol = $("#acq_symbol").value;
      const r = await api("/api/acquisition/register", { method: "POST", body: JSON.stringify(body) });
      show(out, {
        dataset_id: r.dataset_id,
        research_eligible: r.research_eligible,
        quality_status: r.quality.quality_status,
        cost_model: r.cost_model.eligibility,
        rejection_reasons: r.catalog_entry.rejection_reasons,
        paper_eligible: r.catalog_entry.paper_eligible,
        paper_eligibility_reason: r.catalog_entry.paper_eligibility_reason,
      });
    } catch (e) { show(out, `ERROR: ${e.message}`); }
  };
}

async function renderSourceTarget() {
  const d = await api("/api/acquisition/source_target");
  const rows = (d.comparison_rows || []).map(r => `
    <tr>
      <td>${r.field}</td>
      <td class="mono">${typeof r.source === "object" ? JSON.stringify(r.source) : r.source}</td>
      <td class="mono">${typeof r.target === "object" ? JSON.stringify(r.target) : r.target}</td>
      <td>${r.matches ? `<span class="pos">MATCH</span>` : `<span class="neg">DIFFERS</span>`}</td>
      <td>${r.verified == null ? "-" : (r.verified ? "VERIFIED" : "UNVERIFIED")}</td>
    </tr>`).join("");
  const warns = (d.warnings || []).map(w => `
    <tr>
      <td class="mono">${w.code}</td>
      <td>${w.severity === "CRITICAL" ? `<span class="neg">${w.severity}</span>` : w.severity}</td>
      <td>${w.message}</td>
    </tr>`).join("") || `<tr><td colspan="3">No warnings</td></tr>`;

  return `
    <h1>Source vs Target</h1>
    <div class="warn-box"><strong>${CFD_BANNER}</strong></div>
    <p class="sub">Historical source broker: <strong>${d.historical_source_broker}</strong> ·
      Target prop broker: <strong>${d.target_prop_broker}</strong> ·
      Target feed imported: <strong>${d.target_feed_imported ? "YES" : "NO"}</strong></p>

    <div class="grid">
      <div class="stat"><div class="label">Warnings</div><div class="value">${d.warning_count}</div></div>
      <div class="stat"><div class="label">Critical</div><div class="value ${d.critical_count ? "neg" : "pos"}">${d.critical_count}</div></div>
      <div class="stat"><div class="label">Research eligible possible</div><div class="value" style="font-size:1rem">YES</div></div>
      <div class="stat"><div class="label">Paper eligible</div>
        <div class="value ${d.paper_eligibility.paper_eligible ? "pos" : "neg"}" style="font-size:1rem">
          ${d.paper_eligibility.paper_eligible ? "YES" : "NO"}</div></div>
    </div>

    <div class="panel">
      <h3>Field comparison</h3>
      <table>
        <thead><tr><th>Field</th><th>Historical source</th><th>Execution target</th><th>Status</th><th>Verified</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>

    <div class="panel">
      <h3>Unresolved warnings</h3>
      <table>
        <thead><tr><th>Code</th><th>Severity</th><th>Detail</th></tr></thead>
        <tbody>${warns}</tbody>
      </table>
      <p class="sub" style="margin-top:0.75rem">${d.paper_eligibility.reason}</p>
      <p class="sub">${d.note}</p>
    </div>`;
}

async function renderHealth() {
  const h = await api("/api/system/health");
  const c = await api("/api/system/capabilities");
  return `<h1>System Health</h1>
    <div class="warn-box">${h.banner}</div>
    <div class="panel"><pre class="logs">${JSON.stringify({health:h, capabilities:c}, null, 2)}</pre></div>`;
}

async function renderArtifacts() {
  const id = state.selectedRunId;
  if (!id) return `<h1>Artifacts</h1><p class="sub">Select a run first.</p>`;
  const a = await api(`/api/runs/${id}/artifacts`);
  return `<h1>Artifacts</h1>
    <p class="sub">Only manifest-registered files are accessible. Path traversal is blocked.</p>
    <div class="panel"><ul>${(a.manifest||[]).map(f => `<li><a href="/api/runs/${id}/artifacts/${encodeURIComponent(f)}" target="_blank">${f}</a></li>`).join("") || "<li>Empty</li>"}</ul></div>`;
}

async function renderLogs() {
  const id = state.selectedRunId;
  if (!id) return `<h1>Logs</h1><p class="sub">Select a run first.</p>`;
  const e = await api(`/api/runs/${id}/events`);
  return `<h1>Logs / Events</h1>
    <p class="sub">Reconnect-safe: pass after_seq to retrieve missed events.</p>
    <div class="logs">${(e.events||[]).map(x => `${x.seq} ${x.timestamp} [${x.event_type}] ${x.message}`).join("\n")}</div>`;
}

const RENDERERS = {
  overview: renderOverview,
  "new-run": async () => {
    const catalog = await api("/api/datasets");
    state.catalog = catalog.datasets || [];
    return newRunForm(state.catalog);
  },
  "data-acquisition": renderDataAcquisition,
  "source-target": renderSourceTarget,
  "data-import": renderDataImport,
  datasets: renderDatasets,
  "dataset-detail": renderDatasetDetail,
  runs: () => renderRunsPage("default"),
  active: () => { setFilter("ACTIVE"); return renderRunsPage("default"); },
  "run-detail": () => renderRunsPage("default"),
  "alpha-miner": renderAlpha,
  wfo: renderWfo,
  registry: renderRegistry,
  portfolio: renderPortfolio,
  vault: renderVault,
  runtime: renderRuntime,
  "data-quality": renderDataQuality,
  health: renderHealth,
  artifacts: renderArtifacts,
  logs: renderLogs,
};

const WIRES = {
  "new-run": wireNewRun,
  "data-acquisition": wireDataAcquisition,
  "data-import": wireDataImport,
  datasets: wireDatasets,
  runs: wireRuns,
  active: wireRuns,
  "run-detail": wireRuns,
  "alpha-miner": wireAlpha,
  registry: wireAlpha,
  runtime: async () => {
    const tb = document.querySelector("tbody");
    if (!tb) return;
    tb.onclick = (ev) => {
      const tr = ev.target.closest("tr[data-id]");
      if (!tr) return;
      setRun(tr.dataset.id);
      route();
    };
  },
};

async function route(options = {}) {
  const background = options && options.background === true;
  const preserveScroll = options && options.preserveScroll === true;
  const savedScrollY = preserveScroll ? window.scrollY : 0;
  stopSSE();
  const raw = location.hash.replace(/^#\/?/, "") || "overview";
  state.page = raw.split("?")[0];
  if (state.page === "active") state.page = "runs";
  nav();
  const app = $("#app");
  if (!background) app.innerHTML = `<p class="sub">Loading…</p>`;
  try {
    const html = await (RENDERERS[state.page] || renderOverview)();
    app.innerHTML = html;
    if (WIRES[state.page]) await WIRES[state.page]();
    if (preserveScroll) requestAnimationFrame(() => window.scrollTo(0, savedScrollY));
  } catch (e) {
    app.innerHTML = `<h1>Error</h1><pre class="logs">${String(e.message || e)}</pre>`;
  }
}

window.addEventListener("hashchange", route);
route();
