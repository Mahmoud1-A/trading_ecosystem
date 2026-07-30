from pathlib import Path
import re

R=Path(__file__).resolve().parents[1]

def rw(p):
    q=R/p
    return q,q.read_text(encoding='utf-8')

def put(q,s):
    q.write_text(s,encoding='utf-8'); print('updated',q.relative_to(R))

def one(s,a,b,n):
    if a not in s: raise RuntimeError(f'{n}: marker missing')
    return s.replace(a,b,1)

# dashboard: no cache, one authoritative snapshot, live rerender, explicit units
q,s=rw('quant_framework/control_plane/static/assets/dashboard.js')
if 'alphaRefreshTimer:' not in s:
    s=one(s,'  constraints: null,\n','  constraints: null,\n  alphaRefreshTimer: null,\n  alphaRefreshInFlight: false,\n  alphaRunActive: false,\n','state')
if 'cache: "no-store"' not in s:
    s=one(s,'  const r = await fetch(path, {\n','  const r = await fetch(path, {\n    cache: "no-store",\n','no-store')
if 'function fmtMetric(' not in s:
    m='''function fmtVal(v) {\n  if (v == null || v === "" || v === "-") return "NOT_EVALUATED";\n  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(4) : "NOT_EVALUATED";\n  return String(v);\n}\n'''
    x=m+'''function fmtMetric(v, digits = 4) {\n  if (v == null || v === "" || v === "-") return "NOT_AVAILABLE";\n  const n = Number(v);\n  if (!Number.isFinite(n)) return statusText(v, "NOT_AVAILABLE");\n  if (n === 0) return Number(0).toFixed(digits);\n  return Math.abs(n) < 10 ** (-digits) ? n.toExponential(3) : n.toFixed(digits);\n}\nfunction fmtPercentMetric(v, digits = 3) {\n  if (v == null || v === "" || v === "-") return "NOT_AVAILABLE";\n  const n = Number(v);\n  return Number.isFinite(n) ? `${fmtMetric(n, digits)}%` : statusText(v, "NOT_AVAILABLE");\n}\n'''
    s=one(s,m,x,'metric formatter')
if 'state.alphaRunActive = ACTIVE.has(run.state);' not in s:
    s=one(s,'  const run = await api(`/api/runs/${id}`);\n','  const run = await api(`/api/runs/${id}`);\n  state.alphaRunActive = ACTIVE.has(run.state);\n','run active')
old='''  let report = {};\n  try { report = await api(`/api/runs/${id}/alpha_miner`); } catch { report = {}; }\n  const cands = await api(`/api/runs/${id}/candidates`);\n  let rows = cands.candidates || [];\n'''
new='''  let report = {};\n  try { report = await api(`/api/runs/${id}/alpha_miner`); } catch { report = {}; }\n  let cands = {candidates: report.candidates || [], tables: report.tables || {}, registry_total: (report.candidates || []).length, rejected_visible: true};\n  if (!Array.isArray(report.candidates) || report.candidates.length === 0) {\n    try { cands = await api(`/api/runs/${id}/candidates`); } catch {}\n  }\n  let rows = cands.candidates || [];\n'''
if old in s: s=s.replace(old,new,1)
s=s.replace('<th>OOS Exp</th><th>PF</th><th>MaxDD</th><th>Calmar</th>','<th>OOS Exp $</th><th>PF</th><th>MaxDD %</th><th>Calmar</th><th>OOS trades</th>')
s=s.replace('''          <td>${fmtVal(c.complexity)}</td>\n          <td>${fmtVal(c.fitness)}</td>\n          <td>${fmtVal(c.median_oos_expectancy)}</td>\n          <td>${fmtVal(c.profit_factor)}</td>\n          <td>${fmtVal(c.max_drawdown)}</td>\n          <td>${fmtVal(c.calmar_mar)}</td>\n          <td>${fmtVal(c.dsr)}</td>\n''','''          <td>${fmtMetric(c.complexity, 4)}</td>\n          <td>${fmtMetric(c.fitness, 4)}</td>\n          <td>${fmtMetric(c.median_oos_expectancy, 6)}</td>\n          <td>${fmtMetric(c.profit_factor, 4)}</td>\n          <td>${fmtPercentMetric(c.max_drawdown_pct, 4)}</td>\n          <td>${fmtMetric(c.calmar ?? c.calmar_mar, 4)}</td>\n          <td>${fmtVal(c.total_oos_trades)}</td>\n          <td>${fmtVal(c.dsr)}</td>\n''')
s=s.replace('colspan="13">No candidates','colspan="14">No candidates')
if 'function scheduleAlphaRefresh(' not in s:
    s=one(s,'\nfunction wireAlpha() {\n','''\nfunction scheduleAlphaRefresh(delay = 900) {\n  if (!["alpha-miner", "registry"].includes(state.page) || state.alphaRefreshTimer || state.alphaRefreshInFlight) return;\n  state.alphaRefreshTimer = setTimeout(async () => {\n    state.alphaRefreshTimer = null; state.alphaRefreshInFlight = true;\n    try { await route({background:true, preserveScroll:true}); } finally { state.alphaRefreshInFlight = false; }\n  }, delay);\n}\n\nfunction wireAlpha() {\n''','scheduler')
if 'scheduleAlphaRefresh(650);' not in s:
    s=one(s,'        } catch {}\n    };\n    es.addEventListener("end", () => { es.close(); });','        } catch {}\n        scheduleAlphaRefresh(650);\n    };\n    es.addEventListener("end", () => { es.close(); scheduleAlphaRefresh(50); });','sse refresh')
s=s.replace('// Reconnect pull missed events then resume','// Pull missed events, refresh, then reconnect')
if 'setTimeout(connect, 1000);' not in s:
    s=s.replace('      }).catch(() => {});\n    };','      }).catch(() => {}).finally(() => { scheduleAlphaRefresh(100); if (state.alphaRunActive) setTimeout(connect, 1000); });\n    };',1)
if 'if (state.alphaRunActive) scheduleAlphaRefresh(1400);' not in s:
    s=one(s,'    connect();\n  }).catch(() => { box.textContent = "Failed to load events"; });','    connect();\n    if (state.alphaRunActive) scheduleAlphaRefresh(1400);\n  }).catch(() => { box.textContent = "Failed to load events"; });','poll')
if 'async function route(options = {})' not in s:
    s=s.replace('async function route() {','async function route(options = {}) {\n  const background = options && options.background === true;\n  const preserveScroll = options && options.preserveScroll === true;\n  const savedScrollY = preserveScroll ? window.scrollY : 0;',1)
    s=s.replace('  app.innerHTML = `<p class="sub">Loading…</p>`;','  if (!background) app.innerHTML = `<p class="sub">Loading…</p>`;',1)
    s=s.replace('    if (WIRES[state.page]) await WIRES[state.page]();','    if (WIRES[state.page]) await WIRES[state.page]();\n    if (preserveScroll) requestAnimationFrame(() => window.scrollTo(0, savedScrollY));',1)
put(q,s)

# API responses must never be cached
q,s=rw('quant_framework/control_plane/app.py')
if 'Cache-Control' not in s:
    m='''        response.headers["X-Research-Only"] = "true"\n        response.headers["X-Live-Trading-Enabled"] = "false"\n'''
    s=one(s,m,m+'''        if request.url.path.startswith("/api/"):\n            response.headers["Cache-Control"] = "no-store, max-age=0"\n            response.headers["Pragma"] = "no-cache"\n            response.headers["Expires"] = "0"\n''','api cache')
put(q,s)

# Correct PerformanceMetrics field names and units; never silently fabricate zero
q,s=rw('quant_framework/discovery/event_wfo_backend.py')
if 'def fold_oos_metrics_from_net(' not in s:
    marker='\n\n@dataclass\nclass EventDrivenDiscoveryBackend:'
    helper='''\n\ndef fold_oos_metrics_from_net(*, fold_id: int, net: dict[str, Any], fallback_metric: float) -> FoldOOSMetrics:\n    required=("expectancy","sharpe","profit_factor","calmar","max_drawdown_pct","drawdown_duration_bars","worst_day_pct","turnover","n_trades")\n    missing=[k for k in required if k not in net]\n    if missing: raise RuntimeError("PERFORMANCE_METRIC_CONTRACT_MISMATCH: missing "+",".join(missing))\n    sharpe=float(net["sharpe"]); sharpe=sharpe if np.isfinite(sharpe) else float(fallback_metric)\n    return FoldOOSMetrics(fold_id=fold_id, expectancy=float(net["expectancy"]), sharpe=sharpe, profit_factor=float(net["profit_factor"]), calmar=float(net["calmar"]), max_drawdown=float(net["max_drawdown_pct"])/100.0, drawdown_duration=float(net["drawdown_duration_bars"]), worst_day=float(net["worst_day_pct"])/100.0, turnover=float(net["turnover"]), prop_breach_prob=float(net.get("prop_breach_prob",0.0) or 0.0), regime_entropy=float(net.get("regime_entropy",1.0) or 1.0), n_trades=int(net["n_trades"]))\n'''
    s=one(s,marker,helper+marker,'metric helper')
old=re.compile(r'''            folds\.append\(\n                FoldOOSMetrics\(\n                    fold_id=fr\.fold_id,.*?\n                \)\n            \)''',re.S)
if 'fold_oos_metrics_from_net(' not in s[s.find('for fr in result.folds'):]:
    s,n=old.subn('''            if val is None:\n                raise RuntimeError(f"PERFORMANCE_METRIC_CONTRACT_MISMATCH: fold {fr.fold_id} has no validation run")\n            folds.append(fold_oos_metrics_from_net(fold_id=fr.fold_id, net=net, fallback_metric=fr.validation_metric))''',s,count=1)
    if n!=1: raise RuntimeError('metric mapping block missing')
put(q,s)

# Candidate rows: expose percent drawdown and actual OOS trade evidence
q,s=rw('quant_framework/control_plane/alpha_results.py')
if 'from statistics import median' not in s: s=s.replace('from collections import Counter\n','from collections import Counter\nfrom statistics import median\n',1)
s=s.replace('f.get("expectancy") or f.get("oos_metric")','f["expectancy"] if "expectancy" in f else f.get("oos_metric")')
if 'trade_counts = [' not in s[s.find('def build_candidate_row'):s.find('def funnel_counters')]:
    s=one(s,'    calmars = [float(f["calmar"]) for f in folds if isinstance(f, dict) and "calmar" in f]\n','    calmars = [float(f["calmar"]) for f in folds if isinstance(f, dict) and "calmar" in f]\n    trade_counts = [int(f["n_trades"]) for f in folds if isinstance(f, dict) and f.get("n_trades") is not None]\n    total_oos_trades = sum(max(0,n) for n in trade_counts)\n','trades')
s=s.replace('''        "profit_factor": float(sum(pfs) / len(pfs)) if pfs else STATUS_NOT_EVALUATED,\n        "max_drawdown": float(min(dds)) if dds else STATUS_NOT_EVALUATED,\n        "calmar_mar": float(sum(calmars) / len(calmars)) if calmars else STATUS_NOT_EVALUATED,\n''','''        "profit_factor": float(median(pfs)) if pfs else STATUS_NOT_EVALUATED,\n        "max_drawdown": float(min(dds)) if dds else STATUS_NOT_EVALUATED,\n        "max_drawdown_pct": float(min(dds)*100.0) if dds else STATUS_NOT_EVALUATED,\n        "calmar": float(median(calmars)) if calmars else STATUS_NOT_EVALUATED,\n        "calmar_mar": float(median(calmars)) if calmars else STATUS_NOT_EVALUATED,\n        "total_oos_trades": int(total_oos_trades) if trade_counts else STATUS_NOT_EVALUATED,\n        "fold_trade_counts": trade_counts,\n''')
put(q,s)

# focused regression tests
p=R/'tests/test_dashboard_live_metrics.py'
p.write_text('''from pathlib import Path\nimport pytest\nfrom discovery.event_wfo_backend import fold_oos_metrics_from_net\nROOT=Path(__file__).resolve().parents[1]\ndef test_metric_mapping():\n f=fold_oos_metrics_from_net(fold_id=1,fallback_metric=0,net={"expectancy":1,"sharpe":.5,"profit_factor":1.2,"calmar":2,"max_drawdown_pct":-2.5,"drawdown_duration_bars":7,"worst_day_pct":-.8,"turnover":1,"n_trades":9})\n assert f.max_drawdown==pytest.approx(-.025); assert f.worst_day==pytest.approx(-.008); assert f.n_trades==9\ndef test_missing_metric_is_error():\n with pytest.raises(RuntimeError,match="PERFORMANCE_METRIC_CONTRACT_MISMATCH"): fold_oos_metrics_from_net(fold_id=0,fallback_metric=0,net={})\ndef test_live_dashboard_contract():\n s=(ROOT/"quant_framework/control_plane/static/assets/dashboard.js").read_text(encoding="utf-8")\n assert "scheduleAlphaRefresh" in s and "report.candidates" in s and "OOS Exp $" in s and "MaxDD %" in s\n''',encoding='utf-8')
print('wrote',p.relative_to(R))
print('Dashboard live metrics fix applied; run focused test then full suite.')
