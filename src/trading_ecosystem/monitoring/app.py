from __future__ import annotations

import html
import json
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from trading_ecosystem.common.config import data_dir
from trading_ecosystem.discovery.leaderboard import load_top50, top50_public
from trading_ecosystem.discovery.sector_best import load_sector_best, sector_best_public
from trading_ecosystem.discovery.universe import processed_artifact
from trading_ecosystem.monitoring import control_ops, discovery_control
from trading_ecosystem.monitoring.control_page import CONTROL_HTML
from trading_ecosystem.monitoring.state_store import MONITOR


def _load_paper_states() -> dict[str, Any]:
    """Read sim/alpaca paper state files without importing execution.paper_loop (cycle-safe)."""
    out: dict[str, Any] = {}
    root = data_dir() / "processed"
    for broker, name in (("sim", "paper_state_sim.json"), ("alpaca", "paper_state_alpaca.json")):
        path = root / name
        if not path.exists():
            continue
        try:
            out[broker] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    legacy = root / "paper_state.json"
    if "sim" not in out and legacy.exists():
        try:
            out["sim"] = json.loads(legacy.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return out


def _load_discovery_report() -> dict[str, Any] | None:
    path = processed_artifact("discovery_report")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _hydrate_last_summary() -> None:
    """Refresh discovery boards/book from the active universe artifacts on disk."""
    from trading_ecosystem.discovery.aggregate_book import load_portfolio_book, portfolio_book_path
    from trading_ecosystem.discovery.leaderboard import leaderboard_path
    from trading_ecosystem.discovery.sector_best import sector_best_path
    from trading_ecosystem.discovery.universe import get_active_universe, load_universe

    uid = get_active_universe()
    uni = load_universe(uid)
    snap = MONITOR.snapshot()
    disc = dict(snap.get("discovery") or {})
    patch: dict[str, Any] = {
        "universe": uid,
        "universe_label": uni.get("label_en") or uid,
    }

    try:
        entries = top50_public(load_top50(leaderboard_path(uid)).get("entries") or [])
        if entries:
            patch["top50"] = entries
    except Exception:  # noqa: BLE001
        pass

    try:
        sectors = sector_best_public(
            load_sector_best(sector_best_path(uid)).get("sectors") or {}
        )
        if sectors:
            patch["best3_by_sector"] = sectors
    except Exception:  # noqa: BLE001
        pass

    # Prefer locked portfolio book for the active universe; fall back to report.
    book = load_portfolio_book(portfolio_book_path(uid))
    members = list(book.get("members") or [])
    pm = dict(book.get("metrics") or {})
    if members:
        patch["selected_portfolio"] = [
            {
                "strategy_id": (e.get("candidate") or {}).get("strategy_id"),
                "class_name": (e.get("candidate") or {}).get("class_name"),
                "symbols": (e.get("candidate") or {}).get("symbols") or [],
                "sector": e.get("sector")
                or ((e.get("candidate") or {}).get("symbols") or [None])[0],
                "oos_cagr": (e.get("fitness") or {}).get("oos_cagr"),
                "oos_max_dd": (e.get("fitness") or {}).get("oos_max_dd"),
                "passed": (e.get("fitness") or {}).get("passed"),
                "score": (e.get("fitness") or {}).get("score"),
            }
            for e in members
        ]
        if pm:
            patch["portfolio_metrics"] = {
                **pm,
                "universe": uid,
                "book_score": book.get("score", pm.get("book_score")),
            }
        be = book.get("best_ever") or {}
        if be.get("metrics"):
            patch.setdefault("last_summary", {})
            # keep existing last_summary keys if present
            summary = dict(disc.get("last_summary") or {})
            summary["best_book_cagr"] = (be.get("metrics") or {}).get("cagr")
            summary["best_book_max_dd"] = (be.get("metrics") or {}).get("max_drawdown")
            summary["best_book_mar"] = (be.get("metrics") or {}).get("mar")
            summary["universe"] = uid
            patch["last_summary"] = summary
    else:
        report = _load_discovery_report()
        if report:
            rpm = report.get("portfolio_metrics") or {}
            selected = report.get("selected_portfolio") or []
            if rpm:
                patch["portfolio_metrics"] = {**rpm, "universe": uid}
            if selected:
                patch["selected_portfolio"] = [
                    {
                        "strategy_id": (e.get("candidate") or {}).get("strategy_id"),
                        "class_name": (e.get("candidate") or {}).get("class_name"),
                        "symbols": (e.get("candidate") or {}).get("symbols") or [],
                        "sector": e.get("sector")
                        or ((e.get("candidate") or {}).get("symbols") or [None])[0],
                        "oos_cagr": (e.get("fitness") or {}).get("oos_cagr"),
                        "oos_max_dd": (e.get("fitness") or {}).get("oos_max_dd"),
                        "passed": (e.get("fitness") or {}).get("passed"),
                        "score": (e.get("fitness") or {}).get("score"),
                    }
                    for e in selected
                ]
            if disc.get("status") != "running" and not disc.get("last_summary"):
                summary = report.get("summary")
                if summary:
                    patch["last_summary"] = {**summary, "universe": uid}
                    patch["status"] = disc.get("status") or "done"

    try:
        from trading_ecosystem.discovery.stability import load_readiness

        readiness = load_readiness(uid)
        if readiness:
            patch["readiness"] = readiness
            summary = dict(patch.get("last_summary") or disc.get("last_summary") or {})
            summary["readiness_level"] = readiness.get("level")
            summary["ready_for_paper"] = bool(
                (readiness.get("promotion") or {}).get("allow_freeze_paper")
            )
            patch["last_summary"] = summary
    except Exception:  # noqa: BLE001
        pass

    if patch:
        MONITOR.discovery_patch(**patch)

    # Hydrate paper cards (sim / alpaca) from dedicated state files.
    try:
        papers = _load_paper_states()
        if papers:
            cur = dict((MONITOR.snapshot().get("paper") or {}))
            for broker, st in papers.items():
                cur[broker] = {
                    "equity": st.get("equity"),
                    "cash": st.get("cash"),
                    "ann_return": st.get("ann_return"),
                    "ann_return_min": st.get("ann_return_min"),
                    "ann_return_max": st.get("ann_return_max"),
                    "period_return": st.get("period_return"),
                    "days_elapsed": st.get("days_elapsed"),
                    "positions": st.get("positions"),
                    "updated_at": st.get("updated_at"),
                    "halted": st.get("halted"),
                    "daily_halted": st.get("daily_halted"),
                    "daily_dd_pct": st.get("daily_dd_pct"),
                    "total_dd_pct": st.get("total_dd_pct"),
                    "strategies": st.get("strategies"),
                }
            MONITOR.update(paper=cur)
    except Exception:  # noqa: BLE001
        pass


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{float(v) * 100:.2f}%"


def _readiness_html(readiness: dict[str, Any]) -> str:
    if not readiness:
        return (
            '<div class="sub"><strong>Stability gate</strong> — waiting for first readiness report '
            "(after a discovery generation). API: <code>/api/readiness</code></div>"
        )
    level = str(readiness.get("level") or "not_ready")
    promo = readiness.get("promotion") or {}
    checks = readiness.get("checks") or []
    live_checks = readiness.get("live_checks") or []
    ok_n = sum(1 for c in checks if c.get("ok"))
    rows = ""
    for c in checks:
        mark = "✓" if c.get("ok") else "✗"
        hard = "hard" if c.get("hard", True) else "soft"
        detail = c.get("detail")
        if isinstance(detail, dict):
            detail = ", ".join(f"{k}={v}" for k, v in detail.items())
        rows += (
            f"<tr><td>{mark}</td><td><code>{html.escape(str(c.get('name') or ''))}</code></td>"
            f"<td>{html.escape(hard)}</td>"
            f"<td>{html.escape(str(detail))}</td></tr>"
        )
    live_rows = ""
    for c in live_checks:
        mark = "✓" if c.get("ok") else "✗"
        detail = c.get("detail")
        if isinstance(detail, dict):
            detail = ", ".join(f"{k}={v}" for k, v in detail.items())
        live_rows += (
            f"<tr><td>{mark}</td><td><code>{html.escape(str(c.get('name') or ''))}</code></td>"
            f"<td>{html.escape(str(detail))}</td></tr>"
        )
    return f"""
  <div class="sub">
    <strong>Stability gate / readiness</strong>:
    <code>{html.escape(level)}</code>
    — {html.escape(str(readiness.get('message') or ''))}<br/>
    Freeze paper: <strong>{promo.get('allow_freeze_paper')}</strong> |
    Live: <strong>{promo.get('allow_live')}</strong> |
    Checks: {ok_n}/{len(checks)} |
    API: <code>/api/readiness</code>
  </div>
  <table>
    <thead><tr><th>OK</th><th>Check</th><th>Type</th><th>Detail</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="4">No checks yet</td></tr>'}</tbody>
  </table>
  <p class="sub"><strong>Live tier checks</strong></p>
  <table>
    <thead><tr><th>OK</th><th>Check</th><th>Detail</th></tr></thead>
    <tbody>{live_rows or '<tr><td colspan="3">—</td></tr>'}</tbody>
  </table>
"""


def _discovery_section(disc: dict[str, Any]) -> str:
    status = str(disc.get("status") or "idle")
    running = status == "running"
    percent = float(disc.get("percent") or 0.0)
    message = html.escape(str(disc.get("message") or ""))
    phase = html.escape(str(disc.get("phase") or "—"))
    mode = html.escape(str(disc.get("mode") or "batch"))
    generation = disc.get("generation")
    total_evaluated = disc.get("total_evaluated") or 0
    current = disc.get("current") or 0
    total = disc.get("total") or 0
    eta = html.escape(str(disc.get("eta_human") or ("…" if running else "—")))
    elapsed = disc.get("elapsed_sec")
    avg = disc.get("avg_sec_per_candidate")
    sid = html.escape(str(disc.get("current_strategy_id") or "—"))
    cls = html.escape(str(disc.get("current_class") or "—"))
    sym = html.escape(str(disc.get("current_symbol") or "—"))
    source = html.escape(str(disc.get("current_source") or "—"))
    passed = disc.get("passed_so_far") or 0
    failed = disc.get("failed_so_far") or 0
    funnel = disc.get("funnel") or {}
    fitness_note = html.escape(
        str(disc.get("fitness_folds_note") or "validate + yearly oos only; vault sealed")
    )
    best = _pct(disc.get("best_oos_cagr_so_far"))
    status_label = {
        "idle": "Idle — not searching",
        "running": "Running — evolving / trying algorithms",
        "done": "Done",
        "error": "Error",
    }.get(status, status)

    best_ever = disc.get("best_ever") or {}
    best_ever_html = ""
    if best_ever:
        best_ever_html = f"""
        <div class="sub">
          <strong>Best algorithm so far (hall of fame)</strong><br/>
          {html.escape(str(best_ever.get('class_name') or '—'))}
          on <code>{html.escape(','.join(best_ever.get('symbols') or []) or '—')}</code>
          | OOS CAGR: {_pct(best_ever.get('oos_cagr'))}
          | passed: {best_ever.get('passed')}<br/>
          id: <code>{html.escape(str(best_ever.get('strategy_id') or '—'))}</code>
          | source: {html.escape(str(best_ever.get('source') or '—'))}
        </div>
        """

    summary = disc.get("last_summary") or {}
    summary_html = ""
    if summary:
        summary_html = f"""
        <div class="sub">
          <strong>Last generation / run summary</strong><br/>
          Evaluated (batch report): {summary.get('candidates_evaluated', summary.get('total_evaluated', '—'))} |
          Passed: {summary.get('passed_prop_and_stability', '—')} |
          Target 30% hit: {summary.get('target_reached', '—')}<br/>
          Best passed OOS CAGR: {_pct(summary.get('best_passed_oos_cagr'))} |
          Best any OOS CAGR: {_pct(summary.get('best_oos_cagr_any'))}
        </div>
        """

    pm = disc.get("portfolio_metrics") or {}
    selected = disc.get("selected_portfolio") or []
    sel_rows = ""
    for i, row in enumerate(selected, start=1):
        sector = row.get("sector") or ((row.get("symbols") or [None])[0])
        sel_rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><code>{html.escape(str(sector or ''))}</code></td>"
            f"<td>{html.escape(str(row.get('class_name') or ''))}</td>"
            f"<td>{_pct(row.get('oos_cagr'))}</td>"
            f"<td>{_pct(row.get('oos_max_dd'))}</td>"
            f"<td><code>{html.escape(str(row.get('strategy_id') or ''))}</code></td>"
            f"</tr>"
        )
    portfolio_html = f"""
  <h2>Combined Portfolio — {html.escape(str(disc.get('universe_label') or disc.get('universe') or 'universe'))}</h2>
  <p class="sub">
    Showing the <strong>active discovery universe</strong>
    (<code>{html.escape(str(disc.get('universe') or '—'))}</code>).
    Paper trading sleeve remains the separate ETF book in
    <code>strategies.paper.live.yaml</code>.
    Control: <a href="/control">/control</a>
  </p>
  {_readiness_html(disc.get("readiness") or {})}
  <div class="grid">
    <div class="metric"><div>Book CAGR</div><strong>{_pct(pm.get('cagr'))}</strong></div>
    <div class="metric"><div>Book MaxDD</div><strong>{_pct(pm.get('max_drawdown'))}</strong></div>
    <div class="metric"><div>Book MAR</div><strong>{(f"{float(pm['mar']):.2f}" if pm.get('mar') is not None else '—')}</strong></div>
    <div class="metric"><div>Profit factor</div><strong>{(f"{float(pm['profit_factor']):.2f}" if pm.get('profit_factor') is not None else '—')}</strong></div>
    <div class="metric"><div>Trades</div><strong>{pm.get('num_trades', '—')}</strong></div>
    <div class="metric"><div>Win rate</div><strong>{_pct(pm.get('win_rate'))}</strong></div>
    <div class="metric"><div>Members</div><strong>{pm.get('members', len(selected))}</strong></div>
    <div class="metric"><div>Mode</div><strong>{html.escape(str(pm.get('mode') or '—'))}</strong></div>
    <div class="metric"><div>Cost profile</div><strong>{html.escape(str(pm.get('cost_profile') or disc.get('cost_profile', {}).get('profile') if isinstance(disc.get('cost_profile'), dict) else disc.get('cost_profile') or '—'))}</strong></div>
    <div class="metric"><div>Slip / Taker bps</div><strong>{html.escape(str((disc.get('cost_profile') or {}).get('slippage_bps') if isinstance(disc.get('cost_profile'), dict) else '—'))} / {html.escape(str((disc.get('cost_profile') or {}).get('taker_fee_bps') if isinstance(disc.get('cost_profile'), dict) else pm.get('taker_fee_bps') or '—'))}</strong></div>
    <div class="metric"><div>Swaps accepted</div><strong>{pm.get('swaps_accepted', '—')}</strong></div>
    <div class="metric"><div>Book score</div><strong>{(f"{float(pm['book_score']):.3f}" if pm.get('book_score') is not None else '—')}</strong></div>
    <div class="metric"><div>Best book CAGR</div><strong>{_pct((disc.get('last_summary') or {}).get('best_book_cagr') if disc.get('last_summary') else None)}</strong></div>
  </div>
  <table>
    <thead>
      <tr><th>#</th><th>Sector</th><th>Class (#1)</th><th>OOS CAGR</th><th>OOS MaxDD</th><th>Strategy ID</th></tr>
    </thead>
    <tbody>{sel_rows or '<tr><td colspan="6">Waiting for sector champions portfolio…</td></tr>'}</tbody>
  </table>
"""

    top50 = disc.get("top50") or []
    top_rows = ""
    for row in top50[:50]:
        syms = ",".join(row.get("symbols") or [])
        top_rows += (
            f"<tr>"
            f"<td>{row.get('rank')}</td>"
            f"<td>{html.escape(str(row.get('class_name') or ''))}</td>"
            f"<td><code>{html.escape(syms)}</code></td>"
            f"<td>{_pct(row.get('oos_cagr'))}</td>"
            f"<td>{_pct(row.get('oos_max_dd'))}</td>"
            f"<td>{html.escape(str(row.get('passed')))}</td>"
            f"<td>{html.escape(str(row.get('source') or ''))}</td>"
            f"<td><code>{html.escape(str(row.get('strategy_id') or ''))}</code></td>"
            f"</tr>"
        )
    top50_html = f"""
  <h2>Top 50 Absolute</h2>
  <p class="sub">
    Global best algorithms by score (max 50) — not split by sector.
    File: <code>data/processed/discovery_top50.json</code>
  </p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Class</th><th>Symbol</th><th>OOS CAGR</th><th>OOS MaxDD</th>
        <th>Passed</th><th>Source</th><th>Strategy ID</th>
      </tr>
    </thead>
    <tbody>{top_rows or '<tr><td colspan="8">Empty — run te-discover to populate</td></tr>'}</tbody>
  </table>
"""

    best3 = disc.get("best3_by_sector") or {}
    sector_blocks = ""
    for sym, rows in sorted(best3.items()):
        body = ""
        for row in rows:
            body += (
                f"<tr>"
                f"<td>{row.get('rank_in_sector')}</td>"
                f"<td>{html.escape(str(row.get('class_name') or ''))}</td>"
                f"<td>{_pct(row.get('oos_cagr'))}</td>"
                f"<td>{_pct(row.get('oos_max_dd'))}</td>"
                f"<td>{html.escape(str(row.get('passed')))}</td>"
                f"<td>{html.escape(str(row.get('source') or ''))}</td>"
                f"<td><code>{html.escape(str(row.get('strategy_id') or ''))}</code></td>"
                f"</tr>"
            )
        sector_blocks += f"""
      <h3>{html.escape(str(sym))} — best 3</h3>
      <table>
        <thead>
          <tr><th>#</th><th>Class</th><th>OOS CAGR</th><th>OOS MaxDD</th>
          <th>Passed</th><th>Source</th><th>Strategy ID</th></tr>
        </thead>
        <tbody>{body or '<tr><td colspan="7">Empty</td></tr>'}</tbody>
      </table>
"""
    best3_html = f"""
  <h2>Best 3 by Sector</h2>
  <p class="sub">
    Separate registry: strongest 3 algorithms <strong>for each symbol/sector</strong>,
    independent from the absolute Top 50.
    File: <code>data/processed/discovery_best3_by_sector.json</code>
    | API: <code>/api/best3-by-sector</code>
  </p>
  {sector_blocks or '<p class="sub">Empty — waiting for discovery results per symbol.</p>'}
"""

    bar_class = "bar-run" if running else "bar-idle"
    gen_label = generation if generation is not None else "—"
    eval_n = int(funnel.get("evaluated") or 0)
    elig_n = int(funnel.get("preliminary_eligible") or passed or 0)
    rej_n = int(failed if failed is not None else max(0, eval_n - elig_n))
    # Prefer funnel so Eligible + Rejected always equals Evaluated
    if eval_n:
        rej_n = max(0, eval_n - elig_n)
    funnel_rows = ""
    funnel_labels = [
        ("generated", "Generated"),
        ("evaluated", "Evaluated successfully"),
        ("preliminary_eligible", "Preliminary eligible"),
        ("score_qualified", "Score qualified"),
        ("behaviorally_unique", "Behaviorally unique"),
        ("book_accepted", "Book accepted"),
        ("vault_eligible", "Vault eligible"),
        ("vault_pending", "Vault pending"),
        ("vault_tested", "Vault tested"),
        ("vault_passed_failed", "Vault passed / failed"),
        ("paper_promoted", "Paper promoted"),
    ]
    for key, label in funnel_labels:
        if key == "vault_passed_failed":
            val = f"{int(funnel.get('vault_passed') or 0)} / {int(funnel.get('vault_failed') or 0)}"
        else:
            val = str(int(funnel.get(key) or 0))
        funnel_rows += (
            f"<tr><td>{html.escape(label)}</td>"
            f"<td><strong>{html.escape(val)}</strong></td></tr>"
        )
    funnel_note = (
        "Preliminary eligible = Prop/validate gate only (not vault). "
        "Vault eligible = stability gate passed, may enter sealed vault. "
        "Vault pending = eligible but not tested yet. "
        "Vault tested = sealed vault simulation ran. "
        "Vault passed/failed = vault test outcome. "
        "Paper promoted = actual freeze-to-paper events."
    )
    return f"""
  <h2>Discovery</h2>
  <p class="status-{html.escape(status)}"><strong>{html.escape(status_label)}</strong> — {message}</p>
  <div class="progress-wrap">
    <div class="progress {bar_class}" style="width:{percent:.1f}%"></div>
  </div>
  <div class="grid">
    <div class="metric"><div>Mode</div><strong>{mode}</strong></div>
    <div class="metric"><div>Phase</div><strong>{phase}</strong></div>
    <div class="metric"><div>Workers</div><strong>{disc.get('workers') if disc.get('workers') is not None else '—'}</strong></div>
    <div class="metric"><div>Generation</div><strong>{gen_label}</strong></div>
    <div class="metric"><div>Evaluated (session)</div><strong>{eval_n or total_evaluated}</strong></div>
    <div class="metric"><div>Batch progress</div><strong>{current} / {total} ({percent:.1f}%)</strong></div>
    <div class="metric"><div>ETA (this batch)</div><strong>{eta}</strong></div>
    <div class="metric"><div>Elapsed</div><strong>{elapsed if elapsed is not None else '—'}s</strong></div>
    <div class="metric"><div>Avg / candidate</div><strong>{avg if avg is not None else '—'}s</strong></div>
    <div class="metric"><div>Eligible / Rejected</div><strong>{elig_n} / {rej_n}</strong></div>
    <div class="metric"><div>Best OOS CAGR (session)</div><strong>{best}</strong></div>
    <div class="metric"><div>Top50 count</div><strong>{len(top50)}</strong></div>
    <div class="metric"><div>Sectors tracked</div><strong>{len(best3)}</strong></div>
    <div class="metric"><div>Source</div><strong>{source}</strong></div>
  </div>
  <div class="sub">
    Fitness folds: <code>{fitness_note}</code>
    — Eligible = preliminary gate on <strong>validate + yearly research OOS</strong> only.
    Sealed <strong>Vault</strong> never enters Eligible/Rejected, score, Top50, or book selection.
  </div>
  <h3>Discovery funnel (session)</h3>
  <p class="sub">{html.escape(funnel_note)}</p>
  <table>
    <thead><tr><th>Stage</th><th>Count</th></tr></thead>
    <tbody>{funnel_rows or '<tr><td colspan="2">Waiting for first batch…</td></tr>'}</tbody>
  </table>
  <div class="sub">
    <strong>Currently trying:</strong> {cls} on <code>{sym}</code> via <code>{source}</code><br/>
    Strategy id: <code>{sid}</code>
  </div>
  {best_ever_html}
  {summary_html}
  {portfolio_html}
  {best3_html}
  {top50_html}
"""


def _require_control_token(x_control_token: str | None) -> None:
    if not control_ops.control_token():
        raise HTTPException(
            status_code=403,
            detail="CONTROL_TOKEN is not set in server .env — writes disabled",
        )
    if not control_ops.token_ok(x_control_token):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Control-Token")


def create_app() -> FastAPI:
    MONITOR.load()
    _hydrate_last_summary()
    app = FastAPI(title="Trading Ecosystem Monitor", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/control", response_class=HTMLResponse)
    def control_panel() -> str:
        return CONTROL_HTML

    @app.get("/api/control/status")
    def control_status() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        return {
            "control": {
                "systemd": control_ops.systemd_available(),
                "writes_enabled": bool(control_ops.control_token()),
            },
            "services": {
                "paper": control_ops.unit_status(control_ops.PAPER_UNIT),
                "monitor": control_ops.unit_status(control_ops.MONITOR_UNIT),
            },
            "snapshot": MONITOR.snapshot(),
            "prop": control_ops.load_prop_policy(),
            "sleeve": control_ops.live_sleeve_info(),
            "discovery_control": discovery_control.discovery_process_status(),
        }

    @app.get("/api/control/prop-rules")
    def control_prop_rules() -> dict:
        return control_ops.load_prop_policy()

    @app.get("/api/control/logs")
    def control_logs(lines: int = Query(80, ge=10, le=300)) -> dict:
        return control_ops.paper_logs(lines)

    @app.post("/api/control/paper/{action}")
    def control_paper_action(
        action: str,
        x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    ) -> dict:
        _require_control_token(x_control_token)
        result = control_ops.paper_action(action)
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or result.get("output") or "failed")
        MONITOR.push_alert("info", f"Control panel: paper {action}")
        return result

    @app.get("/api/control/discovery")
    def control_discovery_status() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        return discovery_control.discovery_process_status()

    @app.post("/api/control/discovery/{action}")
    def control_discovery_action(
        action: str,
        universe: str | None = Query(default=None),
        x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    ) -> dict:
        _require_control_token(x_control_token)
        action_l = action.strip().lower()
        if action_l == "start":
            result = discovery_control.start_discovery(universe=universe)
        elif action_l == "stop":
            result = discovery_control.request_discovery_stop()
        elif action_l in {"universe", "select"}:
            if not universe:
                raise HTTPException(status_code=400, detail="universe query param required")
            result = discovery_control.select_universe(universe)
        else:
            raise HTTPException(status_code=400, detail="action must be start|stop|universe")
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error") or "failed")
        return result

    @app.get("/api/state")
    def state() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        return MONITOR.snapshot()

    @app.get("/api/discovery")
    def discovery() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        return MONITOR.snapshot().get("discovery", {})

    @app.get("/api/top50")
    def top50() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        board = load_top50()
        return {
            "updated_at": board.get("updated_at"),
            "size": board.get("size"),
            "count": len(board.get("entries") or []),
            "entries": top50_public(board.get("entries") or []),
        }

    @app.get("/api/portfolio")
    def portfolio() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        disc = MONITOR.snapshot().get("discovery") or {}
        return {
            "portfolio_metrics": disc.get("portfolio_metrics") or {},
            "selected_portfolio": disc.get("selected_portfolio") or [],
            "readiness": disc.get("readiness") or {},
            "source": "discovery_report.json (refreshed each generation)",
        }

    @app.get("/api/readiness")
    def readiness() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        from trading_ecosystem.discovery.stability import load_readiness
        from trading_ecosystem.discovery.universe import get_active_universe

        uid = get_active_universe()
        disc = MONITOR.snapshot().get("discovery") or {}
        payload = disc.get("readiness") or load_readiness(uid)
        return {
            "universe": uid,
            "level": payload.get("level"),
            "ready": payload.get("ready"),
            "message": payload.get("message"),
            "promotion": payload.get("promotion") or {},
            "checks": payload.get("checks") or [],
            "live_checks": payload.get("live_checks") or [],
            "metrics": payload.get("metrics") or {},
            "regimes": payload.get("regimes") or {},
            "updated_at": payload.get("updated_at"),
        }

    @app.get("/api/best3-by-sector")
    def best3_by_sector() -> dict:
        MONITOR.load()
        _hydrate_last_summary()
        board = load_sector_best()
        return {
            "updated_at": board.get("updated_at"),
            "best_per_sector": board.get("best_per_sector"),
            "sector_count": len(board.get("sectors") or {}),
            "sectors": sector_best_public(board.get("sectors") or {}),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        MONITOR.load()
        _hydrate_last_summary()
        snap = MONITOR.snapshot()
        disc = snap.get("discovery") or {}
        refresh = 5 if disc.get("status") == "running" else 15
        rows = "".join(
            f"<tr><td>{html.escape(str(a.get('ts','')))}</td>"
            f"<td>{html.escape(str(a.get('level','')))}</td>"
            f"<td>{html.escape(str(a.get('message','')))}</td></tr>"
            for a in snap.get("alerts", [])
        )
        decisions = "".join(
            f"<li><code>{html.escape(str(d))}</code></li>"
            for d in snap.get("recent_decisions", [])[:10]
        )
        agents = html.escape(str(snap.get("agents")))
        discovery_html = _discovery_section(disc)
        paper = snap.get("paper") or {}
        sim_p = paper.get("sim") or {}
        alp_p = paper.get("alpaca") or {}

        def _ann(v: Any) -> str:
            if v is None:
                return "—"
            return f"{float(v) * 100:.2f}%"

        paper_html = f"""
  <h2>Paper Trading (fake money)</h2>
  <p class="sub">
    Sim = local replay/paper. Alpaca = Alpaca Paper account.
    Live sleeve = top strategies sized to Prop max positions (discovery still searches the full book).
    Ann = annualized return from session start. Min/Max = lowest/highest ann seen so far.
  </p>
  <div class="grid">
    <div class="metric"><div>Sim Equity</div><strong>{sim_p.get('equity', '—')}</strong></div>
    <div class="metric"><div>Sim Ann</div><strong>{_ann(sim_p.get('ann_return'))}</strong></div>
    <div class="metric"><div>Sim Ann Min</div><strong>{_ann(sim_p.get('ann_return_min'))}</strong></div>
    <div class="metric"><div>Sim Ann Max</div><strong>{_ann(sim_p.get('ann_return_max'))}</strong></div>
    <div class="metric"><div>Alpaca Equity</div><strong>{alp_p.get('equity', '—')}</strong></div>
    <div class="metric"><div>Alpaca Ann</div><strong>{_ann(alp_p.get('ann_return'))}</strong></div>
    <div class="metric"><div>Alpaca Ann Min</div><strong>{_ann(alp_p.get('ann_return_min'))}</strong></div>
    <div class="metric"><div>Alpaca Ann Max</div><strong>{_ann(alp_p.get('ann_return_max'))}</strong></div>
  </div>
"""
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>TE Monitor</title>
  <meta http-equiv="refresh" content="{refresh}"/>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; background: #f6f3ee; color: #1c1c1c; }}
    h1 {{ font-size: 1.6rem; }}
    .toplink {{ margin: 0 0 1rem; }}
    .toplink a {{ color: #2f5d50; font-weight: bold; }}
    h2 {{ margin-top: 2rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 1rem; }}
    .metric {{ background: #fff; padding: 1rem; border: 1px solid #ddd; }}
    .warn {{ color: #8a1c1c; font-weight: bold; }}
    .sub {{ margin: 0.8rem 0 0.2rem; padding: 0.8rem; background: #fff; border: 1px solid #ddd; }}
    .progress-wrap {{ background: #ddd; height: 14px; margin: 0.6rem 0 1rem; }}
    .progress {{ height: 14px; }}
    .bar-run {{ background: #2f5d50; }}
    .bar-idle {{ background: #8a8a8a; }}
    .status-running {{ color: #1f4d3a; }}
    .status-done {{ color: #1c1c1c; }}
    .status-error {{ color: #8a1c1c; }}
    .status-idle {{ color: #555; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    td, th {{ border-bottom: 1px solid #ddd; padding: 0.4rem; text-align: left; font-size: 0.9rem; }}
    code {{ font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>Trading Ecosystem Monitor</h1>
  <p class="toplink"><a href="/control">→ لوحة تحكم التداول (تشغيل / إيقاف / MRM)</a></p>
  <div class="grid">
    <div class="metric"><div>Equity</div><strong>{snap.get('equity')}</strong></div>
    <div class="metric"><div>Daily DD %</div><strong class="{'warn' if (snap.get('daily_dd_pct') or 0) > 3 else ''}">{snap.get('daily_dd_pct')}</strong></div>
    <div class="metric"><div>Total DD %</div><strong class="{'warn' if (snap.get('total_dd_pct') or 0) > 7 else ''}">{snap.get('total_dd_pct')}</strong></div>
    <div class="metric"><div>Halted</div><strong>{snap.get('halted')}</strong></div>
    <div class="metric"><div>Daily Halt</div><strong>{snap.get('daily_halted')}</strong></div>
    <div class="metric"><div>Last Bar</div><strong>{html.escape(str(snap.get('last_bar_ts') or '—'))}</strong></div>
  </div>
  {paper_html}
  {discovery_html}
  <h2>Agents</h2>
  <pre>{agents}</pre>
  <h2>Recent Risk Decisions</h2>
  <ul>{decisions or '<li>None</li>'}</ul>
  <h2>Alerts</h2>
  <table><thead><tr><th>Time</th><th>Level</th><th>Message</th></tr></thead><tbody>{rows}</tbody></table>
</body>
</html>"""

    return app
