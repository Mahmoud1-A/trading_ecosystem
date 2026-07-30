# Trading Ecosystem

Multi-agent trend-following stack with a centralized Master Risk Manager, free yfinance data to Parquet, event-driven backtests, and Alpaca Paper execution.

## Quick start

```bash
cd trading_ecosystem
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env

te-ingest --start 2018-01-01
te-backtest --start 2018-01-01
te-discover --start 2018-01-01 --phase all
te-paper --broker sim
te-paper --broker alpaca --strategies configs/strategies.discovered.yaml
te-monitor --port 8080
```

Open http://127.0.0.1:8080 for the dashboard.

## Algorithm discovery

`te-discover` searches simple single-factor strategy families, then anomaly/pattern candidates, with walk-forward evaluation **under HARD prop rules** (`configs/prop_rules.yaml`). Target CAGR (default 30%) is an aspirational rank threshold — risk limits are never relaxed.

- Config: `configs/discovery.yaml`
- Report: `data/processed/discovery_report.json`
- Export: `configs/strategies.discovered.yaml`

Phases: `--phase families|anomaly|all`. Use `--max-candidates N` for smoke runs.

### Continuous evolution (24/7)

```bash
te-discover --continuous
# stop after 3 generations (smoke):
te-discover --continuous --generations 3 --batch-size 8
```

This mode keeps inventing, mutating, and crossing over elite algorithms under HARD prop rules, updates the **Top 50** registry, and rewrites `strategies.discovered.yaml` each generation. Stop with Ctrl+C or create `data/processed/discovery_stop.flag`.

- Absolute Top 50: `data/processed/discovery_top50.json` (`GET /api/top50`)
- Best 3 per sector (independent): `data/processed/discovery_best3_by_sector.json` (`GET /api/best3-by-sector`)

## Architecture

Strategies emit OrderIntent only. The Master Risk Manager is the sole gate before any broker adapter (PaperSimBroker or AlpacaPaperGateway).

Prop-style rules live in configs/prop_rules.yaml (daily/total DD, news windows, sizing, exposure).

## VPS / Docker

```bash
docker compose -f deploy/docker-compose.yml up -d monitor
docker compose -f deploy/docker-compose.yml --profile jobs run --rm ingest
docker compose -f deploy/docker-compose.yml --profile jobs run --rm backtest
```

Systemd unit: deploy/systemd/te-monitor.service.

## Notes

- Copy `.env.example` to `.env` and set `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` for paper orders (`te-paper --broker alpaca`). Keep secrets local — never commit `.env`.
- Optional Telegram alerts: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
- Target returns are research goals, not guarantees. Survival under DD limits comes first.
- If API keys were ever pasted into chat or screenshots, rotate them in the Alpaca dashboard.
