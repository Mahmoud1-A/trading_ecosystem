# Quant Framework

Intraday quantitative **research and backtesting** framework for Futures and CFDs under prop-firm risk constraints.

**Status:** Phase 5 complete (WFO, vault, registry, reports).  
**Not live-trading ready.**

## Architecture (hybrid)

1. **Vectorized computation layer** — features, regimes, raw alpha with explicit availability timestamps
2. **Event-driven execution & risk layer** — orders, fills, portfolio, friction, prop FSM

## Phase 5 (complete)

- Rolling walk-forward (train / validate / step) with purge + embargo
- Candidate lineage hashing
- Immutable Validation Vault (one-shot per lineage; blocks optimization/ranking misuse)
- Experiment registry (keeps rejected trials for DSR/PBO)
- Console report + equity curve chart
- Reproducibility checks (same seed → same fills/trades/equity)

## Setup

```bash
cd quant_framework
pip install -e ".[dev]"
pytest -v
python main.py
```

## Roadmap

| Phase | Scope | Status |
|-------|--------|--------|
| 1 | Config, asset specs, data validation | Complete |
| 2 | Event execution, information timing, friction | Complete |
| 3 | Prop risk FSM | Complete |
| 4 | Strategy protocol, regime, sizing, metrics | Complete |
| 5 | Rolling WFO, vault, registry, reports | **Complete** |
