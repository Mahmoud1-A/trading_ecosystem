from pathlib import Path
import pytest
from discovery.event_wfo_backend import fold_oos_metrics_from_net
ROOT=Path(__file__).resolve().parents[1]
def test_metric_mapping():
 f=fold_oos_metrics_from_net(fold_id=1,fallback_metric=0,net={"expectancy":1,"sharpe":.5,"profit_factor":1.2,"calmar":2,"max_drawdown_pct":-2.5,"drawdown_duration_bars":7,"worst_day_pct":-.8,"turnover":1,"n_trades":9})
 assert f.max_drawdown==pytest.approx(-.025); assert f.worst_day==pytest.approx(-.008); assert f.n_trades==9
def test_missing_metric_is_error():
 with pytest.raises(RuntimeError,match="PERFORMANCE_METRIC_CONTRACT_MISMATCH"): fold_oos_metrics_from_net(fold_id=0,fallback_metric=0,net={})
def test_live_dashboard_contract():
 s=(ROOT/"quant_framework/control_plane/static/assets/dashboard.js").read_text(encoding="utf-8")
 assert "scheduleAlphaRefresh" in s and "report.candidates" in s and "OOS Exp $" in s and "MaxDD %" in s
