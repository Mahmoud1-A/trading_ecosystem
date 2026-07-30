from __future__ import annotations

from trading_ecosystem.discovery.continuous import run_continuous_discovery
from trading_ecosystem.monitoring.state_store import MONITOR


def test_continuous_one_generation(tmp_path):
    report = tmp_path / "report.json"
    export = tmp_path / "strategies.yaml"
    result = run_continuous_discovery(
        start="2022-01-01",
        batch_size=3,
        max_generations=1,
        sleep_sec=0,
        export_path=export,
        report_path=report,
    )
    assert result["summary"]["generation"] == 1
    assert result["summary"]["total_evaluated"] >= 1
    assert report.exists()
    assert export.exists()
    disc = MONITOR.snapshot()["discovery"]
    assert disc.get("mode") == "continuous" or disc.get("last_summary", {}).get("mode") == "continuous"
