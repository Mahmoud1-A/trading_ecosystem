from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, *, expected: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0 and new in text:
        return
    if count != expected:
        raise RuntimeError(
            f"{path}: expected {expected} occurrence(s), found {count}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_reproducible_fingerprint() -> None:
    path = ROOT / "quant_framework" / "discovery" / "multi_family_campaign.py"
    text = path.read_text(encoding="utf-8")

    marker = "\n\n@dataclass\nclass FamilyCampaignConfig:"
    helper = '''

# Operational search/session metadata must never contaminate a scientific
# reproducibility fingerprint. Two semantically identical NEW/RESUME runs may
# have different program IDs, cache counters, or pending queues while producing
# exactly the same research artifact.
_REPRODUCIBLE_ALLOCATION_EXCLUDED_KEYS = frozenset(
    {
        "search_program_id",
        "source_run_id",
        "resumed_from_run_id",
        "session_totals",
        "evaluation_cache_hits",
        "evaluation_cache_misses",
        "pending_by_gate",
    }
)


def _reproducible_allocation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return allocation evidence stripped only of run/session volatility."""
    return {
        key: value
        for key, value in payload.items()
        if key not in _REPRODUCIBLE_ALLOCATION_EXCLUDED_KEYS
    }
'''
    if "def _reproducible_allocation_payload(" not in text:
        if marker not in text:
            raise RuntimeError(f"{path}: FamilyCampaignConfig marker missing")
        text = text.replace(marker, helper + marker, 1)

    old = '                "allocation": allocation_payload,\n'
    new = (
        '                "allocation": '
        '_reproducible_allocation_payload(allocation_payload),\n'
    )
    count = text.count(old)
    if count:
        if count != 2:
            raise RuntimeError(
                f"{path}: expected 2 fingerprint allocation entries, found {count}"
            )
        text = text.replace(old, new)
    elif text.count(new) != 2:
        raise RuntimeError(f"{path}: reproducible allocation patch not present")

    path.write_text(text, encoding="utf-8")


def patch_acquisition_tests() -> None:
    dsl = ROOT / "quant_framework" / "tests" / "test_dsl_regime_gate_typing.py"
    old_dsl = '''        parent = json.loads(MULTIYEAR_PARENT.read_text(encoding="utf-8"))
        assert parent.get("state") == "RUNNING"
        assert 2024 not in (parent.get("years") or [])
        pid = parent.get("pid")
        if pid:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert str(pid) in (out.stdout or "")
'''
    new_dsl = '''        parent = json.loads(MULTIYEAR_PARENT.read_text(encoding="utf-8"))
        state = str(parent.get("state") or "")
        assert state in {"RUNNING", "PAUSED", "COMPLETED"}
        assert 2024 not in (parent.get("years") or [])
        pid = parent.get("pid")
        # A completed immutable acquisition is healthy and has no live worker.
        # Only active lifecycle states are required to prove the PID is alive.
        if state in {"RUNNING", "PAUSED"} and pid:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert str(pid) in (out.stdout or "")
'''
    replace_once(dsl, old_dsl, new_dsl)

    real = ROOT / "quant_framework" / "tests" / "test_real_data_alpha_miner_backend.py"
    old_real = '''        parent = json.loads(MULTIYEAR_PARENT.read_text(encoding="utf-8"))
        assert parent.get("state") in {"RUNNING", "PAUSED"}
        # Must not have been rewritten to include 2024
        years = parent.get("years") or []
        assert 2024 not in years
        pid = parent.get("pid")
        if pid:
            alive = False
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                # Windows: os.kill(pid, 0) is unreliable — fall back to tasklist
                import subprocess

                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {int(pid)}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                alive = str(pid) in (out.stdout or "")
            assert alive, f"acquisition pid {pid} not running"
'''
    new_real = '''        parent = json.loads(MULTIYEAR_PARENT.read_text(encoding="utf-8"))
        state = str(parent.get("state") or "")
        assert state in {"RUNNING", "PAUSED", "COMPLETED"}
        # Must not have been rewritten to include protected diagnostic year 2024.
        years = parent.get("years") or []
        assert 2024 not in years
        pid = parent.get("pid")
        if state in {"RUNNING", "PAUSED"} and pid:
            alive = False
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                # Windows: os.kill(pid, 0) is unreliable — fall back to tasklist
                import subprocess

                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {int(pid)}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                alive = str(pid) in (out.stdout or "")
            assert alive, f"acquisition pid {pid} not running"
'''
    replace_once(real, old_real, new_real)


def main() -> None:
    patch_reproducible_fingerprint()
    patch_acquisition_tests()
    print("Full-suite lifecycle and fingerprint regressions fixed.")


if __name__ == "__main__":
    main()
