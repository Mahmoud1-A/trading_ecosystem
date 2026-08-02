from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.write_text(text, encoding="utf-8")
    ast.parse(text, filename=str(path))


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def patch_fitness() -> None:
    rel = "quant_framework/discovery/fitness.py"
    text = read(rel)
    text = text.replace(
        "min_median_profit_factor: float = 1.10",
        "min_median_profit_factor: float = 1.0",
    )
    text = text.replace(
        "min_risk_normalized_annual_return: float = 0.12",
        "min_risk_normalized_annual_return: float = 0.0",
    )
    text = replace_once(
        text,
        "        if median_pf < float(self.min_median_profit_factor):\n",
        "        if not (median_pf > float(self.min_median_profit_factor)):\n",
        label="PF compatibility gate",
    )
    write(rel, text)


def patch_regression_test() -> None:
    rel = "tests/test_quant_robust_fitness_economic.py"
    text = read(rel)
    old = "result = RobustFitness(min_total_oos_trades=8).score("
    new = (
        "result = RobustFitness(\n"
        "        min_total_oos_trades=8,\n"
        "        min_risk_normalized_annual_return=0.12,\n"
        "    ).score("
    )
    if text.count(old) == 2:
        text = text.replace(old, new)
    elif text.count(new) != 2:
        raise RuntimeError("economic regression tests: expected two constructors")
    write(rel, text)


def patch_multi_family() -> None:
    rel = "quant_framework/discovery/multi_family_campaign.py"
    text = read(rel)

    config_fields = (
        "    # Economic objective: compatibility defaults stay permissive for direct\n"
        "    # unit construction; family_campaign_from_config enables research defaults.\n"
        "    min_median_profit_factor: float = 1.0\n"
        "    min_risk_normalized_annual_return: float = 0.0\n"
        "    target_risk_drawdown: float = 0.05\n"
        "    max_risk_scale: float = 4.0\n"
        "    drawdown_floor: float = 0.0025\n"
        "    calmar_cap: float = 3.0\n"
    )
    marker = "    max_oos_drawdown: float = 0.20\n"
    if config_fields not in text:
        text = replace_once(text, marker, marker + config_fields, label="economic config fields")

    dict_fields = (
        "            \"min_median_profit_factor\": self.min_median_profit_factor,\n"
        "            \"min_risk_normalized_annual_return\": self.min_risk_normalized_annual_return,\n"
        "            \"target_risk_drawdown\": self.target_risk_drawdown,\n"
        "            \"max_risk_scale\": self.max_risk_scale,\n"
        "            \"drawdown_floor\": self.drawdown_floor,\n"
        "            \"calmar_cap\": self.calmar_cap,\n"
    )
    dict_marker = "            \"max_oos_drawdown\": self.max_oos_drawdown,\n"
    if dict_fields not in text:
        text = replace_once(text, dict_marker, dict_marker + dict_fields, label="economic config serialization")

    helper = '''    def _build_fitness_model(self) -> RobustFitness:\n        """One bounded economic objective for discovery, Stress, and robustness."""\n        cfg = self.config\n        return RobustFitness(\n            min_total_oos_trades=int(cfg.min_oos_trades),\n            min_oos_trades_per_fold=int(cfg.min_oos_trades_per_fold),\n            max_oos_drawdown=float(cfg.max_oos_drawdown),\n            min_median_profit_factor=float(cfg.min_median_profit_factor),\n            min_risk_normalized_annual_return=float(\n                cfg.min_risk_normalized_annual_return\n            ),\n            target_risk_drawdown=float(cfg.target_risk_drawdown),\n            max_risk_scale=float(cfg.max_risk_scale),\n            drawdown_floor=float(cfg.drawdown_floor),\n            calmar_cap=float(cfg.calmar_cap),\n        )\n\n'''
    emit_block = '''    def _emit(self, name: str, payload: dict[str, Any] | None = None) -> None:\n        if self.progress_hook is not None:\n            self.progress_hook(name, payload or {})\n\n'''
    if helper not in text:
        text = replace_once(text, emit_block, emit_block + helper, label="fitness builder")

    pattern = re.compile(
        r"        fitness = RobustFitness\(\n"
        r"            min_total_oos_trades=int\((?:cfg|self\.config)\.min_oos_trades\),\n"
        r"            min_oos_trades_per_fold=int\((?:cfg|self\.config)\.min_oos_trades_per_fold\),\n"
        r"            max_oos_drawdown=float\((?:cfg|self\.config)\.max_oos_drawdown\),\n"
        r"        \)"
    )
    text, count = pattern.subn("        fitness = self._build_fitness_model()", text)
    if count not in {0, 3}:
        raise RuntimeError(f"fitness constructor wiring: expected 3 or 0 replacements, got {count}")
    if count == 0 and text.count("fitness = self._build_fitness_model()") < 3:
        raise RuntimeError("fitness constructor wiring missing")

    tester_anchor = "                synthetic_stress_forbidden=forbid,\n            )"
    tester_repl = "                synthetic_stress_forbidden=forbid,\n                progress_hook=self.progress_hook,\n            )"
    if "progress_hook=self.progress_hook" not in text:
        if text.count(tester_anchor) != 2:
            raise RuntimeError("StressTester progress wiring: expected two constructors")
        text = text.replace(tester_anchor, tester_repl)

    stress_start = "        for family_id, rec, cand in ordered:\n            if time.perf_counter() - t0 >= float(cfg.max_runtime_seconds):"
    stress_enum = "        for candidate_index, (family_id, rec, cand) in enumerate(ordered, start=1):\n            if time.perf_counter() - t0 >= float(cfg.max_runtime_seconds):"
    stress_section_start = text.index("    def _run_stress_phase(")
    stress_section_end = text.index("    @classmethod\n    def robustness_entry_eligibility", stress_section_start)
    section = text[stress_section_start:stress_section_end]
    if stress_enum not in section:
        section = replace_once(section, stress_start, stress_enum, label="stress candidate enumeration")

    before_run = "            budget_before = int(counters.stress)\n            baseline_arts = None\n"
    started = '''            self._emit(\n                "STRESS_CANDIDATE_STARTED",\n                {\n                    "candidate_id": cand.candidate_id,\n                    "family_id": family_id,\n                    "candidate_index": candidate_index,\n                    "total_candidates": len(ordered),\n                    "scenarios": list(chosen),\n                    "stress_consumed": int(counters.stress),\n                    "max_stress_evaluations": int(budget.max_stress_evaluations),\n                },\n            )\n            budget_before = int(counters.stress)\n            baseline_arts = None\n'''
    if "STRESS_CANDIDATE_STARTED" not in section:
        section = replace_once(section, before_run, started, label="stress candidate started event")

    completion_anchor = '''            self._append_status(\n                candidate_id=cand.candidate_id,\n                family_id=family_id,\n                generation=gen,\n                prior_status=STRESS_TESTED,\n                new_status=decision,\n                reason=reason,\n                artifact_refs={\n                    "pass_rate": summary.pass_rate,\n                    "required_pass_rate": summary.required_pass_rate,\n                    "summary": summary.as_dict(),\n                },\n            )\n'''
    completion_repl = completion_anchor + '''            # Crash-safe Stress drain: persist every completed candidate, not only\n            # the entire phase. A restart resumes from the remaining gate queue.\n            if self._live_checkpoint is not None:\n                from discovery.search_resume import sync_pending_gate_queues\n\n                ckpt = self._live_checkpoint\n                ckpt.candidate_gates[cand.candidate_id] = decision\n                ckpt.stress_summaries = [\n                    s.as_dict() for s in self.candidate_stress_summaries\n                ]\n                ckpt.status_history = [e.as_dict() for e in self.candidate_status_history]\n                ckpt.status_seq = int(self._status_seq)\n                sync_pending_gate_queues(ckpt)\n                self._persist_live_checkpoint()\n            self._emit(\n                "STRESS_CANDIDATE_COMPLETED",\n                {\n                    "candidate_id": cand.candidate_id,\n                    "family_id": family_id,\n                    "candidate_index": candidate_index,\n                    "total_candidates": len(ordered),\n                    "decision": decision,\n                    "reason": reason,\n                    "pass_rate": summary.pass_rate,\n                    "scenarios_executed": summary.total_scenarios_executed,\n                    "stress_consumed": int(counters.stress),\n                    "max_stress_evaluations": int(budget.max_stress_evaluations),\n                },\n            )\n'''
    if "STRESS_CANDIDATE_COMPLETED" not in section:
        section = replace_once(section, completion_anchor, completion_repl, label="stress candidate checkpoint")
    text = text[:stress_section_start] + section + text[stress_section_end:]

    parser_marker = '''        max_oos_drawdown=float(raw.get("max_oos_drawdown", raw.get("max_drawdown_limit", 0.20))),\n'''
    parser_fields = parser_marker + '''        min_median_profit_factor=float(raw.get("min_median_profit_factor", 1.10)),\n        min_risk_normalized_annual_return=float(\n            raw.get("min_risk_normalized_annual_return", 0.12)\n        ),\n        target_risk_drawdown=float(raw.get("target_risk_drawdown", 0.05)),\n        max_risk_scale=float(raw.get("max_risk_scale", 4.0)),\n        drawdown_floor=float(raw.get("drawdown_floor", 0.0025)),\n        calmar_cap=float(raw.get("calmar_cap", 3.0)),\n'''
    if "min_risk_normalized_annual_return=float(" not in text[text.index("def family_campaign_from_config"):]:
        text = replace_once(text, parser_marker, parser_fields, label="economic config parser")

    write(rel, text)


def patch_stress() -> None:
    rel = "quant_framework/discovery/stress.py"
    text = read(rel)
    field_anchor = "    synthetic_stress_forbidden: bool = False\n"
    field_repl = field_anchor + "    progress_hook: Callable[[str, dict[str, Any]], None] | None = None\n"
    if "progress_hook: Callable[[str, dict[str, Any]], None] | None" not in text:
        text = replace_once(text, field_anchor, field_repl, label="StressTester progress field")

    method_anchor = '''    def run(\n        self,\n'''
    emit_method = '''    def _emit(self, name: str, payload: dict[str, Any]) -> None:\n        if self.progress_hook is not None:\n            self.progress_hook(name, payload)\n\n    def run(\n        self,\n'''
    if "    def _emit(self, name: str, payload: dict[str, Any])" not in text:
        text = replace_once(text, method_anchor, emit_method, label="StressTester emit method")

    loop_old = "        for scenario in chosen:\n            if self.counters.stress >= self.budget.max_stress_evaluations:\n"
    loop_new = "        for scenario_index, scenario in enumerate(chosen, start=1):\n            if self.counters.stress >= self.budget.max_stress_evaluations:\n"
    if loop_new not in text:
        text = replace_once(text, loop_old, loop_new, label="stress scenario enumeration")

    before_symbol = '''            # Scenarios that can never be a genuine rerun must not consume a\n'''
    started = '''            self._emit(\n                "STRESS_SCENARIO_STARTED",\n                {\n                    "candidate_id": candidate.candidate_id,\n                    "scenario": scenario,\n                    "scenario_index": scenario_index,\n                    "total_scenarios": len(chosen),\n                    "stress_consumed": int(self.counters.stress),\n                    "max_stress_evaluations": int(self.budget.max_stress_evaluations),\n                },\n            )\n\n''' + before_symbol
    if "STRESS_SCENARIO_STARTED" not in text:
        text = replace_once(text, before_symbol, started, label="stress scenario started")

    counter_anchor = "            self.counters.stress += 1\n\n        stress_counter_delta"
    counter_repl = '''            self.counters.stress += 1\n            completed = results[-1]\n            self._emit(\n                "STRESS_SCENARIO_COMPLETED",\n                {\n                    "candidate_id": candidate.candidate_id,\n                    "scenario": scenario,\n                    "scenario_index": scenario_index,\n                    "total_scenarios": len(chosen),\n                    "status": completed.status,\n                    "passed": completed.passed,\n                    "failure_reason": completed.failure_reason,\n                    "stress_consumed": int(self.counters.stress),\n                    "max_stress_evaluations": int(self.budget.max_stress_evaluations),\n                },\n            )\n\n        stress_counter_delta'''
    if "STRESS_SCENARIO_COMPLETED" not in text:
        text = replace_once(text, counter_anchor, counter_repl, label="stress scenario completed")

    write(rel, text)


def patch_jobs() -> None:
    rel = "quant_framework/control_plane/jobs.py"
    text = read(rel)
    event_anchor = '''        "FINALIST_SELECTED": EventType.FINALIST_SELECTED,\n'''
    event_fields = event_anchor + '''        # Multi-family resume / Stress visibility. Generic PROGRESS events are\n        # intentional: payload carries candidate/scenario detail without adding\n        # execution semantics to the EventType enum.\n        "SEARCH_RESUME_STARTED": EventType.PROGRESS,\n        "MULTI_FAMILY_STRESS_STARTED": EventType.PROGRESS,\n        "STRESS_CANDIDATE_STARTED": EventType.PROGRESS,\n        "STRESS_SCENARIO_STARTED": EventType.PROGRESS,\n        "STRESS_SCENARIO_COMPLETED": EventType.PROGRESS,\n        "STRESS_CANDIDATE_COMPLETED": EventType.PROGRESS,\n        "SEARCH_CHECKPOINT_SAVED": EventType.PROGRESS,\n        "MULTI_FAMILY_STRESS_COMPLETED": EventType.PROGRESS,\n        "MULTI_FAMILY_ROBUSTNESS_STARTED": EventType.PROGRESS,\n        "MULTI_FAMILY_ROBUSTNESS_COMPLETED": EventType.PROGRESS,\n'''
    if '"STRESS_CANDIDATE_STARTED": EventType.PROGRESS' not in text:
        text = replace_once(text, event_anchor, event_fields, label="control-plane stress event map")

    progress_anchor = '''        progress = min(95.0, 10.0 + 80.0 * (counter_proxy.generated / gen_cap))\n'''
    progress_repl = progress_anchor + '''        if name in {\n            "STRESS_CANDIDATE_STARTED",\n            "STRESS_CANDIDATE_COMPLETED",\n        }:\n            idx = int(payload.get("candidate_index") or 0)\n            total = max(1, int(payload.get("total_candidates") or 1))\n            progress = min(95.0, 80.0 + 15.0 * (idx / total))\n        elif name in {"STRESS_SCENARIO_STARTED", "STRESS_SCENARIO_COMPLETED"}:\n            cidx = int(payload.get("candidate_index") or 0)\n            ctotal = max(1, int(payload.get("total_candidates") or 1))\n            sidx = int(payload.get("scenario_index") or 0)\n            stotal = max(1, int(payload.get("total_scenarios") or 1))\n            fractional = ((max(cidx, 1) - 1) + sidx / stotal) / ctotal\n            progress = min(95.0, 80.0 + 15.0 * fractional)\n'''
    if "fractional = ((max(cidx, 1) - 1)" not in text:
        text = replace_once(text, progress_anchor, progress_repl, label="stress progress percent")

    write(rel, text)


def main() -> None:
    patch_fitness()
    patch_regression_test()
    patch_multi_family()
    patch_stress()
    patch_jobs()
    print("Economic objective and Stress observability hardening applied.")


if __name__ == "__main__":
    main()
