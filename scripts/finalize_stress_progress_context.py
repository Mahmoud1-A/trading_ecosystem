from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.write_text(text, encoding="utf-8")
    ast.parse(text, filename=str(path))


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def patch_stress_context() -> None:
    rel = "quant_framework/discovery/stress.py"
    text = _read(rel)
    text = _replace_once(
        text,
        "    progress_hook: Callable[[str, dict[str, Any]], None] | None = None\n",
        "    progress_hook: Callable[[str, dict[str, Any]], None] | None = None\n"
        "    progress_context: dict[str, Any] = field(default_factory=dict)\n",
        "StressTester progress context field",
    )
    text = _replace_once(
        text,
        "    def _emit(self, name: str, payload: dict[str, Any]) -> None:\n"
        "        if self.progress_hook is not None:\n"
        "            self.progress_hook(name, payload)\n",
        "    def _emit(self, name: str, payload: dict[str, Any]) -> None:\n"
        "        if self.progress_hook is not None:\n"
        "            # Candidate-level context is supplied by MultiFamilyCampaign so\n"
        "            # scenario progress is monotonic across the full Stress queue.\n"
        "            self.progress_hook(name, {**self.progress_context, **payload})\n",
        "StressTester progress context merge",
    )
    _write(rel, text)


def patch_campaign_context() -> None:
    rel = "quant_framework/discovery/multi_family_campaign.py"
    text = _read(rel)
    old = (
        "            try:\n"
        "                results = tester.run(\n"
        "                    cand,\n"
        "                    base_fitness=float(rec.fitness.fitness) if rec.fitness else 0.0,\n"
        "                    scenarios=chosen,\n"
        "                    baseline_artifacts=baseline_arts,\n"
        "                )\n"
    )
    new = (
        "            tester.progress_context = {\n"
        "                \"candidate_index\": candidate_index,\n"
        "                \"total_candidates\": len(ordered),\n"
        "                \"family_id\": family_id,\n"
        "            }\n"
        "            try:\n"
        "                results = tester.run(\n"
        "                    cand,\n"
        "                    base_fitness=float(rec.fitness.fitness) if rec.fitness else 0.0,\n"
        "                    scenarios=chosen,\n"
        "                    baseline_artifacts=baseline_arts,\n"
        "                )\n"
    )
    text = _replace_once(text, old, new, "Stress candidate progress context")
    _write(rel, text)


def main() -> None:
    patch_stress_context()
    patch_campaign_context()
    print("Stress scenario progress context finalized.")


if __name__ == "__main__":
    main()
