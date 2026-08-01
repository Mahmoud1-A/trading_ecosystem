"""Dashboard Continue Search draft persistence + terminal refresh stability."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from control_plane.app import create_app
from control_plane.run_manager import RunManager
from discovery.search_resume import apply_budget_extension
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    mgr = RunManager(tmp_path / "cp_cs", max_workers=1)
    return TestClient(create_app(mgr, root=tmp_path / "cp_cs"))


def _js(client: TestClient) -> str:
    return client.get("/assets/dashboard.js").text


class TestContinueSearchFormDrafts:
    def test_default_runtime_is_10800_not_3600(self, client: TestClient) -> None:
        js = _js(client)
        assert "value=\"3600\"" not in js or "CS_DRAFT_DEFAULTS" in js
        assert "runtime: 10800" in js
        assert "CS_DRAFT_DEFAULTS" in js
        assert "getContinueSearchDraft" in js
        assert "saveContinueSearchDraft" in js
        # Rendered value comes from draft, not a hardcoded 3600.
        assert "value=\"${csDraft.runtime}\"" in js
        assert "value=\"${csDraft.generated}\"" in js
        assert "value=\"${csDraft.full_wfo}\"" in js

    def test_no_artificial_html_max_on_runtime(self, client: TestClient) -> None:
        js = _js(client)
        # Continue Search runtime input must not carry max=...
        m = re.search(
            r'id="cs_runtime"[^>]*>',
            js,
        )
        assert m is not None
        tag = m.group(0)
        assert "max=" not in tag

    def test_drafts_keyed_by_run_and_mirrored_in_session_storage(
        self, client: TestClient
    ) -> None:
        js = _js(client)
        assert "continueSearchDrafts" in js
        assert "cp_continue_search_drafts" in js
        assert "sessionStorage.setItem(CS_DRAFT_STORAGE_KEY" in js
        assert "wireContinueSearchDraftInputs" in js
        assert 'el.addEventListener("input", saveFromInputs)' in js
        assert 'el.addEventListener("change", saveFromInputs)' in js

    def test_continue_search_submits_draft_runtime_not_hardcoded(
        self, client: TestClient
    ) -> None:
        js = _js(client)
        # Submit path must use saved draft values (e.g. 10800), not literals.
        assert "additional_runtime_seconds: Number(draft.runtime)" in js
        assert "additional_generated_budget: Number(draft.generated)" in js
        assert "additional_full_wfo_budget: Number(draft.full_wfo)" in js
        assert "Number($(\"#cs_runtime\")?.value || 0)" not in js

    def test_terminal_sse_end_does_not_refresh_unless_active(
        self, client: TestClient
    ) -> None:
        js = _js(client)
        # scheduleAlphaRefresh itself gates on alphaRunActive.
        assert "if (!state.alphaRunActive) return;" in js
        # SSE end handler must not refresh terminal pages.
        assert "if (state.alphaRunActive) scheduleAlphaRefresh(50);" in js
        assert re.search(
            r'es\.addEventListener\("end",\s*\(\)\s*=>\s*\{\s*es\.close\(\);\s*scheduleAlphaRefresh\(50\);\s*\}\)',
            js,
        ) is None
        # Terminal pages do not auto-connect the live stream.
        assert "if (state.alphaRunActive) {\n      connect();\n      scheduleAlphaRefresh(1400);" in js.replace(
            "\r\n", "\n"
        ) or (
            "if (state.alphaRunActive) {" in js
            and "connect();" in js
            and "scheduleAlphaRefresh(1400)" in js
        )

    def test_active_run_refresh_still_functions(self, client: TestClient) -> None:
        js = _js(client)
        assert "function scheduleAlphaRefresh" in js
        assert "if (!state.alphaRunActive) return;" in js
        assert "route({background:true, preserveScroll:true})" in js
        assert "state.alphaRunActive = ACTIVE.has(run.state);" in js
        # Active path still schedules refresh after events load.
        assert "scheduleAlphaRefresh(1400)" in js
        assert "scheduleAlphaRefresh(650)" in js

    def test_background_render_uses_saved_draft(self, client: TestClient) -> None:
        """Editing 10800 must survive re-render because values come from drafts."""
        js = _js(client)
        # Draft save on input + render from draft is the mechanism that keeps 10800
        # visible across route({background:true}) innerHTML replacement.
        assert "saveContinueSearchDraft(runId" in js
        assert "const csDraft = getContinueSearchDraft(id);" in js
        assert "value=\"${csDraft.runtime}\"" in js
        # Default is already 10800, so typing 10800 (or leaving default) persists.
        assert "runtime: 10800" in js


class TestSessionRuntimeSemantics:
    def test_additional_runtime_replaces_session_cap_not_adds(self) -> None:
        class Cfg:
            max_runtime_seconds = 7200.0
            total_candidate_budget = 60
            max_full_wfo = 18
            max_evaluated_candidates = 60

        cfg = Cfg()
        apply_budget_extension(
            cfg,
            additional_runtime_seconds=10800.0,
            additional_generated_budget=10,
            additional_full_wfo_budget=5,
        )
        assert cfg.max_runtime_seconds == 10800.0
        assert cfg.total_candidate_budget == 70
        assert cfg.max_full_wfo == 23
        assert cfg.max_evaluated_candidates == 70

    def test_zero_additional_runtime_keeps_prior_cap(self) -> None:
        class Cfg:
            max_runtime_seconds = 7200.0
            total_candidate_budget = 60
            max_full_wfo = 18
            max_evaluated_candidates = None

        cfg = Cfg()
        apply_budget_extension(cfg, additional_runtime_seconds=0.0)
        assert cfg.max_runtime_seconds == 7200.0


class TestContinueSearchBootstrapStillWired:
    def test_continue_search_endpoint_and_bootstrap_helpers_present(
        self, client: TestClient
    ) -> None:
        js = _js(client)
        assert "/continue_search" in js
        assert "Continue Search" in js
        # Server-side bootstrap path remains available.
        from control_plane import search_bootstrap

        assert hasattr(search_bootstrap, "prepare_search_program_session")
        assert hasattr(search_bootstrap, "bootstrap_checkpoint_from_run")
