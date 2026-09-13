"""Only verified procedures become historical advice; drafts never activate."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from jarvis.core.protocols import ToolResult
from jarvis.skills.procedure_learning import procedure_note, persist_procedure


def test_procedure_note_requires_verified_success_and_enough_real_steps():
    steps = ['S1: open_app ok (Fixture)', 'S2: type ok (Confirmed)', 'S0: click FAIL (missing)']
    note = procedure_note(goal='Prepare a fixture draft', steps=steps,
        proof='[cu] done (verified: Draft shows the requested text)', exit_code=0,
        observed_at=datetime(2026, 9, 13, tzinfo=UTC))
    assert note and '2026-09-13' in note and 'Draft shows the requested text' in note
    assert 'FAIL' not in note
    assert 'recheck' in note.lower() and 'permission' in note.lower()
    for code, proof in [(1, '[cu] done (verified: wrong)'), (0, '[cu] done'), (0, '')]:
        assert procedure_note(goal='Task', steps=steps, proof=proof, exit_code=code) is None
    assert procedure_note(goal='Task', steps=steps[:1], proof='done (verified: text)', exit_code=0) is None


async def test_learning_uses_existing_wiki_tool_executor_without_activating_skills():
    class Executor:
        calls = []
        async def execute(self, tool, args, **kwargs):
            self.calls.append((tool, args, kwargs))
            return ToolResult(success=True, output='Saved')
    tool, executor = object(), Executor()
    brain = SimpleNamespace(_tools={'wiki-ingest':tool}, _tool_executor=executor)
    assert await persist_procedure(brain, goal='Fixture', steps=['1: open ok (Fixture)','2: type ok (Text)'],
        proof='done (verified: Text)', exit_code=0)
    assert executor.calls[0][0] is tool
    assert executor.calls[0][1]['source'].startswith('procedure:verified:')
    assert not await persist_procedure(brain, goal='Failed', steps=['1: open ok (App)','2: click ok (Button)'],
        proof='failed', exit_code=1)
    assert len(executor.calls) == 1
