"""The aux quality gate: a third case between success and failure.

``assess_aux`` decides whether an aux answer actually carries future tasks.
AuxStage records the verdict rather than raising on it -- see its docstring for
why this one is filtered at analysis time while mechanical failures are not.

The two UNUSABLE strings below are verbatim from real qwen2.5-coder:7b runs, not
invented: both are the model following the "quote a detail" half of the prompt
and then stopping before the list.
"""

from __future__ import annotations

import pytest

from conftest import AUX_PROMPT, StubBackend, make_adapter, user_only
from foresight.adapters.base import CallerAdapter
from foresight.context import InboundRequest, RequestContext
from foresight.errors import AuxFailure, GuardViolation
from foresight.stages import MIN_FUTURE_TASK_ITEMS, AuxStage, assess_aux

GOOD = """**Quoted Detail:** `cli/sync.py`

**Plausible FUTURE Tasks:**
1. **Implement Detailed Output on Dry-Run Mode**: more detail about what changes.
2. **Logging Dry-Run Information**: record planned changes for debugging.
3. **Concurrency in Dry-Run Mode**: analyse impact concurrently.
4. **Integration with Version Control**: simulate interaction with Git.
5. **Automated Backup in Dry-Run**: allow a rollback.
"""

# Verbatim from the observed 7% failure: quoted the detail, then stopped.
OBSERVED_FAILURE = "Distinctive detail from the task:  \n`clients/api.py`"

# Verbatim from an earlier failure: hallucinated a stub, no quote, no list.
OBSERVED_STUB_FAILURE = '''```python
def sync_files(source, destination):
    """
    Syncs files from source to destination.
    """
    # Function implementation here
```'''


def test_a_real_five_item_answer_is_usable():
    verdict = assess_aux(GOOD)
    assert verdict["items"] == 5
    assert verdict["usable"] is True


def test_the_observed_failure_is_unusable():
    verdict = assess_aux(OBSERVED_FAILURE)
    assert verdict["items"] == 0
    assert verdict["usable"] is False


def test_the_observed_stub_failure_is_unusable():
    assert assess_aux(OBSERVED_STUB_FAILURE)["usable"] is False


def test_bullet_lists_count_too():
    """The prompt asks for a list, not for numbering."""
    verdict = assess_aux("- add validation\n- support more formats\n- retry on 429\n")
    assert verdict["items"] == 3
    assert verdict["usable"] is True


def test_markdown_bold_is_not_mistaken_for_a_bullet():
    """`**Detail:**` starts with a star but is not a list item."""
    assert assess_aux("**Detail:** `api.py`\n**Another:** thing\n")["items"] == 0


def test_repeated_numbering_is_not_credited_twice():
    """Distinct numbers, so a model that restarts its list gains nothing."""
    assert assess_aux("1. one\n1. one again\n1. and again\n")["items"] == 1


def test_threshold_boundary():
    two = "1. a\n2. b\n"
    three = "1. a\n2. b\n3. c\n"
    assert assess_aux(two)["usable"] is False
    assert assess_aux(three)["usable"] is True
    assert MIN_FUTURE_TASK_ITEMS == 3


def _ctx(body: dict) -> RequestContext:
    req = InboundRequest(body=body)
    return RequestContext(req=req, body_out=dict(body))


@pytest.mark.asyncio
async def test_weak_aux_is_noted_and_still_used(aux_spec):
    """Flagged, not raised: the enhancement still happens, but it is marked.

    Raising here would abort the caller's whole session, which at experiment
    scale biases which tasks finish. The note is what lets the instance be
    excluded afterwards instead.
    """
    backend = StubBackend(reply_text=OBSERVED_FAILURE)
    stage = AuxStage(make_adapter(backend, aux_spec), aux_spec)
    ctx = _ctx(user_only("Add retry to fetch_user()"))

    await stage.run(ctx)

    assert ctx.aux is not None
    assert ctx.aux.text == OBSERVED_FAILURE
    assert "aux_s" in ctx.timings
    assert any(n.startswith("weak_aux:") for n in ctx.notes)


@pytest.mark.asyncio
async def test_usable_aux_adds_no_note(aux_spec):
    backend = StubBackend(reply_text=GOOD)
    stage = AuxStage(make_adapter(backend, aux_spec), aux_spec)
    ctx = _ctx(user_only("Add a --dry-run flag"))

    await stage.run(ctx)

    assert ctx.notes == []


@pytest.mark.asyncio
async def test_empty_aux_still_raises(aux_spec):
    """The gate is additive: a mechanical failure is still a hard failure."""
    backend = StubBackend(reply_text="   \n  ")
    stage = AuxStage(make_adapter(backend, aux_spec), aux_spec)
    ctx = _ctx(user_only("x"))

    with pytest.raises(AuxFailure, match="empty"):
        await stage.run(ctx)


class _ContaminatingAdapter(CallerAdapter):
    """An adapter whose aux run modifies the workspace."""

    name = "contaminating"

    async def run_aux(self, req, aux):
        raise GuardViolation(["/repo/aux_wrote_this.txt"], "/repo")


@pytest.mark.asyncio
async def test_guard_violation_is_not_collapsed_into_aux_failure(aux_spec):
    """Regression: AuxStage catches broad Exception to funnel adapter failures.

    GuardViolation was being swallowed by that catch-all and re-raised as
    AuxFailure, so contamination surfaced as 502 aux_failure instead of 500
    workspace_contaminated. Only the live run caught it -- the LocalAdapter
    tests call run_aux directly and never traverse AuxStage.

    The distinction matters at analysis time: a failed aux call loses an
    instance, a contaminated workspace corrupts a measurement.
    """
    adapter = _ContaminatingAdapter(
        aux_backend=StubBackend(), aux_spec=aux_spec, aux_prompt=AUX_PROMPT
    )
    stage = AuxStage(adapter, aux_spec)
    ctx = _ctx(user_only("x"))

    with pytest.raises(GuardViolation) as caught:
        await stage.run(ctx)

    assert caught.value.error_type == "workspace_contaminated"
    assert caught.value.status_code == 500
    assert caught.value.changed == ["/repo/aux_wrote_this.txt"]
