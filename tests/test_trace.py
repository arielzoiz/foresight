"""Trace records: the experiment's primary evidence.

Two levels here. The server-level tests drive real requests through create_app
with a real TraceWriter and read the file back, because record *shape* is what
analysis code will depend on. The unit tests cover the two pieces with logic of
their own: UsageSniffer's parsing, and the startup writability check.

test_e2e.py covers the same ground through the real shipped config against
fake_upstream, including the caller-opted-in streaming usage path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from conftest import user_only, with_history
from foresight.config import ForesightConfig, Runtime
from foresight.errors import ConfigError, TraceFailure
from foresight.server import create_app
from foresight.trace import TraceWriter, UsageSniffer
from test_config import BASE
from test_server import build_test_runtime
from tools.fake_upstream import _sse_chunks

TARGET_KEYS = {
    "request_id",
    "received_at",
    "role",
    "served_model",
    "adapter",
    "builder",
    "session_key",
    "is_session_start",
    "is_agent_turn",
    "phase",
    "enhanced",
    "prompt_in",
    "prompt_out",
    "prompt_in_chars",
    "prompt_out_chars",
    "aux",
    "guard",
    "message_count",
    "tool_count",
    "stream",
    "upstream_status",
    "usage",
    "timings",
    "notes",
    "error",
}


def make_tracer(tmp_path: Path) -> TraceWriter:
    """A writer under tmp_path, with the parent directory not yet existing."""
    writer = TraceWriter(tmp_path / "traces" / "test.jsonl")
    writer.preflight()
    return writer


def records(writer: TraceWriter) -> list[dict]:
    if not writer.path.exists():
        return []
    return [json.loads(line) for line in writer.path.read_text().splitlines() if line.strip()]


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# -- record shape ---------------------------------------------------------


@pytest.mark.asyncio
async def test_target_record_has_exactly_the_expected_keys(tmp_path):
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("Fix parse_date()"))
        assert resp.status_code == 200

    rows = records(tracer)
    assert len(rows) == 1
    row = rows[0]

    # Pinned deliberately: analysis code reads these keys, so a rename should
    # break a test rather than a notebook.
    assert set(row) == TARGET_KEYS
    assert row["role"] == "target"
    assert row["adapter"] == "generic"
    assert row["builder"] == "template"
    assert row["error"] is None
    assert row["guard"] is None
    assert row["enhanced"] is True
    assert row["prompt_in"] == "Fix parse_date()"
    assert row["prompt_in_chars"] == len("Fix parse_date()")
    assert row["prompt_out_chars"] == len(row["prompt_out"])
    assert "future: X, Y, Z" in row["prompt_out"]
    assert row["aux"]["source"] == "chat_call"
    assert row["aux"]["provenance"]["served_name"] == "aux-model"
    # The quality verdict travels with every row; see stages.assess_aux.
    assert row["aux"]["items"] == 0  # the stub reply is not a task list
    assert row["aux"]["usable"] is False


@pytest.mark.asyncio
async def test_disabled_tracer_writes_nothing(tmp_path):
    path = tmp_path / "never_created.jsonl"
    runtime, _, _ = build_test_runtime(tracer=None)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("x"))
        assert resp.status_code == 200

    assert not path.exists()


@pytest.mark.asyncio
async def test_config_without_a_trace_path_has_no_tracer():
    """Opting out is a config value, not a NoopWriter -- same as guard: null."""
    runtime = Runtime(ForesightConfig(**BASE))
    try:
        assert runtime.tracer is None
    finally:
        await runtime.aclose()


# -- the invariant the whole feature exists to prove ----------------------


@pytest.mark.asyncio
async def test_aux_runs_once_but_every_request_is_enhanced(tmp_path):
    """One aux call per session; the enhancement present on every request.

    The machine-checkable form of the re-application invariant: turn 1 carries
    an aux_s timing because aux actually ran, turn 2 does not because the result
    came from the session store -- and yet both rows are enhanced, with
    identical prompt_out.
    """
    tracer = make_tracer(tmp_path)
    runtime, _, aux_backend = build_test_runtime(tracer=tracer)

    task = "Refactor the retry logic"
    async with client_for(create_app(runtime)) as client:
        first = await client.post("/v1/chat/completions", json=user_only(task))
        second = await client.post(
            "/v1/chat/completions",
            json=with_history(
                task,
                {"role": "assistant", "content": "reading files"},
                {"role": "tool", "content": "def request(): ..."},
            ),
        )
    assert first.status_code == second.status_code == 200

    start, cont = records(tracer)
    assert len(aux_backend.calls) == 1

    assert start["is_session_start"] is True
    assert "aux_s" in start["timings"]

    assert cont["is_session_start"] is False
    assert "aux_s" not in cont["timings"]

    assert start["session_key"] == cont["session_key"]
    assert start["enhanced"] is cont["enhanced"] is True
    assert start["prompt_out"] == cont["prompt_out"]
    assert cont["aux"]["text"] == start["aux"]["text"]


@pytest.mark.asyncio
async def test_guard_verdict_lands_on_every_row_of_the_session(tmp_path):
    """The workspace verdict travels like the aux verdict does.

    Milestone 2's guard runs once, inside run_aux, on the session-start request.
    Every later request of that session reuses the stored AuxResult and never
    re-enters the adapter -- so a verdict written only where the guard ran would
    be absent from most rows of the very session it would invalidate. Filtering
    an experiment on ``guard.workspace_intact`` has to work per row.
    """
    from conftest import StubBackend
    from foresight.adapters.base import AdapterOptions
    from foresight.adapters.local import LocalAdapter
    from foresight.llm import ModelSpec

    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "parser.py").write_text("def parse_date(s):\n    return s\n")

    aux_spec = ModelSpec(served_name="aux-model", model="fake-aux", base_url="http://x/v1")
    adapter = LocalAdapter(
        aux_backend=StubBackend(),
        aux_spec=aux_spec,
        aux_prompt="Repo {{ workspace }}, task {{ prompt }}, write {{ answer_file }}",
        options=AdapterOptions(
            name="local",
            workspace=str(workspace),
            agent_cmd=[sys.executable, str(REPO / "tools" / "fake_agent.py"),
                       "--answer-file", "{answer_file}", "{prompt}"],
            timeout_s=60.0,
        ),
        guard_name="manifest",
    )

    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer, adapter=adapter)

    task = "Make parse_date() reject malformed input"
    async with client_for(create_app(runtime)) as client:
        first = await client.post("/v1/chat/completions", json=user_only(task))
        second = await client.post(
            "/v1/chat/completions",
            json=with_history(
                task,
                {"role": "assistant", "content": "reading parser.py"},
                {"role": "tool", "content": "def parse_date(s): ..."},
            ),
        )
    assert first.status_code == second.status_code == 200

    start, cont = records(tracer)
    assert start["aux"]["source"] == "agent"
    assert start["is_session_start"] is True and cont["is_session_start"] is False

    # The point of the test: identical on both, though only the first ran a guard.
    for row in (start, cont):
        assert row["guard"] == {"verdict": "intact", "workspace_intact": True}


@pytest.mark.asyncio
async def test_passthrough_arm_records_an_unenhanced_prompt(tmp_path):
    """The control arm is visible in the trace, and still pays for aux."""
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer, builder_name="passthrough")
    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("Fix parse_date()"))
        assert resp.status_code == 200

    row = records(tracer)[0]
    assert row["builder"] == "passthrough"
    assert row["enhanced"] is False
    assert row["prompt_in"] == row["prompt_out"]
    # Aux still ran: a control run must not be cheaper for reasons unrelated to
    # the prompt.
    assert row["aux"] is not None
    assert row["usage"]["aux"] is not None


# -- upstream facts -------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_records_upstream_status_and_usage(tmp_path):
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer)
    async with client_for(create_app(runtime)) as client:
        await client.post("/v1/chat/completions", json=user_only("x"))

    row = records(tracer)[0]
    assert row["stream"] is False
    assert row["upstream_status"] == 200
    assert row["usage"]["upstream"] == {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.mark.asyncio
async def test_upstream_error_status_is_recorded_faithfully(tmp_path):
    """A failing target is not a foresight error, and the row proves which it was."""
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer, target_status=503)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("x"))
        assert resp.status_code == 503

    row = records(tracer)[0]
    assert row["upstream_status"] == 503
    assert row["error"] is None  # foresight did its job; the upstream did not
    assert row["enhanced"] is True


@pytest.mark.asyncio
async def test_streaming_relays_bytes_and_traces_after_the_stream_drains(tmp_path):
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json={**user_only("x"), "stream": True})
        assert resp.status_code == 200
        assert resp.content.endswith(b"data: [DONE]\n\n")

    row = records(tracer)[0]
    assert row["stream"] is True
    # Neither is knowable from a stream: Backend.stream yields bytes and never
    # exposes the status, and this caller did not ask for usage.
    assert row["upstream_status"] is None
    assert row["usage"]["upstream"] is None
    assert "stream_incomplete" not in row["notes"]


# -- failure paths --------------------------------------------------------


@pytest.mark.asyncio
async def test_aux_failure_writes_a_thin_error_record(tmp_path):
    tracer = make_tracer(tmp_path)
    runtime, target_backend, _ = build_test_runtime(tracer=tracer, aux_status=500)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("Fix parse_date()"))
        assert resp.status_code == 502

    row = records(tracer)[0]
    assert row["error"]["type"] == "aux_failure"
    assert "500" in row["error"]["message"]
    assert row["enhanced"] is False
    assert row["prompt_in"] == "Fix parse_date()"
    # Recoverable even though no RequestContext existed: the server asks the
    # adapter directly, so failed instances stay joinable to their session.
    assert row["session_key"]
    # The target was never called, so there is nothing upstream to report.
    assert "upstream_status" not in row
    assert target_backend.calls == []


@pytest.mark.asyncio
async def test_aux_bypass_is_recorded_without_pipeline_fields(tmp_path):
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "aux-model", "messages": [{"role": "user", "content": "raw"}]},
        )
        assert resp.status_code == 200

    row = records(tracer)[0]
    assert row["role"] == "aux"
    assert row["served_model"] == "aux-model"
    assert row["error"] is None
    assert row["usage"]["aux"] is None
    assert row["usage"]["upstream"] == {"prompt_tokens": 1, "completion_tokens": 1}
    for absent in ("session_key", "enhanced", "prompt_in", "adapter", "builder"):
        assert absent not in row


@pytest.mark.asyncio
async def test_unknown_model_is_not_traced(tmp_path):
    """No model was called, so there is nothing for the trace to account for."""
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer)
    async with client_for(create_app(runtime)) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
        )
        assert resp.status_code == 400

    assert records(tracer) == []


@pytest.mark.asyncio
async def test_a_failed_trace_write_fails_the_request(tmp_path, monkeypatch):
    """Decision: losing traces silently is worse than failing loudly."""
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer)

    def boom(record):
        raise TraceFailure("disk full")

    monkeypatch.setattr(tracer, "write", boom)

    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("x"))

    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "trace_failure"


@pytest.mark.asyncio
async def test_a_failed_trace_write_never_masks_an_aux_failure(tmp_path, monkeypatch):
    """The one exception: the caller needs the diagnosis, not the symptom."""
    tracer = make_tracer(tmp_path)
    runtime, _, _ = build_test_runtime(tracer=tracer, aux_status=500)

    def boom(record):
        raise TraceFailure("disk full")

    monkeypatch.setattr(tracer, "write", boom)

    async with client_for(create_app(runtime)) as client:
        resp = await client.post("/v1/chat/completions", json=user_only("x"))

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "aux_failure"


# -- startup ---------------------------------------------------------------


def test_missing_parent_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "trace.jsonl"
    writer = TraceWriter(path)
    writer.preflight()
    writer.write({"hello": "world"})
    assert json.loads(path.read_text().strip()) == {"hello": "world"}


def test_unwritable_trace_path_fails_at_startup(tmp_path):
    """A fatal-at-request-time policy needs a fail-at-startup escape hatch."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")

    config = ForesightConfig(**{**BASE, "trace": {"path": str(blocker / "trace.jsonl")}})
    with pytest.raises(ConfigError, match="not writable"):
        Runtime(config)


@pytest.mark.asyncio
async def test_configured_trace_path_builds_a_writer(tmp_path):
    path = tmp_path / "traces" / "run.jsonl"
    config = ForesightConfig(**{**BASE, "trace": {"path": str(path)}})
    runtime = Runtime(config)
    try:
        assert runtime.tracer is not None
        assert runtime.tracer.path == path
        assert path.exists()  # preflight created the directory and the file
    finally:
        await runtime.aclose()


# -- UsageSniffer ----------------------------------------------------------


def test_sniffer_returns_none_when_the_stream_carries_no_usage():
    sniffer = UsageSniffer()
    sniffer.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
    sniffer.feed(b"data: [DONE]\n\n")
    assert sniffer.usage() is None


def test_sniffer_ignores_a_line_beheaded_by_tail_trimming():
    """Trimming can cut the oldest line in half; unparseable lines are skipped."""
    sniffer = UsageSniffer(tail_bytes=80)
    sniffer.feed(b'data: {"usage":{"prompt_tokens":999},"padding":"' + b"x" * 200 + b'"}\n\n')
    sniffer.feed(b"data: [DONE]\n\n")
    assert sniffer.usage() is None


@pytest.mark.asyncio
async def test_sniffer_recovers_usage_from_a_real_fake_upstream_stream():
    """Against the bytes fake_upstream actually emits for include_usage."""
    expected = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    sniffer = UsageSniffer()
    async for chunk in _sse_chunks("id", 0, "fake-target", "some reply text", usage=expected):
        sniffer.feed(chunk)
    assert sniffer.usage() == expected
