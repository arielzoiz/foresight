"""Summarise an aux agent's opencode session export.

Answers the one question that decides whether an aux run is reportable
evidence or not: *did the aux agent actually read the code?* -- and, when it
did not, which of the two failure modes happened.

Why this is not the trace
-------------------------
``trace.jsonl`` records one ``role: "aux"`` row per model call, so counting
rows already tells you whether a tool round-trip occurred: two rows is
opencode's title-generation call plus one agent turn and nothing else, three
or more means a tool ran and its result came back. That is the verdict, but it
is not the diagnosis -- two very different failures produce exactly two rows:

* the model never attempted a tool (a prompt or capability problem), or
* it attempted one and nothing downstream could parse what it emitted (a wire
  format problem -- the wrong ``--tool-call-parser``, or a model that ignores
  its own chat template).

The trace cannot separate those, because foresight relays aux traffic as raw
bytes and never parses it. opencode can: it records, per turn, whether a
``tool`` part was produced at all. So this reads opencode's own transcript,
exported by ``deploy/tau-slurm/foresight.sbatch`` into
``$RUN/aux-agent/ses_*.json``.

Usage:
    python tools/aux_transcript.py $RUN/aux-agent/ses_*.json
    python tools/aux_transcript.py --full $RUN/aux-agent/ses_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

#: Enough to recognise a fenced JSON blob or an XML tool call masquerading as
#: prose, which is exactly the failure this tool exists to make visible.
SNIPPET = 400


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


def _tool_part(part: dict) -> tuple[str, str, str]:
    """(name, status, detail) for a tool part, across opencode's shapes."""
    state = part.get("state") or {}
    name = part.get("tool") or state.get("title") or "?"
    status = state.get("status") or "?"
    detail = state.get("error") or state.get("output") or ""
    if not detail and state.get("input") is not None:
        detail = json.dumps(state["input"])
    return str(name), str(status), str(detail)


def summarise(path: str, limit: int) -> dict:
    with open(path, encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)

    info = data.get("info") or {}
    messages = data.get("messages") or []

    print(f"\n=== {path}")
    print(f"    session   {info.get('id')}  ({info.get('slug')})")
    print(f"    model     {(info.get('model') or {}).get('id')}  agent={info.get('agent')}")
    print(f"    directory {info.get('directory')}")
    tokens = info.get("tokens") or {}
    print(f"    tokens    in={tokens.get('input')} out={tokens.get('output')}")

    steps = 0
    tool_calls: list[str] = []
    final_text = ""

    for message in messages:
        minfo = message.get("info") or {}
        role = minfo.get("role")
        print(f"\n  -- {role}" + (f"  finish={minfo['finish']}" if minfo.get("finish") else ""))

        for part in message.get("parts") or []:
            kind = part.get("type")
            if kind == "step-start":
                steps += 1
                print(f"     [step {steps}]")
            elif kind == "step-finish":
                print(f"     [step end: {part.get('reason')}]")
            elif kind == "text":
                text = part.get("text") or ""
                if role == "assistant":
                    final_text = text
                print(f"     text: {_clip(text, limit)}")
            elif kind == "reasoning":
                print(f"     reasoning: {_clip(part.get('text') or '', limit)}")
            elif kind == "tool":
                name, status, detail = _tool_part(part)
                tool_calls.append(name)
                print(f"     TOOL {name} [{status}]: {_clip(detail, limit)}")
            elif kind in ("file", "patch"):
                print(f"     {kind}: {_clip(json.dumps(part), limit)}")

    # A model call is one step, so steps is the number of aux rows this session
    # should have contributed to trace.jsonl -- cross-check them.
    print(f"\n  SUMMARY  model-calls(steps)={steps}  tool-calls={len(tool_calls)}")
    if tool_calls:
        print(f"           tools used: {', '.join(tool_calls)}")
        print("           GROUNDED -- the agent ran tools against the workspace.")
    else:
        print("           NOT GROUNDED -- no tool ever ran. Read the final text below:")
        print("             prose answer      -> the model did not attempt a tool")
        print("             JSON/XML blob     -> it tried and nothing parsed it;")
        print("                                  suspect --tool-call-parser")
        print(f"           final text: {_clip(final_text, limit)}")

    return {"steps": steps, "tools": len(tool_calls)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="ses_*.json exports")
    parser.add_argument(
        "--full",
        action="store_true",
        help="do not truncate text and tool output",
    )
    args = parser.parse_args()

    # Expanded here as well as by the shell: the common invocation quotes the
    # glob to stop the shell eating it when the run directory is remote.
    paths: list[str] = []
    for pattern in args.paths:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    limit = 0 if args.full else SNIPPET
    seen = 0
    for path in paths:
        try:
            summarise(path, limit)
            seen += 1
        except (OSError, json.JSONDecodeError) as exc:
            print(f"\n=== {path}\n    unreadable: {exc}", file=sys.stderr)

    if not seen:
        print("no readable session exports found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
