"""Replay a model's raw output through every vLLM tool-call parser. No GPU.

Why this exists
---------------
When an agent run comes back ungrounded, the question is always "would ANY
parser have recognised what the model emitted?" -- and answering it by
restarting vLLM with a different `--tool-call-parser` costs a queue wait plus a
multi-GB checkpoint load, per guess.

It does not have to. A tool parser is a pure function from the model's output
text to a list of tool calls; it touches no GPU and no weights, only a
tokenizer. So every candidate can be tried offline, in seconds, against the
exact string the model produced -- which the aux transcript
(`$RUN/aux-agent/ses_*.json`, see tools/aux_transcript.py) now preserves
verbatim.

Use it to decide whether a GPU run is worth submitting at all: if no parser
recognises the output, the fix is the model or the prompt, not a flag.

Usage:
    python tools/probe_tool_parsers.py --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct
    python tools/probe_tool_parsers.py --tokenizer <id> --text-file captured.txt
    python tools/probe_tool_parsers.py --tokenizer <id> --parsers hermes,xlam
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The samples below are the shapes actually observed on this project, kept here
# so the default run reproduces the known cases rather than needing a capture.
SAMPLES: dict[str, str] = {
    # What Qwen2.5-Coder-7B-Instruct emitted under the tool-forcing aux prompt,
    # recorded in deploy/tau-slurm/README.md. A fenced block, single object.
    "qwen2.5-7b: fenced json, single object": (
        "I'll inspect the repository.\n\n"
        '```json\n{"name": "glob", "arguments": {"pattern": "**/*.{js,ts}"}}\n```'
    ),
    # The same thing as an array -- included to isolate whether the array
    # requirement (not the fence) is what rejects the sample above.
    "fenced json, array of one": (
        "I'll inspect the repository.\n\n"
        '```json\n[{"name": "glob", "arguments": {"pattern": "**/*.py"}}]\n```'
    ),
    # Bare object, no fence.
    "bare json object": '{"name": "glob", "arguments": {"pattern": "**/*.py"}}',
    # The other shape the 7B was reported to produce.
    "function_call xml": (
        '<function_call>{"name": "glob", "arguments": {"pattern": "**/*.py"}}'
        "</function_call>"
    ),
    # The format hermes actually wants -- the control. If this does not parse,
    # the harness is broken, not the model.
    "hermes tool_call tags": (
        '<tool_call>\n{"name": "glob", "arguments": {"pattern": "**/*.py"}}\n</tool_call>'
    ),
    # What Qwen3-Coder's chat template emits.
    "qwen3-coder xml": (
        "<tool_call>\n<function=glob>\n<parameter=pattern>\n**/*.py\n"
        "</parameter>\n</function>\n</tool_call>"
    ),
}


def build_request(tools: list[dict]):
    # Moved between vLLM versions (0.29 has it under chat_completion/), so
    # import it the way the parsers themselves do rather than by a fixed path.
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
    except ImportError:  # older layout
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    return ChatCompletionRequest(
        model="probe",
        messages=[{"role": "user", "content": "list the python files"}],
        tools=tools,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HF id or path; read from HF_HOME, no download if already cached",
    )
    parser.add_argument(
        "--text-file",
        help="file holding one captured model output; overrides the built-in samples",
    )
    parser.add_argument(
        "--parsers",
        help="comma-separated subset to try (default: every registered parser)",
    )
    args = parser.parse_args()

    # Must precede the vllm import: on a login node there is no GPU, and this
    # keeps the platform probe quiet rather than warning about it.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    from transformers import AutoTokenizer
    from vllm.tool_parsers import ToolParserManager

    print(f"loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "find files by pattern",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            },
        }
    ]
    request = build_request(tools)

    if args.text_file:
        with open(args.text_file, encoding="utf-8", errors="replace") as handle:
            samples = {args.text_file: handle.read()}
    else:
        samples = SAMPLES

    # NOT ToolParserManager.tool_parsers -- parsers are registered lazily, so
    # that dict is empty until each module is imported and the sweep would
    # silently test nothing. (Caught only because the hermes control sample,
    # which must parse, came back NONE like everything else.)
    if hasattr(ToolParserManager, "list_registered"):
        names = sorted(ToolParserManager.list_registered())
    else:
        names = sorted(
            set(ToolParserManager.tool_parsers) | set(ToolParserManager.lazy_parsers)
        )
    if not names:
        print("no tool parsers registered -- cannot probe", file=sys.stderr)
        return 1
    if args.parsers:
        wanted = {n.strip() for n in args.parsers.split(",")}
        names = [n for n in names if n in wanted]

    for label, text in samples.items():
        print(f"\n{'=' * 70}\nOUTPUT: {label}\n{'-' * 70}")
        print(text)
        print("-" * 70)

        hits: list[str] = []
        for name in names:
            try:
                cls = ToolParserManager.get_tool_parser(name)
                instance = cls(tokenizer)
            except Exception as exc:  # noqa: BLE001 -- report, never abort the sweep
                # Common and informative: several parsers demand a sentinel
                # token that this tokenizer does not define, so they could never
                # have served this model regardless of output shape.
                reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
                print(f"  {name:22} unusable: {reason}")
                continue

            try:
                result = instance.extract_tool_calls(text, request)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:22} raised: {type(exc).__name__}: {str(exc)[:60]}")
                continue

            if getattr(result, "tools_called", False) and result.tool_calls:
                calls = ", ".join(
                    f"{c.function.name}({c.function.arguments})" for c in result.tool_calls
                )
                print(f"  {name:22} *** PARSED: {calls}")
                hits.append(name)

        print("-" * 70)
        print(f"  parsers that recognised this output: {', '.join(hits) if hits else 'NONE'}")

    print(
        "\nNote: a parser appearing here is necessary but not sufficient -- it must\n"
        "also be one the SERVER can run for this checkpoint, and it must not break\n"
        "the model's normal (non-tool) output. Verify on a real server before\n"
        "trusting it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
