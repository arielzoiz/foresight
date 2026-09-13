#!/usr/bin/env python
"""Run one task through both experimental arms, N times each, and save the replies.

The cheapest honest test of the hypothesis: same task, same model, same code
path, and the ONLY difference is whether the future-task block is present. It
produces no score -- you read the two sets of replies and judge whether the
treatment code left the door open for those future tasks. Use it to find out
whether there is any signal at all before investing in benchmark plumbing.

Why one config and not two
--------------------------
Both arms are derived from a single config by flipping ``builder.name``, which
is exactly the claim the design makes ("switching arms is one config value").
Two hand-maintained config files could drift, and any drift would silently
become a second independent variable. Only the trace path is also overridden, so
the arms do not interleave into one file.

Why in-process
--------------
Each arm is a real foresight app driven over httpx.ASGITransport, the same way
tests/test_e2e.py does it. Everything except the socket is real -- routing,
pipeline, adapter, builder, backend, tracer -- and there are no ports to collide
and no servers to leave running.

Runs are interleaved (B, A, B, A, ...) so neither arm gets a systematically
warmer or colder model.

Usage:
    python tools/ab_run.py --config configs/ollama.yaml --runs 5 \\
        --task "Add a --dry-run flag to the sync command in cli/sync.py ..."

Watch it live, from another terminal:
    tail -f runs/<timestamp>/trace-B.jsonl
    tail -f runs/<timestamp>/progress.log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foresight.config import Runtime, load_config
from foresight.server import create_app

ARMS = (("B", "template", "treatment"), ("A", "passthrough", "control"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ab_run")
    parser.add_argument("--config", default="configs/ollama.yaml")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--task", default=None, help="the task prompt")
    parser.add_argument("--task-file", default=None, help="read the task from a file")
    parser.add_argument("--out", default="runs")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="per-request seconds"
    )
    args = parser.parse_args()
    if not args.task and not args.task_file:
        parser.error("give --task or --task-file")
    return args


def build_arms(config_path: str, outdir: Path):
    """One config in, two wired apps out, differing only in builder."""
    base = load_config(config_path)
    built = {}
    for arm, builder, _label in ARMS:
        cfg = base.model_copy(deep=True)
        cfg.builder.name = builder
        cfg.trace.path = str(outdir / f"trace-{arm}.jsonl")
        runtime = Runtime(cfg)
        built[arm] = (runtime, create_app(runtime))
    return base, built


async def one_call(app, served: str, task: str, timeout: float) -> tuple[str, float]:
    transport = httpx.ASGITransport(app=app)
    payload = {"model": served, "messages": [{"role": "user", "content": task}]}
    started = time.monotonic()
    async with httpx.AsyncClient(
        transport=transport, base_url="http://ab", timeout=timeout
    ) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
    took = time.monotonic() - started

    if resp.status_code != 200:
        return f"<HTTP {resp.status_code}: {resp.text[:400]}>", took
    body = resp.json()
    try:
        return body["choices"][0]["message"]["content"], took
    except (KeyError, IndexError, TypeError):
        return f"<unreadable reply: {json.dumps(body)[:400]}>", took


def last_trace_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main() -> None:
    args = parse_args()
    task = (
        Path(args.task_file).read_text().strip() if args.task_file else args.task.strip()
    )

    outdir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    progress = (outdir / "progress.log").open("a", buffering=1)

    def say(line: str) -> None:
        print(line, flush=True)
        progress.write(line + "\n")

    base, built = build_arms(args.config, outdir)
    served = base.models["target"].served_name

    say(f"task    : {task}")
    say(f"config  : {args.config}  (target={base.models['target'].model})")
    say(f"runs    : {args.runs} per arm, interleaved")
    say(f"outdir  : {outdir}")
    say(f"watch   : tail -f {outdir}/trace-B.jsonl")
    say("")
    (outdir / "task.txt").write_text(task + "\n")

    try:
        for i in range(1, args.runs + 1):
            for arm, _builder, label in ARMS:
                runtime, app = built[arm]
                reply, took = await one_call(app, served, task, args.timeout)

                row = last_trace_row(Path(runtime.tracer.path))
                sent = row["prompt_out"] if row else task
                aux = (row or {}).get("aux") or {}
                verdict = (
                    "n/a"
                    if arm == "A"
                    else ("usable" if aux.get("usable") else f"WEAK({aux.get('items')})")
                )

                path = outdir / f"{arm}-run{i}.md"
                path.write_text(
                    f"# arm {arm} ({label}) — run {i}\n\n"
                    f"- reply chars: {len(reply)}\n"
                    f"- seconds: {took:.1f}\n"
                    f"- aux verdict: {verdict}\n\n"
                    f"## prompt actually sent to the model\n\n"
                    f"```\n{sent}\n```\n\n"
                    f"## reply\n\n{reply}\n"
                )
                say(
                    f"run {i} arm {arm} ({label:9}) "
                    f"{took:6.1f}s  reply={len(reply):5}ch  aux={verdict}"
                )
        # Inside the try, before the finally closes the log `say` writes to.
        summarise(outdir, say)
    finally:
        for runtime, _ in built.values():
            await runtime.aclose()
        progress.close()


def summarise(outdir: Path, say) -> None:
    say("")
    say("=" * 70)
    for arm, _builder, label in ARMS:
        rows = read_rows(outdir / f"trace-{arm}.jsonl")
        replies = sorted(outdir.glob(f"{arm}-run*.md"))
        lengths = [len(p.read_text().split("## reply\n\n", 1)[-1]) for p in replies]
        if not lengths:
            continue
        say(f"arm {arm} ({label})  n={len(lengths)}")
        say(
            f"  reply chars   min={min(lengths)} "
            f"median={int(statistics.median(lengths))} max={max(lengths)}"
        )
        say(f"  enhanced      {sum(1 for r in rows if r.get('enhanced'))}/{len(rows)}")
        if arm == "B":
            weak = [r for r in rows if (r.get("aux") or {}).get("usable") is False]
            say(f"  weak aux      {len(weak)}/{len(rows)}  <- exclude these when judging")
    say("")
    say("Reply length is a sanity check, not a result. Read the code:")
    say(f"  ls {outdir}")
    say("Ask of each pair: did arm B leave the door open for the future tasks")
    say("(a seam, a parameter, a separated step) where arm A hard-coded it?")


if __name__ == "__main__":
    asyncio.run(main())
