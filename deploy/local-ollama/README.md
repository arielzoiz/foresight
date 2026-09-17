# foresight on a local machine, with Ollama

**This is one deployment target, not part of the `foresight` library.** The
library (`foresight/`) knows nothing about Ollama or Docker Desktop;
everything here is what it takes to run it — plus a real model and a real
opencode agent — on a laptop or workstation with no cluster, no shared GPU,
and (for `SweCiAdapter`) a Docker Desktop install. If you're on the TAU Slurm
cluster instead, see `deploy/tau-slurm/`.

It answers two different questions, and they need different setups:

| | `GenericAdapter` / `LocalAdapter` | `SweCiAdapter` |
|---|---|---|
| What it tests | the proxy, the prompt, a real aux agent over a real repo | the same, plus real Docker mechanics |
| Needs | Ollama, opencode | Ollama (or a fake upstream), opencode, Docker Desktop |
| Configs | `configs/ollama.yaml`, `configs/local.yaml` | `configs/swe_ci.yaml` |
| Milestone | 1 / 2 | 3 |

## 1. One-time setup

```sh
brew install ollama          # starts as a background service on install
ollama pull qwen3:8b         # see §3 before picking a model
curl http://127.0.0.1:11434/api/version

npm install -g opencode-ai
opencode --version
```

Ollama serves an OpenAI-compatible API at `/v1` with no auth, so
`backend: openai_compat` needs no `api_key_env` — there is no key to send.

**If `ollama pull`/`ollama` commands hang with no output:** check
`ps aux | grep "ollama serve"` for its process **state**, not just that it
exists. A `T` (stopped) state means it was suspended — most often a stray
Ctrl+Z instead of backgrounding it properly — not crashed; `kill -CONT <pid>`
resumes it, no restart needed. Start it as `ollama serve &` or via
`brew services start ollama` to avoid this.

## 2. Running

```sh
# Milestone 1: aux is one chat call over the request body, no repo, no opencode needed
python -m foresight.server --config configs/ollama.yaml

# Milestone 2: aux is a real opencode subprocess over a real repo -- EDIT
# adapter.workspace and adapter.agent_cmd[0] first
python -m foresight.server --config configs/local.yaml
```

Then drive it as the target caller yourself:

```sh
tools/opencode_home.sh --home /tmp/foresight-target-home --model target-model \
    --base-url http://127.0.0.1:8000/v1 --context 8192 --output 2048

cd <the workspace configs/local.yaml points at>
HOME=/tmp/foresight-target-home opencode run --model custom/target-model \
    "<a real coding task for this repo>"
```

`tools/opencode_home.sh` writes the `auth.json`/`opencode.json` pair opencode
needs for a `custom` provider — do this for **both** the target's home
(above) and aux's home (whatever `adapter.env.HOME` names in the config you
picked), each with its own `--model` (`target-model` / `aux-model`). Give
them different `--home` directories: opencode's session DB lives under
`$HOME`, and sharing it between aux and target corrupts both.

## 3. Which model, and what actually happened when we tried

**Skip vLLM here.** It's CUDA/ROCm-first with no real Apple Silicon backend;
Ollama is what `configs/ollama.yaml`/`configs/local.yaml` already target, and
it's the only thing that's actually been made to work locally on this
project so far.

**None of the models tried so far are trustworthy as aux**, and the failure
mode is worth knowing before you spend time on your own:

| Model | Tool-call format | Aux reliability | Notes |
|---|---|---|---|
| `qwen2.5-coder:7b` / `:14b` | **broken** | n/a | emits bare JSON instead of `<tool_call>` tags its own template demands; never structures into `tool_calls` regardless of size |
| `qwen3:8b` | correct | **~10% success** | 9 of 10 real aux invocations exited with empty stdout; the one success was aux asking for help with its own broken tool call, which the quality gate still passed |
| `qwen3:14b` | correct | mechanically reliable, **ungrounded** | never failed outright, but took 462s and still never found the one file in a two-file repo |
| `qwen3-coder:30b` (the model this project's own results already flagged as worth trying) | — | **doesn't fit** | 18.56GB minimum at Ollama's lightest quant — rules out any Mac with 16GB or less regardless of quality |

Full detail, trace evidence and reproduction steps:
`results/qwen3-8b-ollama-local/`, `results/qwen3-14b-ollama-local/`.

**Before trusting any model here, check its raw tool-calling shape first** --
cheaper than a full opencode run:

```sh
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model": "<name>", "stream": false,
  "messages": [{"role":"user","content":"What is the exact content of /tmp/x.txt? You cannot know this without reading it."}],
  "tools": [{"type":"function","function":{"name":"read_file","description":"Read a file",
    "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
}'
```

A working model returns a populated `tool_calls` array and empty `content`.
Bare JSON *inside* `content`, with no `tool_calls` field at all, means this
model+server combination cannot drive a real agent regardless of how well it
tests in isolation — this is exactly how `qwen2.5-coder` was ruled out.

## 4. `SweCiAdapter` + Docker, locally

Three things are specific to running this on a Mac rather than on real Linux
Docker (the primary target `configs/swe_ci.yaml` assumes), all measured
directly rather than assumed:

**`docker` may not be on `PATH`.** Docker Desktop's CLI symlink
(`/usr/local/bin/docker`) is not guaranteed to be installed — on this
project's own test machine it was not, and `/usr/local/bin` isn't writable
without `sudo`. Check `command -v docker` first; if it's missing, either
install the symlink from Docker Desktop's own Settings → Advanced → "Install
CLI tools", or add its real location to `PATH` for the session:

```sh
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

**Use `host.docker.internal`, not `172.17.0.1`, in `foresight_base_url`.**
`configs/swe_ci.yaml`'s default (`172.17.0.1`, the bridge gateway) is correct
for real Linux Docker but does **not** reach the host under Docker Desktop —
measured directly: `curl` from inside a container to `172.17.0.1` fails, to
`host.docker.internal` succeeds. This only applies to the direction
*container → host* (aux calling back into foresight); foresight itself
runs on the host and reaches Ollama at plain `127.0.0.1`, same as anywhere
else — don't apply this substitution there too.

**Memory: this combination did not fit in 16GB.** Running Docker Desktop's
VM (its own multi-GB reservation), a real SWE-CI task container, and a
locally-loaded Ollama model *at the same time* got the process killed for
low system memory twice in a row on a 16GB M1 Pro — once with `qwen3:14b`,
once again after switching to the smaller `qwen3:8b`, ruling out "just use
the smaller model" as the fix. If you hit this: free memory before starting
(`ollama stop <model>`, close other apps), reduce Docker Desktop's own VM
memory allocation (Settings → Resources), or point `models.target`/
`models.aux` at a model server running on a *different* machine instead of
loading one locally — `base_url` is just a URL, nothing else about this setup
changes. See `results/swe-ci-mechanics-m1-docker/findings.md` for the full
account, including what **did** work (all the actual Docker/exec/bootstrap
mechanics, verified clean with `tools/fake_upstream.py` standing in for the
model).

**SWE-CI's own `config.toml` loader crashes on macOS**, independent of any of
the above — `get_docker_storage_disk()` (in the separate `SWE-CI` repo, not
this one) unconditionally raises on any non-Linux `platform.system()`. Its
own `--config_file` mechanism is the sanctioned fix: make a copy (e.g.
`config_local.toml`) with `docker.storage_disk` set to any non-empty
placeholder and `docker.read_bps`/`write_bps` left empty (those are what
would actually need `storage_disk`'s value, in a real `--device-read-bps`
flag; leaving them empty means the placeholder is never used for real), then
pass `--config_file config_local.toml` to every SWE-CI command.

## 5. Recording a finished run

```sh
tools/collect_local_run.sh --config PATH --trace PATH --workspace DIR \
    --aux-home DIR --label <label>
$EDITOR results/<label>/findings.md      # the one part no script can write
git add results/<label> && git commit
```

Sibling to `deploy/tau-slurm`'s `tools/collect_run.sh`, for a run with no
shared Slurm run directory: config, trace and aux's opencode session
transcripts come from separate paths you pass explicitly, and
`run-settings.txt` records the local server's own version and this host's
hardware instead of `sacct`/vLLM. See `results/README.md` for what each file
in a results directory is.

**opencode's own `session list`/`export` may not work at all**, independent
of anything above — confirmed broken for headless `opencode run` sessions on
opencode 1.18.30 specifically (`export` answers "Session not found" for a
session ID read directly out of its own database). When this happens,
`collect_local_run.sh` still produces `config.yaml`, `trace.jsonl` and
`prompts.txt`; only `aux-agent/*.json` and the `summary.txt` derived from
them are affected. Check `opencode --version` before assuming the collection
script is broken.

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ollama pull`/`ollama <anything>` hangs forever | `ollama serve` is suspended (`T` state in `ps aux`), not crashed — `kill -CONT <pid>` |
| model returns bare JSON in `content`, no `tool_calls` field | this model cannot drive tool calls through Ollama regardless of size — see §3, try a different model family |
| aux exits 0 with empty stdout | the model attempted a tool call and omitted a required argument, then didn't recover with a text answer — a model-reliability problem, not a config one; see `results/qwen3-8b-ollama-local/` |
| `command not found: docker` | Docker Desktop's CLI symlink isn't installed — see §4 |
| `SweCiAdapter` aux fails with opencode's generic `UnknownError: Unexpected server error` | check `docker exec <cid> cat /tmp/aux-home/.local/share/opencode/auth.json` inside the container for a zero-byte file — a real, now-fixed bug (`_exec_argv` needs `stdin=True` for the bootstrap call) |
| every aux call after the first `docker exec`s into a container that's already gone | a real, now-fixed bug in container-ID cache invalidation across container recreation — needs foresight `>= db3a5bd`/`52c16d5` |
| process killed with no error, "low system memory" | Docker Desktop's VM + a container + a loaded model together exceeded available RAM — see §4, this is a hardware ceiling, not a hang to wait out |
| SWE-CI's own commands raise `NotImplementedError: darwin` on import | `get_docker_storage_disk()` — use a `--config_file` override, see §4 |
| aux's opencode session directory is wrong / `collect_local_run.sh` finds no sessions to export | check `foresight` is `>= db3a5bd` (the `$PWD` fix) — an older build silently misattributes every `LocalAdapter` aux session to wherever the foresight server itself was launched from |
