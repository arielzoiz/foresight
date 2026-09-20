# SWE-CI A/B: control vs foresight (5 tasks, 5 epochs, one run per arm)

Run on 2026-09-19 from a macOS laptop (Docker Desktop, TAU VPN). Both models were served
by vLLM on one cluster node; nothing on the cluster was touched by this run.

- **Control:** SWE-CI (`opencode`) -> target model directly.
- **Foresight:** SWE-CI (`opencode`) -> foresight -> target model; foresight's aux agent
  explores the live task container (`docker exec`) and enriches the prompt.
- **Models:** target `target-model` and aux `aux-model`, both `Qwen3-Coder-30B-A3B-Instruct-FP8`
  on separate URLs (ports 8001 / 8002 of the same node), no auth.

## Identical settings in both arms

`agent_name = opencode`, `mode = tdd`, `splitting = default`, `[init] max_workers = 1`,
`[evolve] max_workers = 1`, `max_epoch = 5`, architect and programmer `max_try = 3`.
The two SWE-CI configs differ only in `experiment_name` and `base_url`
(`configs/config_control.toml` vs `configs/config_foresight.toml`). Mac-only settings
(`storage_disk = local-docker-desktop`, empty read/write bps, memory 2048mb/1024mb) are in both.
`max_workers = 1` is required: foresight identifies the calling container by IP or as the
only running container.

Tasks: the first 5 rows of `metadata/default.csv` (`configs/tasks.txt`).

## Results

Gap per epoch, starting state first (lower is better). EvoScore from `swe_ci.summarize`.

| Task | Control gaps | Foresight gaps | Control EvoScore | Foresight EvoScore |
|---|---|---|---|---|
| inline-snapshot | 16, 16, 17, 18, 17, 106 | 16, 16, 16, 16, 16, 18 | -0.0625 | -0.0013 |
| copyparty 452592 | 16, 16, 8, 8, 8, 8 | 16, 16, 4, 4, 4, 4 | 0.4000 | 0.6000 |
| copyparty 8c52b8 | 7, 7, 7, 7, 7, 6 | 7, 7, 6, 6, 6, pytest not executed | 0.0286 | -0.1143 |
| copyparty ccdace | 10, 8, 3, 3, 3, 3 | 10, 4, 3, 3, 3, 3 | 0.6000 | 0.6800 |
| anyio 439951 | 73, 149, 148, 148, 149, 147 | 73, 111, 64, 63, 64, 95 | -0.0735 | 0.0650 |
| **Average** | | | **0.1785** | **0.2459** |

Resolved 0.0 in both arms. Zero-regression rate 0.6 (control) vs 0.4 (foresight).
Plots: `plots/results_gap_per_epoch.png`, `plots/results_evoscore.png`
(regenerate with `python plot_results.py`).

## Foresight / aux provenance (`data/foresight/trace.jsonl.gz`, `aux_sessions.json`)

- 1,695 trace rows (1,219 target, 476 aux). 50 aux sessions (25 architect + 25 programmer).
- `resolved_by`: `sole_container` for all 50. `fallback`: none, so **0 `body_only`, nothing excluded**.
- All 50 aux results usable; guard `intact` on all 50 (aux never modified the workspace); 0 trace errors.
- Aux run time: mean 10.5 s, max 24.2 s. The prompt sent to the target grew by mean 1,761
  chars (max 2,736). All 1,157 agent-turn target requests were enhanced.

## Incidents

- **Control, copyparty 8c52b8, epoch 5:** architect attempt 1 wrote no `/app/requirement.xml`;
  attempt 2 succeeded.
- **Control, anyio, epoch 3:** architect attempt 1 failed with a context-length error (prompt
  about 230k tokens + 32k requested output = 262,145 vs a 262,144 limit); attempt 2 succeeded.
- **Foresight, copyparty 8c52b8, epoch 5:** the programmer left a `SyntaxError` in
  `copyparty/authsrv.py` (line 1010, "expected 'except' or 'finally' block"). Four of five test
  files import it, so pytest reported "collected 5 items / 4 errors", returned code 2, and wrote
  no results. SWE-CI recorded `gap: -1`, skipped the epoch's scoring and ended the task
  ("Reached the exit condition"). This is why that task scores below control. A model coding
  slip; guard `intact`, so aux did not cause it.
- Container exits with code 137 in the log are normal teardown after each pytest run, not OOM.
- The foresight arm had no architect or programmer retries.

## Timing (indicative only: the two arms shared the model server)

| | Start -> end (local) | Init | Evolution | Total |
|---|---|---|---|---|
| Control | 19:33 -> 21:40 | 23 min | 1 h 43 min | 2 h 07 min |
| Foresight | 21:41 -> 23:09 | 16 min | 1 h 12 min | 1 h 28 min |

Foresight init was faster because the Docker images were already built. Control's anyio
(52 min vs 12 min) includes about 33 minutes of architect retries.

## Caveats

- 5 tasks, one run per arm, sampled model output: the differences are suggestive, not
  significant. One bad edit (as in copyparty 8c52b8, or control's inline-snapshot regression)
  moves a task's score a lot.
- 5 epochs is short for a claim about future maintainability. Gaps plateau early in several
  tasks in both arms, so more epochs may not help without other changes.
- Best next step: repeat the same 5 tasks (new experiment names) to measure run-to-run noise,
  then a longer-horizon experiment as its own run. A passthrough-builder arm would separate the
  effect of the aux text from the proxy itself.

## Code change and maintainability per epoch

From `metrics.csv` (per arm, task and epoch end) and `RUN.md`; plots `plots/results_churn_per_epoch.png`,
`results_mi_per_epoch.png`, `results_pylint_per_epoch.png`. Lines changed count accepted epochs only;
the one epoch whose pytest could not run (foresight, copyparty 8c52b8, epoch 5) is marked `!`.

| Task | Lines changed, control | Lines changed, foresight | MI change, control | MI change, foresight |
|---|---|---|---|---|
| inline-snapshot | 218 | 218 | +0.25 | +0.95 |
| copyparty 452592 | 87 | 33 | -0.02 | -0.00 |
| copyparty 8c52b8 | 107 | 309 | -0.02 | -0.05 |
| copyparty ccdace | 49 | 83 | -0.01 | -0.00 |
| anyio 439951 | 349 | 170 | -0.26 | -0.13 |
| **Total** | **810** | **813** | | |

- The arms changed about the same amount of code in total (810 vs 813 lines), but distributed very
  differently: foresight made larger edits on copyparty 8c52b8 (309 vs 107) and ccdace (83 vs 49),
  and smaller ones on copyparty 452592 (33 vs 87) and anyio (170 vs 349).
- **Neither score moved much.** The pylint note changed by at most 0.04 in either arm. The
  maintainability index (radon, weighted by source lines) barely moves on the large copyparty repo,
  where an edit of tens of lines is a rounding error against about 27,000 statements. The one visible
  difference, inline-snapshot (MI +0.95 for foresight vs +0.25 for control), is a single task and a
  single run, and that task's test gap did not improve in either arm, so it is not evidence that
  foresight yields more maintainable code.
- Read these as "did the code get worse or better while the tests moved", not as a measure of the
  size of an effect. A real maintainability comparison needs more tasks, more epochs and repeated
  runs.
- Scoring notes: `mi` is SWE-CI's own `mi_score`; `pylint` is a corrected pylint run, because SWE-CI's
  `pylint_score` never passes `--recursive=y` and returns 0 on these repos (SWE-CI does not call it).
  Both exclude `tests/`. The experiment folders sat on an external drive, where macOS wrote `._*`
  sidecar files beside every moved file; the first scoring pass counted them as syntax errors and made
  every later epoch look worse than the start in both arms alike. They are now excluded from the diffs
  and the scores.

## What else is in this folder

- `RUN.md` is generated by `tools/collect_swe_ci_ab.py` from the files below (run summary, results, code
  change and quality per epoch, timing, cost, aux provenance, incidents); this file is the interpretation.
  `manifest.json` holds the rows, each task's repo and both commits (from the benchmark csv), the settings,
  and whether the arms' settings matched (they did). `metrics.csv` is one row per arm, task and epoch end.
  `environment.txt` records the commits, versions, hardware and the model endpoints. The model servers had
  expired by the time of the last collection, so their lines were kept from the earlier one, and say so.
- `data/<arm>/<task>/epoch_<N>/` holds, per epoch: `non-passed/summary.jsonl` (the failing tests the
  architect saw at the start of the epoch), `requirement.xml` (the architect's output) and
  `edit.diff` (the code change the programmer made, `a/` = before, `b/` = after). For the task whose
  last epoch broke pytest (copyparty 8c52b8, foresight arm), `epoch_5/edit.diff` is the patch that
  introduced the `SyntaxError`.
- `data/<arm>/<task>/final/non-passed/summary.jsonl` is the failing-test list AFTER the last epoch (each
  `epoch_<N>/` holds the state at the START of epoch N). Its length equals the last recorded gap for all
  ten task and arm pairs; for foresight's copyparty 8c52b8, whose last epoch could not run pytest, it is the
  last accepted state. `data/<arm>/swe_ci_stdout.log` is the console output of the `swe_ci.evaluate`
  process (SWE-CI's effective config and the final table), not the agent's log.
- What this run did NOT record: the aux agent's own reads and tool calls, and the target model's replies
  (`adapter.aux_export_dir` and `trace.log_replies` did not exist yet); the per-epoch code snapshots and
  `test_report.json` files, which stay only in `SWE-CI/experiments/` (about 1.5 GB). The vLLM setup is in
  `server-side/setup.md`, as reported by the server side; the sampling defaults the servers applied are
  still missing. `examples/aux-export-sample/` comes from a separate one-task test run.
- Read `requirement.xml` as an OUTPUT, not an input: in the foresight arm it was written after
  foresight added aux's future-task list to the prompt. The base task of an architect session is the
  failing-test summary; the base task of a programmer session is that epoch's `requirement.xml`.
  Epoch 1 is the cleanest place to compare the arms, since both start from identical code and tests.
- `data/foresight/AUX_PAIRS.md` puts each aux answer next to its base task (the failing tests for an
  architect session, that epoch's `requirement.xml` for a programmer session) and the task's commits.
  `data/foresight/aux_sessions.json` also records each aux run's `task_id`, `epoch` and `phase`
  (joined by timestamp against `task.log`, 50 of 50 matched). It holds aux's final answer only; the
  aux agent's own reads and tool calls were not recorded in this run. `examples/aux-export-sample/`
  shows what the export added afterwards records, from a separate one-task test run.
- The foresight code used for this run is commit `2d65a15`. The aux export and the collector were
  written after it.

## Reproducing (local side only)

1. SWE-CI: `PYTHONPATH=src .venv/bin/python -u -m swe_ci.evaluate --config_file config_control.toml`.
2. Start foresight (`python -m foresight.server --config configs/swe_ci_expt.yaml`, port 8010),
   then `... --config_file config_foresight.toml`. Run the arms one after the other.
3. `python tools/collect_swe_ci_ab.py ...` gathers both arms into a results folder; the command and
   the naming convention (`ab-test__swe-ci__rows-<rows>__ep<epochs>`) are in the main README.

The `swe_ci_expt.yaml` here is the foresight config used (local model URLs, `host.docker.internal`
for the containers, no API keys). `swe_ci_config` inside it is an absolute path on the
laptop and must be edited.
