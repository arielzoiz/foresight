# Handoff: SWE-CI A/B run 2 (rows 6+, 20 epochs), 2026-09-20/21

Target and aux: `Qwen3-Coder-30B-A3B-Instruct-FP8` on `n-b200.cs.tau.ac.il` (:8001 `target-model`, :8002 `aux-model`).
Driver, scripts and logs: `SWE-CI/runs/orchestrate/` (`pipeline.sh`, `collect_batch.sh`, `bonus_control.sh`, `final_recollect.sh`, `events.log`).
Per-run working dirs: `SWE-CI/runs/c<row>/` (control) and `runs/f<a>-<b>/` (foresight); `SWE-CI/experiments/` holds links to them.

## Still to complete (check each; the snapshot below says which apply)

1. **Server-side part** into `results/<label>/server-side/` for every batch (checklist in `TEMPLATE.md`): vLLM version, launch flags,
   sampling defaults, hardware, job ids/times, preemptions/OOMs, other node load, vLLM logs. Do this before the jobs are cancelled, because that information (especially the vLLM logs) is lost afterwards.
2. **Cancel the vLLM jobs** (`scancel`) only after item 1 (they do not stop themselves; they end at ~03:57 IDT anyway).
3. **Findings** (`findings.md`) per batch. The collector only writes `RUN.md` (data), not interpretation.
4. **Pylint scores:** batches were first collected with `--no-pylint` (maintainability index only). `final_recollect.sh` reruns them
   with pylint after the experiments; if a batch shows `COLLECT FAILED` in the events below, rerun
   `ROWS="<a> <b>" SCORE_FLAGS="--score-jobs 2" bash SWE-CI/runs/orchestrate/collect_batch.sh`.
5. **Truncated or unfinished tasks:** any foresight task stopped at the 03:10 cutoff is partial (report over the first k epochs only,
   not the 20-epoch score). Any row listed under "Control-only rows" below has NO foresight arm yet: run it later on new vLLM jobs
   (same checkpoint and launch flags; note the different job/node), using the prepared `SWE-CI/config_foresight_r<a>-<z>.toml` and
   `foresight/configs/swe_ci_ab-test__swe-ci__foresight__rows-<a>-<z>__ep20.yaml`, then re-collect with both arms (see item 4's command).
6. **Merging batches** (rows 6-7, 8-9, 10-11, same 20-epoch cap): recompute EvoScore over all tasks from the `iteration.jsonl` files with
   one epoch cap; averaging batch averages is only valid for equal task counts. No merge tool exists yet.
7. **Commit** the results folders (I did not commit anything).
8. **Cleanup** when done: `SWE-CI/runs/` (about 0.5 GB per task), `experiments/.partial-*`, `traces/server_*.log`.

## Things that differ from the plan or that a reader should know

- Control rows were run as separate SWE-CI processes (one per row, own `runs/c<row>/` dir), all against :8001 (`target-model`).
- Because control rows ran concurrently (two at a time) and foresight rows one at a time, timing between arms is not comparable
  (tokens and gaps are). The Mac (6 cores, 16 GB, Docker 8 GB) was saturated during the runs.
- Rows 6-9 were collected first with `--no-pylint`; see item 4.
- Foresight identified the task container as `sole_container` in all checked batches; `fallback: body_only` rows must be excluded (none seen in batches 6-7 and 8-9).
- Row 8 foresight arm: last-epoch gap jumps to 208 (not investigated).
- Row 11 (`anthropic-sdk-python 16eb64`): pytest could not run (`x`) in EVERY epoch, in both arms, so this task carries no usable
  gap data. Check whether it is an environment/test-infrastructure problem (task image, tests) rather than a model result before using it.
- Row 13 (`argoproj-labs/hera 7ac8d5`) control was interrupted at the 03:40 bonus stop and resumed after the servers were replaced
  (SWE-CI resumes by experiment name). The interruption had recorded epoch 7 as a failed epoch (`gap = -1`) only because the operator
  removed the running containers; that artifact record was removed (backup: `SWE-CI/runs/orchestrate/backups/c13_iteration.jsonl.before-repair`)
  so epoch 7 is re-run from the same state. Epochs 0-6 were produced before the restart, epoch 7 onward after it, on new (identical) vLLM jobs.
  The control-only `rows-12-13` result collected at 03:56 predates this and is superseded by the later two-arm collection.
- Lesson for the driver: SWE-CI's ProcessPool workers do not carry the config name in their command line, so `pkill -f <config>` leaves them
  alive (holding the task's `.lock`, which makes a later run silently skip the task and print "Evolution complete"). `pipeline3.sh` and `bonus3.sh` now
  kill children and workers too.
- The first vLLM jobs ended 03:56 IDT; the replacement jobs' processes started 03:56:46 / 03:57:20 and passed three consecutive readiness checks
  (health, model id/root/context, a real 7k-token chat request) before the continuation started at 04:14. Load samples: `SWE-CI/runs/orchestrate/load3.csv`.
- Performance finding (important for any future run): the external LACIE disk is pathologically slow for small files (100 file
  create+delete: 64 s vs 0.06 s on the internal SSD; ~200 IOPS, under 1 MB/s). SWE-CI's per-epoch code copies and temp-folder deletes made epochs
  wait on it (one "delete temp folder" step took 11 min; on the SSD it took 0 s). Runs from row 14 on (and the foresight run of rows 12-13 and the
  resumed row 13 control) use working dirs on the internal SSD, `/Users/arielzoizner/swe-ci-runs/<name>`, linked from `SWE-CI/runs/`.
  Rows 6-12 ran from the external disk and are slower/less comparable in timing. Token counts and gaps are unaffected. Old copies:
  `SWE-CI/runs/c13.hdd-old`, `SWE-CI/runs/f12-13.moved-to-ssd`. A paused pylint re-collection (PID 1311, `kill -CONT` to resume) is
  reading many small files from that disk, so run it only when no experiment is running.
- Timeline of interruptions (see the event logs below): a VPN drop (GlobalProtect, 07:38 to 10:02 on 2026-09-21) stopped the run for 2 h 25 min with no loss of data; the first vLLM jobs
  ended 03:56 and the replacements ran until about 15:56, the next pair of jobs (14 h) started 16:33 and the run resumed at 16:44. The foresight cutoff of the second pair was extended
  from 15:07 to 15:40 by the operator because the servers stayed up until about 15:56.
- Row 15 (pypeln) foresight arm was completed across a job replacement (see `results/ab-test__swe-ci__rows-14-15__ep20/NOTES.md`).
- Hung tests are not specific to the foresight arm: foresight epochs of rows 12 and 15 hit SWE-CI's 3600 s pytest timeout, and so did a CONTROL epoch of row 18 (httpdbg). Long
  agent steps (60 min programmer step in row 16 foresight epoch 8, retried automatically by SWE-CI) also occur. None was intervened in. Together they cost several hours of wall-clock time,
  which is why fewer rows were completed than the throughput of the fast epochs suggested. A shorter pytest timeout for future runs would cap these hangs (it must be identical in both arms).

## Snapshot at 2026-09-22 03:01 (regenerated by gen_handoff.py; rerun at the end of the run)

### Results folders (rows 6 onward; 20 epochs, both arms unless listed under control-only)
- ab-test__swe-ci__rows-6-7__ep20
- ab-test__swe-ci__rows-8-9__ep20
- ab-test__swe-ci__rows-10-11__ep20
- ab-test__swe-ci__rows-12-13__ep20
- ab-test__swe-ci__rows-14-15__ep20
- ab-test__swe-ci__rows-16-17__ep20
- ab-test__swe-ci__rows-18__ep20

### Foresight/aux fallbacks per batch
- ab-test__swe-ci__rows-6-7__ep20: fallback: {None: 80} (any `body_only` run must be excluded from analysis)
- ab-test__swe-ci__rows-8-9__ep20: fallback: {None: 80} (any `body_only` run must be excluded from analysis)
- ab-test__swe-ci__rows-10-11__ep20: fallback: {None: 80} (any `body_only` run must be excluded from analysis)
- ab-test__swe-ci__rows-12-13__ep20: fallback: {None: 80} (any `body_only` run must be excluded from analysis)
- ab-test__swe-ci__rows-14-15__ep20: fallback: {None: 77} (any `body_only` run must be excluded from analysis)
- ab-test__swe-ci__rows-16-17__ep20: fallback: {None: 85} (any `body_only` run must be excluded from analysis)
- ab-test__swe-ci__rows-18__ep20: fallback: {None: 40} (any `body_only` run must be excluded from analysis)

### Control-only rows (foresight arm still to run)
- none so far

### Event log, first driver (rows 6-13 start, 2026-09-20 17:59 to 2026-09-21 03:57)
```
17:59:10 EVENT: BATCH rows 6 7 START (control ab-test__swe-ci__control__rows-6-7__ep20)
18:36:53 EVENT: CONTROL rows 6 7 FINISHED in 37 min
18:37:05 EVENT: FORESIGHT rows 6 7 START (server up on :8010)
20:24:06 EVENT: FORESIGHT rows 6-7 FINISHED in 107 min
20:32:31 EVENT: driver restarted on rows 8+ (collection moved to background); rows 6-7 collecting
20:32:31 EVENT: BATCH rows 8 9 START (control ab-test__swe-ci__control__rows-8-9__ep20)
20:33:44 EVENT: CONTROL row 8 START (target :8001)
20:35:14 EVENT: CONTROL row 9 START (target :8001)
21:31:07 EVENT: COLLECTED rows 6-7 into foresight/results (see traces/collect_6-7.log)
21:53:15 EVENT: CONTROL rows 8 9 FINISHED in 80 min
21:53:32 EVENT: FORESIGHT rows 8 9 START (server up on :8010)
23:29:29 EVENT: FORESIGHT rows 8-9 FINISHED in 96 min
23:29:29 EVENT: BATCH rows 8-9 DONE in 176 min; next estimate per 2 rows: 191 min
23:29:29 EVENT: BATCH rows 10 11 START (control ab-test__swe-ci__control__rows-10-11__ep20)
23:31:15 EVENT: CONTROL row 10 START (target :8001)
23:32:45 EVENT: CONTROL row 11 START (target :8001)
23:47:33 EVENT: COLLECTED rows 8-9 into foresight/results (see traces/collect_8-9.log)
00:27:59 EVENT: CONTROL rows 10 11 FINISHED in 58 min
00:28:12 EVENT: FORESIGHT rows 10 11 START (server up on :8010)
02:25:46 EVENT: FORESIGHT rows 10-11 FINISHED in 117 min
02:25:46 EVENT: BATCH rows 10-11 DONE in 176 min; next estimate per 2 rows: 190 min
02:25:46 EVENT: not enough time left (44 min) for batch 12,13; finishing
02:25:46 EVENT: PIPELINE DONE
02:25:50 EVENT: BONUS control-only rows 12 13 START (74 min until hard stop)
03:39:32 EVENT: COLLECTED rows 10-11 into foresight/results [no pylint] (see traces/collect_10-11.log)
03:40:55 EVENT: BONUS: hard stop reached, stopping unfinished control runs
03:41:33 EVENT: BONUS control-only finished; completed rows: 12 (of 12 13)
03:56:33 EVENT: BONUS COLLECTED control-only rows 12-13 into foresight/results (foresight arm still to run later)
03:57:20 EVENT: FINAL: experiments over; starting full re-collection with pylint
07:50:43 EVENT: COLLECTED rows 6-7 into foresight/results [--score-jobs 2] (see traces/collect_6-7.log)
08:16:16 EVENT: COLLECTED rows 8-9 into foresight/results [--score-jobs 2] (see traces/collect_8-9.log)
08:53:19 EVENT: COLLECTED rows 10-11 into foresight/results [--score-jobs 2] (see traces/collect_10-11.log)
08:53:19 EVENT: FINAL: full re-collection finished (see events above for any COLLECT FAILED)
08:53:20 EVENT: FINAL: handoff written to foresight/results/LACIE
```

### Event log, second driver (replacement servers, 2026-09-21 03:27 to 15:40)
```
03:27:34 EVENT: WAITER: watching for the old jobs to end (API down), then for the new identical jobs to serve
03:57:08 EVENT: API DOWN: old vLLM jobs ended; waiting for the new jobs (about 46 min after they start), checking every 5 min
04:02:10 EVENT: API BACK: both endpoints serve the same model (target-model, aux-model, ctx 262144); starting the continuation. Deadlines: foresight until 14:32, bonus until 14:57 (edit cutoff_epoch.txt / bonus_stop_epoch.txt if the real job end differs)
04:02:10 EVENT: BATCH rows 12 13 START (control ab-test__swe-ci__control__rows-12-13__ep20)
04:03:12 EVENT: ABORTED by operator: the continuation launched at 04:02:10 was stopped ~20 s later (servers may not be fully loaded; user asked to wait). No epoch was run. Waiting for a stricter readiness gate.
04:03:51 EVENT: GATE: waiting for 3 consecutive readiness passes (both servers: /health, model id/root/context, real 7k-token chat completion < 120 s)
04:03:52 EVENT: GATE: pass 1/3 - :8001 ok 0.2s prompt_tokens=7026 | :8002 ok 0.1s prompt_tokens=7026
04:05:52 EVENT: GATE: pass 2/3 - :8001 ok 0.2s prompt_tokens=7026 | :8002 ok 0.1s prompt_tokens=7026
04:07:53 EVENT: GATE: pass 3/3 - :8001 ok 0.2s prompt_tokens=7026 | :8002 ok 0.1s prompt_tokens=7026
04:07:55 EVENT: API BACK: both servers ready (processes started 03:56:46 / 03:57:20); starting the continuation. Deadlines: foresight until 15:07, bonus until 15:32 (edit cutoff_epoch.txt / bonus_stop_epoch.txt if the real job end differs)
04:07:56 EVENT: BATCH rows 12 13 START (control ab-test__swe-ci__control__rows-12-13__ep20)
04:08:17 EVENT: CONTROL row 13 START/RESUME (target :8001)
04:09:48 EVENT: CONTROL rows 12 13 FINISHED in 1 min
04:10:00 EVENT: FORESIGHT rows 12 13 START (server up on :8010)
04:14:35 EVENT: RESTART: the 04:08 start was invalid (an orphaned worker from the earlier bonus stop held row 13's lock, so SWE-CI skipped it; the driver then began foresight and was stopped). Orphans killed; row 13's artifact record (epoch 7 marked failed by the operator's container removal) removed so epoch 7 is re-run; kill logic fixed. Restarting batch 12-13.
04:14:36 EVENT: BATCH rows 12 13 START (control ab-test__swe-ci__control__rows-12-13__ep20)
04:14:47 EVENT: CONTROL row 13 START/RESUME (target :8001)
04:38:04 EVENT: PAUSED for maintenance: the external LACIE disk is ~1000x slower than the internal SSD for small files (100 file create+delete: 64 s vs 0.06 s); moving row 13's control run folder to the SSD, then resuming.
04:40:53 EVENT: maintenance done: row 13's run dir is now on the SSD (old HDD copy kept as runs/c13.hdd-old, per-epoch snapshot dirs linked; the snapshot of the operator-interrupted epoch 7 was left out). Restarting the driver.
04:40:54 EVENT: BATCH rows 12 13 START (control ab-test__swe-ci__control__rows-12-13__ep20)
04:41:11 EVENT: CONTROL row 13 START/RESUME (target :8001)
04:42:51 EVENT: driver restarted with a longer batch list (rows 12-41): epochs are now limited by the model, not the disk, so more batches fit before the deadline.
04:42:52 EVENT: BATCH rows 12 13 START (control ab-test__swe-ci__control__rows-12-13__ep20)
04:42:58 EVENT: CONTROL row 13 START/RESUME (target :8001)
05:09:03 EVENT: CONTROL rows 12 13 FINISHED in 26 min
05:09:12 EVENT: FORESIGHT rows 12 13 START (server up on :8010)
06:23:56 EVENT: WARN: foresight no task.log activity for 45+ min (stalled?)
07:31:47 EVENT: FORESIGHT rows 12-13 FINISHED in 142 min
07:31:47 EVENT: BATCH rows 12-13 DONE in 168 min; next estimate per 2 rows: 182 min
07:31:47 EVENT: BATCH rows 14 15 START (control ab-test__swe-ci__control__rows-14-15__ep20)
07:34:00 EVENT: PIPELINE STOPPED(network/VPN, resolved by resume): prep failed for rows 14 15 (see prep.log)
07:34:13 EVENT: BONUS: not run (pipeline halted)
07:34:59 EVENT: FINAL: experiments over; starting full re-collection with pylint
07:41:02 EVENT: RESUME-WAITER: servers unreachable since ~07:35 (VPN/network). Polling every 60 s; will restart after 3 consecutive readiness passes.
07:42:17 EVENT: COLLECTED rows 12-13 into foresight/results [no pylint] (see traces/collect_12-13.log)
08:11:38 EVENT: COLLECTED rows 12-13 into foresight/results [--score-jobs 2] (see traces/collect_12-13.log)
08:11:38 EVENT: FINAL: full re-collection finished (see events above for any COLLECT FAILED)
08:11:39 EVENT: FINAL: handoff written to foresight/results/LACIE
10:02:46 EVENT: RESUME-GATE: pass 1/3 - :8001 ok 0.2s prompt_tokens=7026 | :8002 ok 0.1s prompt_tokens=7026
10:03:03 EVENT: RESUME: VPN reconnected; the vLLM processes are the same as before (started 03:56:46 / 03:57:20), only the network was down. Starting the driver at once for rows 14 onward; deadlines unchanged (foresight until 15:07).
10:03:03 EVENT: BATCH rows 14 15 START (control ab-test__swe-ci__control__rows-14-15__ep20)
10:03:10 EVENT: CONTROL row 14 START/RESUME (target :8001)
10:04:41 EVENT: CONTROL row 15 START/RESUME (target :8001)
11:43:29 EVENT: CONTROL rows 14 15 FINISHED in 100 min
11:43:37 EVENT: FORESIGHT rows 14 15 START (server up on :8010)
14:03:36 EVENT: DEADLINE EXTENDED by operator: foresight cutoff 15:07 -> 15:40 (servers live until ~15:56; row 15's foresight epoch 1 hit a hung test, timeout ~14:37). Bonus control-only runs will not fit and are effectively skipped.
14:22:53 EVENT: WARN: foresight no task.log activity for 45+ min (stalled?)
15:40:27 EVENT: CUTOFF reached: stopping foresight rows 14-15 (truncated)
15:40:35 EVENT: FORESIGHT rows 14-15 TRUNCATED at cutoff; collecting partial results
15:40:35 EVENT: BATCH rows 14-15 DONE in 337 min; next estimate per 2 rows: 140 min
15:40:35 EVENT: PIPELINE DONE
15:40:51 EVENT: BONUS: not enough time left (9 min) for extra control runs
16:44:00 EVENT: COLLECTED rows 14-15 into foresight/results [no pylint] (see traces/collect_14-15.log)
```

### Event log, third driver (next server pair, from 2026-09-21 15:30)
```
15:30:14 EVENT: RUN4 GATE: waiting for the new servers: 3 consecutive readiness passes, and process start times NEWER than the previous jobs (03:56)
16:40:21 EVENT: RUN4 GATE: pass 1/3 - :8001 ok 0.4s prompt_tokens=7026 tool_call=ok | :8002 ok 0.3s prompt_tokens=7026 tool_call=ok
16:42:21 EVENT: RUN4 GATE: pass 2/3 - :8001 ok 0.2s prompt_tokens=7026 tool_call=ok | :8002 ok 0.1s prompt_tokens=7026 tool_call=ok
16:44:22 EVENT: RUN4 GATE: pass 3/3 - :8001 ok 0.2s prompt_tokens=7026 tool_call=ok | :8002 ok 0.1s prompt_tokens=7026 tool_call=ok
16:44:22 EVENT: RUN4: new servers ready (processes started 16:32:54 / 16:33:46); resuming rows 14-15 foresight, then rows 16+. Deadlines: foresight until 05:47, bonus until 06:12
16:44:23 EVENT: BATCH rows 14 15 START (control ab-test__swe-ci__control__rows-14-15__ep20)
16:44:38 EVENT: CONTROL rows 14 15 FINISHED in 0 min
16:44:51 EVENT: FORESIGHT rows 14 15 START (server up on :8010)
17:30:09 EVENT: FORESIGHT rows 14-15 FINISHED in 45 min
17:30:09 EVENT: BATCH rows 14-15 DONE in 45 min; next estimate per 2 rows: 49 min
17:30:09 EVENT: BATCH rows 16 17 START (control ab-test__swe-ci__control__rows-16-17__ep20)
17:31:26 EVENT: CONTROL row 16 START/RESUME (target :8001)
17:32:56 EVENT: CONTROL row 17 START/RESUME (target :8001)
18:43:34 EVENT: COLLECTED rows 14-15 into foresight/results [no pylint] (see traces/collect_14-15.log)
19:10:45 EVENT: CONTROL rows 16 17 FINISHED in 100 min
19:10:58 EVENT: FORESIGHT rows 16 17 START (server up on :8010)
20:42:20 EVENT: WARN: foresight no task.log activity for 45+ min (stalled?)
23:49:33 EVENT: FORESIGHT rows 16-17 FINISHED in 278 min
23:49:33 EVENT: BATCH rows 16-17 DONE in 379 min; next estimate per 2 rows: 409 min
23:49:33 EVENT: batch 18,19 trimmed to 1 row(s): only 358 min left
23:49:33 EVENT: BATCH rows 18 START (control ab-test__swe-ci__control__rows-18__ep20)
23:50:15 EVENT: CONTROL row 18 START/RESUME (target :8001)
00:45:25 EVENT: COLLECTED rows 16-17 into foresight/results [no pylint] (see traces/collect_16-17.log)
01:16:30 EVENT: CONTROL rows 18 FINISHED in 86 min
01:16:37 EVENT: FORESIGHT rows 18 START (server up on :8010)
02:07:42 EVENT: WARN: foresight no task.log activity for 45+ min (stalled?)
02:58:17 EVENT: FORESIGHT rows 18 FINISHED in 101 min
02:58:17 EVENT: BATCH rows 18 DONE in 188 min; next estimate per 2 rows: 313 min
02:58:17 EVENT: not enough time left (169 min) for batch 20,21; finishing
02:58:17 EVENT: PIPELINE DONE
02:58:25 EVENT: BONUS control-only rows 19 20 START (194 min until hard stop)
03:01:30 EVENT: COLLECTED rows 18 into foresight/results [no pylint] (see traces/collect_18.log)
```
