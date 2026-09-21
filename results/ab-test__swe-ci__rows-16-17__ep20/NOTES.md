# Notes for rows 16-17 (read with RUN.md)

- Both rows (checkthechain 13f5d2, checkthechain 60ea1d) ran 20 epochs in both arms, from the internal SSD, on the second replacement pair of vLLM jobs (started 2026-09-21 16:33).
- **Row 16 foresight, epoch 8:** the programmer agent ran 60 min (about 900 model requests in 10 min: a long agent session, not a hang) until SWE-CI's 3600 s programmer limit
  ended it at 20:56; SWE-CI's automatic retry (attempt 2/3) then finished it in 1.5 min. Nothing was intervened in. This cost about one hour of wall-clock time.
- **Row 17 (60ea1d):** many epochs are `x` in both arms because the test run failed to start (`pytest was not executed correctly`, return code 4 or no report); the control arm shows the same.
  Foresight's epoch-4 architect step ran about 30 min (long agent session) before continuing.
- Aux runs: 85, all `sole_container`, no `body_only` fallback (see RUN.md).
- Collected with `--no-pylint` (pylint pending). Timing between arms is not comparable (see the handoff file).
