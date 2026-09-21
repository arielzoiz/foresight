# Notes for rows 14-15 (read with RUN.md)

- **This folder is a partial result.** Row 14 (apprise ebbe26) is complete in both arms (20 epochs). Row 15 (pypeln 9121c6): the control arm ended after 2 epochs
  (gap reached 0, SWE-CI's normal exit condition); the **foresight arm was stopped at the time cutoff after epoch 2 (records `[18, x, x]`)** and is to be
  **resumed on new vLLM jobs**: SWE-CI resumes by experiment name, so epoch 3 onward will be added, and this folder re-collected.
- **Hung tests:** in the foresight arm, row 15's `pytest` hung in epochs 1 (recorded `x` by SWE-CI's 3600 s timeout) and again in epoch 3 (in progress when the run was
  stopped; that attempt has no record and is redone on resume). Epoch 2 is also `x`. In the control arm the same task ran its tests in about 1.5 min per epoch.
  Whether these hangs are related to the treatment is not established (row 12's foresight epoch 19 also hung, in a different project); none was intervened in.
- The foresight aux runs: 46, all `sole_container`, no `body_only` fallback; 44 are joined to an epoch (2 belong to the stopped, unrecorded epoch attempt).
- Row 14 foresight: the gap jumps from about 11 to about 130 at epoch 7 and stays high; not investigated.
- Both arms ran from the internal SSD (fast); collected with `--no-pylint` (pylint pending).
