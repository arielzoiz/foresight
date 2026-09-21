# Notes for rows 14-15 (read with RUN.md)

- **Row 14 (apprise ebbe26)** ran 20 epochs in both arms. Foresight: the gap jumps from about 11 to about 130 at epoch 7 and stays high (not investigated).
- **Row 15 (pypeln 9121c6): control ended after 2 epochs** (gap reached 0, SWE-CI's normal exit condition).
- **Row 15 foresight ran in two parts.** Epochs 1-2 ran on the first replacement vLLM jobs; the run was stopped at the time cutoff (epoch 3 was in progress and had no
  record) and **resumed on new identical vLLM jobs** (SWE-CI resumes by experiment name). Epoch 3 was therefore re-run from the same code state and the run then finished
  (gap 0 in epoch 17). The resumed part used the same checkpoint, launch flags, URLs and settings; the new jobs' process start times are in the handoff file.
- **Hung tests (foresight arm only so far):** in row 15's foresight run, `pytest` hung in epoch 1 (recorded `x` by SWE-CI's 3600 s timeout) and again in the epoch-3 attempt that
  was stopped and redone; epoch 2 is also `x`. In the control arm the same task ran its tests in about 1.5 min per epoch. Whether the hangs are related to the treatment is not
  established (row 12's foresight epoch 19 also hung, in a different project); none was intervened in.
- Aux runs and fallbacks: see RUN.md (no `body_only` fallback expected; check the line there).
- Both arms ran from the internal SSD (fast). Collected with `--no-pylint`; pylint pending.
