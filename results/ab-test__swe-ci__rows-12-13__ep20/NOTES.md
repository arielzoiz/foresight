# Notes for rows 12-13 (read with RUN.md)

- **Row 13 control (hera 7ac8d5) ran in two parts.** Epochs 0-6 ran on the first vLLM jobs; the run was stopped (bonus time limit) and resumed
  after the servers were replaced by identical jobs (same checkpoint, launch flags and URLs). SWE-CI resumes by experiment name. The stop had recorded epoch 7 as a failed
  epoch (`gap = -1`) only because the operator removed the running containers; that artifact record (and its snapshot) was removed so epoch 7 is re-run from the same state.
  Epoch 7 onward ran on the new servers, from the internal SSD. This is the only run in which the control arm was interrupted.
- **Row 12 foresight (claude-agent-sdk-python 91315e), epoch 19:** a test in that project hung (a helper process waiting for input) and SWE-CI's
  3600 s pytest timeout ended it, so the epoch is recorded as `x`. The run then continued normally. Nothing was intervened in.
- Rows 12 and 13 controls: row 12 ran on the external disk, row 13's resumed part on the SSD; timing between arms is not comparable (see the handoff file).
- Collected with `--no-pylint` (maintainability index only); a pylint re-collection is pending.
