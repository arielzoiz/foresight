# Notes for rows 19-20 (read with RUN.md)

- **Now complete in both arms, 20 epochs.** Row 20's foresight arm was earlier truncated at epoch 16 by a GPU job boundary; it resumed on the next
  vLLM job pair (started 2026-09-22 09:43) and finished cleanly (troposphere 14a8b3 foresight EvoScore 0.4598, resolved). SWE-CI resumed by experiment
  name; epochs 0-15 ran on the prior job pair, 16-20 on this one, same checkpoint/settings.
- Row 19 (httpdbg 83ede4): several `x` epochs in both arms (SWE-CI's 3600s pytest timeout / collection errors), consistent with other rows.
- Row 20 (troposphere 14a8b3): control epoch 4 spikes to gap 180 then recovers to ~1; foresight's epoch 3 shows `x` then a spike to 86 at epoch 3-4 -
  both arms hit a bad epoch at roughly the same point, not investigated further.
- 83 aux runs, all resolved by `sole_container`, no `body_only` fallback (see RUN.md). Collected with pylint scoring.
