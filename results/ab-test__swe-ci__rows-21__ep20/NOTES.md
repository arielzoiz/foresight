# Notes for row 21 (read with RUN.md)

- Single-row batch, complete in both arms, 20 epochs, on the third replacement vLLM job pair (started 2026-09-22 09:43).
- Control ran on :8001, foresight's aux ran on :8002 (aux-model). This is the first control run in this project prepared with a
  concurrent sibling (row 22 ran control at the same time, on :8002 with model_name=aux-model instead of target-model) - a settings
  difference from every other control run so far, noted for comparability; not expected to affect the target model's behavior since
  it is the same checkpoint under a different served name.
- Both arms hit one bad epoch each (control epoch 18 spikes to gap 81; foresight has `x` at epochs 1 and 17). Not investigated.
- 40 aux runs, no `body_only` fallback. Collected with pylint scoring.
