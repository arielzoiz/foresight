# Notes for row 22 (read with RUN.md)

- **Partial result.** Control (pdfsyntax 8fa6b3) is complete in both arms up to this point: control 20/20 epochs, **foresight arm truncated at
  epoch 12/20 (records `[46, 32, 17, 13, 10, 10, 10, 15, 15, 17, 26, 19, 13]`)**, stopped by the 4h GPU job budget ending. SWE-CI resumes by
  experiment name, so it will continue from epoch 12 on the next vLLM job pair and this folder will be re-collected.
- Control 22 ran concurrently with control 21 (on :8002, model_name=aux-model - see row 21's NOTES.md for why this differs from other rows).
- 26 aux runs, no `body_only` fallback; 24 joined to an epoch (2 belong to the in-flight epoch 13 attempt, unrecorded on truncation).
- Collected with `--no-pylint` (this and every other batch's initial pass); pylint to be added once complete.
