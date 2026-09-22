# Notes for row 18 (read with RUN.md)

- Single-row batch (httpdbg 22e489), 20 epochs, both arms, on the second replacement vLLM pair (started 2026-09-21 16:33). The gap is 3 from the start and stays 3 in both arms.
- **Hung tests in both arms:** SWE-CI's 3600 s pytest timeout ended one epoch in the control arm (epoch 5) and one in the foresight arm (epoch 2), each recorded `x`; foresight also shows an `x` in
  epoch 19. Nothing was intervened in. Each hang costs about an hour of wall-clock time.
- The control arm ran from the internal SSD; the foresight run's working folder (`runs/f18`) was on the external disk (not linked to the SSD), so its timing is less comparable.
- Aux runs: 40, no `body_only` fallback. Collected with `--no-pylint` (pylint pending).
