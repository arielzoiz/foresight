# Notes for rows 19-20 (read with RUN.md)

- Row 19 (httpdbg 83ede4): 20 epochs, both arms. Row 20 (troposphere 14a8b3): control 20 epochs; **foresight arm truncated at the deadline (16 records, epoch 16), stopped
  because the 14 h GPU jobs were ending** (~06:33); a fresh run would need to resume it under a new job pair. `troposphere` foresight epoch 3 shows `x` then a gap spike to 86
  at epoch 3-4 in foresight vs a spike to 180 at epoch 4 in control -- both arms have a bad epoch there, not investigated.
- Hung tests: httpdbg has several `x` epochs in both arms (SWE-CI's 3600 s pytest timeout).
- Row 19's control ran on the internal SSD; row 20's control (bonus run) also SSD; the foresight run of both was on the SSD.
- Collected with `--no-pylint` (pylint pending). This is the last batch of the run (GPU jobs ended; no batch started after this one).
