# Superseded score campaign

The score campaign under `runs/20260723T112717Z/` is retained as historical
evidence, but it is not used for the final performance disposition.

Its paired correctness runs completed, but the collection bundle ended with
`failures=1`. The Nsys command used an incompatible capture trigger, the NCU
command used the wrong output option, and the first CUDA-graph timing loop did
not balance reference-first and candidate-first captures. Consequently its
profiler commands failed and its graph ratios may contain order bias. The raw
status, logs, paired JSON, and artifact manifest remain intact.

Commits `9029b373810934206a0d0192e36efa278c800fa8` and
`6c86a8de3138dfdf883a5c47924f4fc1d0862abb` repaired the capture order and
profiler commands and added a command smoke test. The replacement campaign is
`runs/20260723T113910Z/`, with profiler artifacts in
`profile/indexer-score-decode-20260723T113910Z/`. It reports `failures=0`, and
its artifact manifest verifies. Only that corrected campaign is
decision-bearing.
