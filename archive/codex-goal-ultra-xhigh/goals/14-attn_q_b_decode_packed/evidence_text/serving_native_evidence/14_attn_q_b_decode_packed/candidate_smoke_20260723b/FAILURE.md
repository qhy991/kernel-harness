# Candidate smoke attempt b

The CPU routing tests passed. Candidate import then entered SGLang's broad
16,384-shape DeepGEMM precompile warmup because this smoke script had not pinned
the serving precompile controls. The run was interrupted after 33 warmup items,
before candidate correctness or timing. Attempt c disables broad precompile and
retains only the two on-demand q_b JIT specializations.
