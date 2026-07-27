# Superseded four-GPU diagnostic

Series 1 entered an NCCL barrier timeout after a rank-local candidate
correctness failure diverged control flow. The timeout log is preserved
unchanged. Series 2 was manually interrupted after reproducing the same
divergence pattern; the four orphaned worker processes from this goal were
terminated by exact PID and no artifact was deleted.

This directory contains no performance or acceptance result. The follow-up
runner change made untimed correctness failures collective so every rank
fails promptly and the original rank-local error is visible. The fresh
`20260723T145846Z` diagnostic supersedes this attempt.
