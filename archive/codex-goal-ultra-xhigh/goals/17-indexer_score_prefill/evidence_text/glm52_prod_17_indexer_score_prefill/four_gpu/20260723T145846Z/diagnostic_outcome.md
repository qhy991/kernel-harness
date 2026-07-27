# Four-GPU diagnostic outcome

This is an independent TP4/DP4 diagnostic, not the official TP8/DP8/EP8
production gate.

All three series failed correctness before timing. The broad balanced-chunk
candidate changed the row-wise selected-page set on rank 1 in series 1 and
on ranks 1 and 3 in series 2 and series 3. The reference stage completed on
every rank, and the collective failure handshake propagated each local
failure to all peers without an NCCL timeout.

No paired JSON or latency result exists because an incorrect candidate must
not enter the timing loop. The diagnostic strengthens the no-replacement
decision and does not authorize any four-rank or eight-rank promotion.
