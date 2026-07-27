# Goal04 page64 V32 coordinate experiment

This run tests one reached production-kernel change against the stock
`flashmla_kv` ABI: for V32/original KV pages, physical token indices are already
the flattened 656-byte TMA coordinates. The candidate replaces runtime page
division, remainder, and address reconstruction with the equivalent direct
coordinate and scale-byte offset. Generic pages, extra KV, invalid-index
masking, and all other backends remain stock/fallback.

The source checkout is the pinned FlashMLA commit used by SGLang, with local
experiment commit `5fa2b1f63aa74a72f2db0e3797ee0ffa867d38cd`. The stock extension
is never overwritten; the candidate is loaded in an isolated Torch operator
namespace and paired with stock in one process.
