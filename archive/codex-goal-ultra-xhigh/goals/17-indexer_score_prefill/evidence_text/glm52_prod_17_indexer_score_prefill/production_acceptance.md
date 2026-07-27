# Production acceptance status

## Official gate

The production acceptance gate remains:

- one node;
- eight ranks;
- TP8/DP8/EP8;
- real `nvidia/GLM-5.2-NVFP4` model weights and tokenizer;
- normal SGLang prefill scheduling with the selected SM100 TRT-LLM DSA
  backend;
- replacement dispatch disabled for baseline and enabled only for an eligible
  candidate bucket;
- end-to-end prefill metric plus PCG/BCG split-region correctness.

No four-rank or standalone ABI result is relabeled as this gate.

## Local external-state blocker

The only local Hugging Face snapshot is
`aec7243e916f585f4d52b97e4530f9a9750b0648`, totaling 36 KiB. It contains
`config.json` and small metadata blobs only:

- no `*.safetensors`;
- no safetensors index;
- no tokenizer files.

The host exposes four schedulable B200s through the required lock wrappers,
not eight. Therefore a real eight-rank SGLang model-server launch cannot be
performed in this environment. Downloading hundreds of gigabytes of model
weights or weakening the rank count is outside this goal's authority.

The independent four-GPU diagnostic was delayed by scheduler exit 75 responses
until the global lock became available; no wrapper was bypassed. Its first
locked attempt exposed a rank-divergent correctness exception that the old
runner masked as an NCCL timeout. After hardening untimed distributed failure
handling, the fresh locked attempt completed three series: every series
rejected the broad candidate before timing for row-wise top-k set mismatches
on rank 1, with rank 3 also failing twice.

No four-rank latency result exists because the candidate was incorrect, and
the diagnostic is not relabeled as the official gate.

Stock SGLang remains the fallback unless and until the unchanged official
gate is run on an eight-B200 host with the complete model artifact.

## Promotion consequence

Even if a rank-local score or complete-indexer bucket exceeds the 3% paired
gate, it is classified as an unpromoted candidate until the official server
gate passes. A no-replacement disposition leaves no runtime dispatch enabled.
