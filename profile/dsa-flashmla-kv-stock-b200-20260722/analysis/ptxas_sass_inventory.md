# Final FlashMLA `ptxas` and native-code inventory

This is a CPU-only static audit of the final `stock-pybind-tensor` and
`combine32-m16-tensor` build artifacts. No CUDA context was initialized and no
kernel was loaded or executed.

## Artifact identity

| Build | FlashMLA commit | extension SHA-256 | wheel SHA-256 | GNU build ID |
| --- | --- | --- | --- | --- |
| `stock-pybind-tensor` | `0657fffdfd1c981517647e043e4ef30ffdc1480f` | `b1afc29425c79cf00ad9687636474bfb7ffc098d81c5013ad1f3ade1966342f9` | `1b762bbcdbdbf1c5a6322325563bb95c0b456c2df80006e799f393af9f2b45b0` | `732a6f6986059798c962769cbb23624b234fff18` |
| `combine32-m16-tensor` | `d18ff63a73dc6519432f59acb9f04365ce14bb10` | `9665dec00cb8caa4a8b5fc42bd40f9e8320d890e1a77705f0428630010539ccb` | `d034eb53a849e08516ca90373f1f849306d60f7ac21c24038c6d4f1b52055738` | `847b4f0a2e2ff7eaa6cd2d70fe01888a61c8e442` |

The two manifests are [stock](build_stock_pybind_tensor.json) and
[candidate](build_combine32_m16_tensor.json). Both record CUTLASS commit
`147f5673d0c1c3dcf66f78d677fd647e4a020219`, clean tracked FlashMLA/CUTLASS
trees, and the same SM100 compile policy. In particular, both compile lines use
`-gencode arch=compute_100f,code=sm_100f`, `-lineinfo`, `--source-in-ptx`, and
`--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,...`:

- sparse V32 main: [stock log line 31](../reports/build_stock_pybind_tensor.log#L31) and [candidate log line 31](../reports/build_combine32_m16_tensor.log#L31)
- combine translation unit: [stock log line 417](../reports/build_stock_pybind_tensor.log#L417) and [candidate log line 410](../reports/build_combine32_m16_tensor.log#L410)

## `ptxas` resource comparison

The final logs report identical resources for corresponding entry functions.
The exact production sparse main is SM100 head64 `ModelType::V32`/ModelType0.
The two combine rows below are the relevant BF16 specializations in the same
`combine.sm_100.cubin`.

| Entry function | Stock citation | Candidate citation | registers | barriers | stack | spill stores / loads | `ptxas` static smem | `cuobjdump` `SHARED` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse main, SM100 head64 V32 | [lines 33-36](../reports/build_stock_pybind_tensor.log#L33-L36) | [lines 33-36](../reports/build_combine32_m16_tensor.log#L33-L36) | 168 | 16 | 0 B | 0 B / 0 B | not printed | 1,024 B |
| combine BF16, bound 160 | [lines 454-457](../reports/build_stock_pybind_tensor.log#L454-L457) | [lines 447-450](../reports/build_combine32_m16_tensor.log#L447-L450) | 48 | 0 | 24 B | 0 B / 0 B | 5,120 B | 6,144 B |
| combine BF16, bound 32 | [lines 474-477](../reports/build_stock_pybind_tensor.log#L474-L477) | [lines 467-470](../reports/build_combine32_m16_tensor.log#L467-L470) | 48 | 0 | 24 B | 0 B / 0 B | 1,024 B | 2,048 B |

The complete combine specialization ladder is present in **both** builds for
both FP16 and BF16: bounds `32, 64, 96, 128, 160`. Every specialization uses 48
registers, zero compiler-reported spills, and a 24-byte stack frame. Its
`ptxas` static-smem ladder is respectively `1,024, 2,048, 3,072, 4,096, 5,120`
bytes. See [stock lines 429-477](../reports/build_stock_pybind_tensor.log#L429-L477)
and [candidate lines 422-470](../reports/build_combine32_m16_tensor.log#L422-L470).

`cuobjdump --dump-resource-usage` accounts for a fixed additional 1,024 bytes
of `SHARED` for these combine entries, so its figures must not be mixed with
the `ptxas` static-smem column. It reports `STACK:24 LOCAL:0` for combine-32 and
combine-160. The SASS contains three `STL` records in each combine function,
consistent with the 24-byte call/stack path, but `ptxas` explicitly classifies
zero bytes as register spills.

The relevant source delta is therefore a dispatch change, not a new combine
kernel body. The [candidate source patch](../../../../sglang/third_party/FlashMLA-goal22/build-artifacts/combine32-m16-tensor/source.patch)
adds `CombineParams.max_num_splits`, sets it to 32 only for the exact GLM-5.2
M16 predicate, and switches the existing `MLA_NUM_SPLITS_SWITCH` on that field.
All other ABIs retain `num_sm_parts`. For the exact predicate (`num_sm_parts ==
148`), this selects the already-present bound-32 specialization instead of the
stock bound-160 specialization. The dispatched static shared-memory allocation
therefore falls from 5 KiB to 1 KiB in `ptxas` accounting (6 KiB to 2 KiB in
`cuobjdump` accounting), while the sparse main is unchanged.

## Native symbol and SASS inventory

`cuobjdump --list-elf` reports the same 23 embedded ELFs in each extension, all
targeting `sm_100`. The production-relevant entries are:

- ELF 1, `model1.sm_100.cubin`: SM100 head64 ModelType1 sparse main;
- ELF 2, `v32.sm_100.cubin`: exact SM100 head64 V32/ModelType0 sparse main;
- ELF 22, `combine.sm_100.cubin`: FP16 and BF16 combine specializations;
- ELF 23, `get_decoding_sched_meta.sm_100.cubin`: decode scheduler metadata.

The SASS inventory contains two SM100 head64 sparse-main entry functions
(ModelType0 and ModelType1) and ten combine entry functions (two dtypes times
five split bounds) in **each** build. `nm -D --defined-only | c++filt` also finds
the same relevant host launch/control symbols in both extensions:

- `Decode_Sm100_Head64_Impl::run_` and `Decode_Sm100_Head64_Impl::get_meta`;
- `sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType0>`;
- `sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType1>`;
- `smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>`;
- `smxx::decode::run_flash_mla_combine_kernel<cutlass::half_t>`.

The complete textual output of `cuobjdump --dump-sass` has the same SHA-256 for
both extensions:

```text
1876bb2ba79f61c88ee326825d6b0e1372006e85276e065184b0da13a2f01c53
```

That output includes the encoded instruction and control words, so the decoded
executable SASS is identical across the two builds. The complete
`--dump-resource-usage` output is also identical, with SHA-256:

```text
d947be5b47203e26272b6e5805b568849458584eadad886cfc5b356257a21122
```

Selected static SASS instruction inventories (identical in stock and
candidate) are:

| Entry function | instruction records | loads/stores (`LDG/LDS/STG/STL/STS`) | synchronization | Blackwell tensor/TMA records |
| --- | ---: | --- | --- | --- |
| sparse main, SM100 head64 V32 | 4,128 | `28 / 83 / 2 / 0 / 119` | `BAR 21`, `DEPBAR 3`, `BSSY 10`, `BSYNC 10` | `UTCHMMA 26`, `UTMALDG 57`, `UTMASTG 8` |
| combine BF16, bound 32 | 280 | `12 / 1 / 5 / 3 / 1` | `BSSY 1`, `BSYNC 1` | none |
| combine BF16, bound 160 | 344 | `16 / 1 / 5 / 3 / 5` | `BSSY 1`, `BSYNC 1` | none |

These are static disassembly-record counts, not dynamic executed-instruction
counts. Selecting combine-32 rather than combine-160 removes 64 static SASS
records from the dispatched combine specialization, including four `LDG` and
four `STS` records; it does not rewrite either specialization.

## ELF/container distinction

The extension-level difference is visible in host ELF sections:

| Section / identity | Stock | Candidate | Delta |
| --- | ---: | ---: | ---: |
| `.text` size | `0x0c48a2` | `0x0c4c42` | `+0x3a0` (+928 B) |
| `.dynsym` size | `0x004008` | `0x004050` | `+0x48` (+72 B) |
| `.nv_fatbin` size | `0x1332e08` (20,131,336 B) | `0x1332e08` (20,131,336 B) | 0 B |
| `.nv_fatbin` SHA-256 | `bb10d1e2dbb194fe91e25f5ad2aff0b72c0ce3d8b22938487dca214b40bea646` | `0bc2aa08f6a1579de78f24827ba82abcb13de24f69afbe8d1a574efdf8b1846a` | different |

The raw fatbin containers are therefore **not** byte-identical even though
their decoded SASS and resource-usage output are identical. Both builds embed
line information and source-in-PTX, and the audited source/header line metadata
differs. This audit did not attempt to attribute every non-SASS cubin metadata
byte; it makes the narrower, directly verified claim about executable SASS and
resources.

## Reproduction and limitations

The audit used only CPU-side readers:

```bash
/usr/local/cuda/bin/cuobjdump --list-elf "$EXTENSION"
/usr/local/cuda/bin/cuobjdump --dump-resource-usage "$EXTENSION"
/usr/local/cuda/bin/cuobjdump --dump-sass "$EXTENSION"
nm -D --defined-only "$EXTENSION" | c++filt
readelf -n "$EXTENSION"
readelf -S -W "$EXTENSION"
```

This evidence does not replace an import/ABI test, dynamic profiling, occupancy
measurement, or end-to-end acceptance. Compile-time resource reports and static
instruction counts do not establish latency. They establish that the candidate
ships the expected symbols and existing SM100 specializations, leaves the exact
sparse-main executable unchanged, and realizes its device-side resource change
by selecting combine-32 at host dispatch.
