# DGX Spark chunked-prefill optimization — 2026-08-09

K-EXAONE-236B-A23B mixed-quant v1 on one GB10 (`sm_121`), ending at
[`Baekpica/ds4@b2faf06`](https://github.com/Baekpica/ds4/commit/b2faf06f1ce2702efa53ac17145b1f56d3fb23b8).
This round targeted the profiled kernels in descending order rather than
changing the scheduler or the 2,048-token chunk size.

## Result

Greedy, thinking disabled, concurrency 1, 128 generated tokens. Every frontier
uses a disjoint corpus slice, so a previous request cannot turn the next cell
into an incremental-prefix measurement. `Prefilled` excludes the tiny template
prefix already resident in the two short cells.

| Frontier | Prompt | Prefilled | Prefill | Decode | TTFT |
|---:|---:|---:|---:|---:|---:|
| 2K | 1,451 | 1,387 | **269.6 t/s** | 10.75 t/s | 5.14 s |
| 8K | 7,925 | 7,923 | **276.5 t/s** | 10.46 t/s | 28.66 s |
| 32K | 31,300 | 31,300 | **245.0 t/s** | 9.00 t/s | 127.78 s |
| 64K | 64,663 | 64,663 | **207.3 t/s** | 7.27 t/s | 311.90 s |

The prior published engine held roughly 54–56 t/s through 34K. The new 31.3K
cell is 4.5x faster than that published 34K result, while 64K is now directly
measured instead of projected. The 2K and 8K rows were remeasured on commit
`b2faf06` after the server warm-up. The 32K and 64K cells were run on the final
optimization candidate immediately before adding an invalid-token bounds guard
to the batch-embedding kernel; valid token IDs take the same numerical path.

Raw records:

- `scratch/prefill-opt/20260809/final-commit-short-warm/cells.jsonl`
- `scratch/prefill-opt/20260809/final-commit-matrix/cells.jsonl` (8K row; the
  first 2K row intentionally retains the excluded boot/page-warm cost)
- `scratch/prefill-opt/20260809/final-matrix/cells.jsonl`
- `scratch/prefill-opt/20260809/tail64-ab/{off,on}/bench/cells.jsonl`

## What changed

1. **IQ2 gate/up:** wide top-8 prefill now uses the existing paired aligned-SoA
   D2R tensor-core entry, then one weighted SwiGLU epilogue. Small batches keep
   the fused vector path.
2. **Q3/Q4 routed experts:** the caller supplies the safe maximum rows per
   expert. A GPU-built compact worklist enumerates only non-empty
   `(expert, column tile, output tile)` triples instead of walking the full
   expert-by-bound rectangle.
3. **Q3/Q4 tails:** the last 128-column tile uses the native 64-column MMQ tile
   when at most 64 rows remain. It stays in the same persistent kernel and
   worklist order.
4. **QK norm + RoPE:** one 128-thread block owns a token, preserves the existing
   RMS reduction order, and computes its 64 NeoX angles once instead of once
   per Q/K head.
5. **Token embeddings:** wide prefill uploads token IDs once and launches one
   Q8 embedding kernel instead of one launch per row. Invalid IDs are guarded
   before indexing the weight table.

Every optimization has a same-binary rollback:

| Path | Rollback |
|---|---|
| IQ2 wide D2R | `DS4_MMQ_D2R_IQ2=0` (or `DS4_MMQ_D2R=0`) |
| expert bound | `DS4_MMQ_EXPERT_BOUND=0` |
| compact Q3/Q4 worklist | `DS4_MMQ_WORKLIST=0` |
| 64-column worklist tail | `DS4_MMQ_WORKLIST_TAIL64=0` |
| token-block QK/RoPE | `DS4_EXAONE_QK_TOKEN_BLOCK=0` |
| batch embedding | `DS4_EXAONE_BATCH_EMBED=0` |

## A/B gates

- Tail64, same binary: 2K **241.4 -> 257.4 t/s** (+6.7%); 8K
  **263.4 -> 276.2 t/s** (+4.8%). Decode stayed within normal run variance.
- Q3/Q4 worklist tests use a fixed-seed, imbalanced top-8 router. Bounded and
  worklist outputs are exact against the generic path (`rel_rms=0`, `bad=0`).
- IQ2 wide D2R passes the integrated aligned-artifact parity gate and was
  1.23x faster than vector slicing in the final standalone run.
- A fixed 4,325-token request produced the same 64 token IDs, content hash,
  finish reason, and usage before and after the QK, embedding, and worklist
  changes.
- `tests/test_exaone_kernels` passes both synthetic fixtures and the real
  85.56 GiB three-shard model. `./ds4_test --server` also passes.

## Profile after the large moves

Nsight Systems was sliced to the actual final 11.6-second request window;
startup/repack kernels are excluded. This profile predates only the tail64
micro-optimization:

| Kernel group | GPU kernel time |
|---|---:|
| IQ2 aligned gate/up D2R | **30.66%** |
| Q3 routed down worklist | **23.65%** |
| dense/shared Q8 MMQ | **14.97%** |
| Q4 routed worklist | **10.77%** |
| batch token embedding | 4.90% |
| HMMA prefill attention | 2.62% |
| token-block QK norm + RoPE | 0.60% |

QK norm/RoPE moved from a major bottleneck to 0.6%. The next optimization
order is therefore still IQ2 D2R, Q3 down, dense Q8, then Q4; attention is no
longer the first place to spend effort. Evidence:
`scratch/prefill-opt/20260809/nsys-final-v2/final.nsys-rep` and
`final.sqlite`.

## Why the Spark Arena vLLM number is still higher

The cited [Spark Arena run](https://spark-arena.com/benchmark/c3980cfa-8700-49b7-ad17-d55c98fd88a4)
is useful as a ceiling, not an apples-to-apples runtime comparison. It serves a
different 180B DeepSeek-V4-Flash checkpoint with MXFP4 experts, FP8 MLA KV,
an 8,192-token batch budget, async scheduling, prefix caching, CUDA graphs,
compile/custom model modifications, and MTP. Its pp2048 result is about
1,233 t/s. K-EXAONE here is a 237B top-8 GQA model whose hot path streams
IQ2/Q3/Q4 GGUF kernels and Q8 dense/shared weights.

vLLM and SGLang combine scheduling with heavily specialized fused kernels,
graph capture/compilation, large token batches, and quant formats designed for
tensor cores. This ds4 round removed software waste in the same spirit, but the
remaining 4–5x gap at short context is now mostly the quantized MoE/dense
kernel stack shown in the profile, not chunk admission or attention.

## API and operational state

`GET /v1/models` now advertises:

- `k-exaone-236b-a23b` — thinking on by default;
- `k-exaone-236b-a23b-chat` — thinking disabled by the model alias.

The 1.60 GiB prefill graph workspace is shared across resident batched
sessions; it is not multiplied per session. Each slot still owns its KV and
small session state. MTP remains default-off because the measured draft path
is not a net decode win.
