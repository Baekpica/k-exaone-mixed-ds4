# K-EXAONE × ds4 DGX Spark — the re-measured matrix, and row-batched decode

Continues `DGX-SPARK-SERVING-OPT-2026-08-08.md` (same day, second half).  That
report fixed the sliding-ring defect and flagged that every number in
`DGX-SPARK-ITEMS-4-6` had been taken against the defective prefill.  This one
re-takes the matrix on the fixed engine, then builds the thing the concurrency
result said was missing.

Host: GB10 (`sm_121`), 121.6 GiB unified, driver 595.71.05.  Model: mixed-quant
v1.  ds4: `feature/exaone-model-loader`, ending the day at `2995e41`.

## Disposition

| Item | State |
|---|---|
| Depth models re-fit on the fixed engine | done — unchanged, previous timings stand |
| MTP acceptance re-measured with a correct verify | done — still a loss everywhere, near break-even at 33K |
| Concurrency re-measured | done — and the cause reattributed |
| Row-batched decode across sessions | **built, measured, shipped** (`2995e41`) |
| Q8 small-batch dispatch defect | found by nsys, fixed in the same commit |
| KV-store wrap race | found by the kernel suite, fixed in the same commit |

## 1. The timings survived the correctness fix

The ring defect changed *what* was computed, not *how much*.  Predicted, now
measured — cold sweep, disjoint frontiers, H-A2 vs the old A run:

| | old fit | new fit |
|---|---|---|
| decode ms/tok | 86.62 + 0.00597·ctx | 86.95 + 0.00588·ctx |
| prefill TTFT | 0.01766·N + 3.57e-7·N² | 0.01784·N + 3.47e-7·N² |
| max residual | 3.1 s | 0.5 s |

Marginal 2048-token chunk at depth 32K: 84.1 s before, 83.1 s after.  Every
per-cell number matched within noise (e.g. 8K: 47.9/7.38 → 47.9/7.30).  The
depth models in `DGX-SPARK-ITEMS-4-6` remain valid as *timings*; only the
quality-bearing results from that round are void.

### The harness had to relearn what "cold" means

The first re-run produced prefill numbers 1.7× too good.  Not a speedup: the
harness grows each stream's prompt from a fixed corpus offset, so frontier
F+1's prompt begins with frontier F's, and the engine — now able to resume a
diverged prefix — rewound past the generated tail and prefilled only the
delta.  Before partial reuse the same layout was accidentally cold, because
the all-or-nothing test always missed.

`--disjoint-frontiers` gives every frontier its own corpus slice and errors
out if the corpus cannot supply them.  The contaminated run is preserved as
`H-A-incremental-prefill-256k/` with a README, because what it measures is
real and useful — the cost of *continuing* a conversation:

| depth | incremental TTFT | cold TTFT |
|---:|---:|---:|
| ~8K | 94.8 s | 171.7 s |
| ~16K | 216.3 s | 387.3 s |
| ~33K | 586.4 s | 971.4 s |

## 2. MTP: still a loss, and the one number that transfers

With the fixed verify (the old one read one KV slot from a *future* position
past depth 128, so its accept/reject decisions were computed against a
corrupted argmax):

| depth | accept | MTP ms/tok | plain | k | needs a > | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 387 | 36.6 % | 147.0 | 95.1 | 2.111 | 1.111 | loss, 1.55× |
| 7 752 | 60.8 % | 148.8 | 132.5 | 1.805 | 0.805 | loss, 1.12× |
| 33 914 | 31.2 % | 297.7 | 286.4 | 1.364 | 0.364 | loss, 1.04× |

Two corrections to earlier narratives.  **Acceptance rates do not transfer
across rounds**: the disjoint-frontier corpus slices differ from the old
round's, and acceptance is a property of the text (36.6 % vs the old 69.3 % at
the same depth is corpus, not verify).  What does transfer is **k**, a timing
ratio: 1.364 at 33K in both rounds.  And the old fix list's "warm the MTP ring
toward the 69 % seen shallow" item is withdrawn — that 69 % was never a
comparable number.  The honest statement: at 33K the loss is 4 %, break-even
needs five points of acceptance, and the default-off + auto-quench disposition
stays right.

## 3. Concurrency: the cause was never MoE routing

`DGX-SPARK-ITEMS-4-6` measured flat aggregate decode (11.12 → 10.80 t/s from
1 to 8 streams) and attributed it "partly inherent to top-8-of-128 routing".
Reading the code falsified that: `ds4_sessions_eval_batch_exaone` was a
sequential loop — each session ran a full single-row decode, with one sync at
the end.  There was no row batching for routing to be the limit *of*.

### Row-batched decode (`2995e41`)

`exaone_graph_layer` now takes an optional row context: n_tok rows from n_tok
different sessions, one token each.  Weight-bound stages (projections, dense
and shared MLP, router, routed experts, LM head) run through the same batched
kernels prefill and the 2-row MTP verify already use; RoPE, KV store and
attention stay per-row against each session's own ring.  Prefill-scratch
borrowing is safe because the server admits no prefill while a decode is
pending — checked in `server_prefill_enter`, not assumed.

Two defects found under it, both fixed in the same commit:

- **Q8_0 small-batch dispatch.**  n_tok==1 read the resident aligned artifact;
  n_tok 2..8 fell through to raw-pointer mmq on the *unpinned model mapping*.
  nsys: `mul_mat_q<Q8_0>` averaged 2.4 ms against a 95 µs median — ATS
  first-touch on mmap pages.  The aligned vec entry documents N ∈ [1,8] as its
  envelope and reads the weight stream once per row for all columns; small
  batches now go there.
- **KV-store wrap race.**  One launch writing two tokens to the same ring slot
  raced on block order.  Unreachable in production since the ring resize, but
  the kernel is now deterministic under wrap (dead writes are skipped), and
  the kernel suite that caught it stays.

### Measured

Step cost by batch width (server log, `DS4_SERVER_BATCH_LOG=1`):

| width | ms/step | ms/token | vs width 1 |
|---:|---:|---:|---:|
| 1 | 85.3 | 85.3 | 1.00× |
| 2 | 157.3 | 78.6 | 1.84× |
| 4 | 304.6 | 76.2 | 3.57× |
| 8 | 591.7 | 74.0 | 6.94× |

Steady-state aggregate decode (`--batched-session 8 -c 4096`, all streams
decoding, prefill-contaminated windows excluded):

| conc | before | after | Δ |
|---:|---:|---:|---:|
| 1 | 11.12 | 11.49 | — |
| 2 | 9.80 | 12.09 | +23 % |
| 4 | 10.03 | 13.01 | +30 % |
| 8 | 10.80 | 12.22 | +13 % |

Concurrency now scales *up* from single-stream instead of dipping below it.
It is a real win and an honest fraction of the theoretical one.

### The numerics contract changed, deliberately

Batched rows take kernels whose accumulation order differs from the
single-row path (chiefly the MoE dedup kernel), so **greedy output across
batch widths is not bit-stable at near-ties**.  Measured with
`tests/test_exaone_decode_batch` on the real model, same session states, full
vocab: argmax agreed on 95/96 rows; the one flip sat at a 0.08 top-2 margin;
worst relative logits difference ~1e-1.  End-to-end, a 55-token probe against
a 7-stream background diverged at one genuinely ambiguous token and
re-converged ("옷자락을 휘날리며" vs "뺨을 스치며").  An indexing bug flips
argmax at whole-logit margins, and the test fails on any flip above 0.25 —
that is the line between the batched-inference contract every production
engine has and an actual defect.  Sequential decode (width 1) is unchanged.

### What still bounds it

Per-token step cost is nearly flat in width (~74 ms/token marginal), so the
aggregate ceiling at width 8 is ~13.5 t/s.  The remaining time is in the
routed-expert stages — 8 rows route to a mostly-disjoint union of experts
(inherent, roughly 6× the single-row routed traffic on random text), through
the aligned gate/up dedup kernel and the per-(row,slot) down leg (tunable,
upstream's kernel tier).  Attribution of that remainder is the top item on the
next round's list.  The steady-state Q8-dispatch delta was small on a warm
server — the mmap was already hot — so the dispatch fix matters most on the
first touches after boot, and for keeping small-batch reads off the ATS path.

## Order of work, revised

| # | Item | Why |
|---|---|---|
| 1 | Attribute the ~340 ms/step width-8 remainder (nsys, steady state) | it decides whether the next win is dedup-down, launch consolidation, or accepting the routed floor |
| 2 | MTP at depth with the fixed verify path | k fell to 1.36 at 33K; the 2-row verify now also rides the aligned vec path, so re-measure k before deciding anything |
| 3 | Rerun the deeper concurrency leg (16K ctx twin) | phase I covered the short-prompt twin only |

## Artifacts

| What | Path |
|---|---|
| cold matrix, fixed engine | `scratch/matrix/H-A2-plain-256k/`, `H-B2-mtp-noquench-256k/`, `H-C2b-batched8-c16k/`, `H-C2s-batched8-short/` |
| incremental-prefill record | `scratch/matrix/H-A-incremental-prefill-256k/` |
| row-batch acceptance | `scratch/matrix/I1-rowbatch-short/` (pre-dispatch-fix), `I3-rowbatch-final/` |
| engine-level A/B | `scratch/matrix/decode-batch-diag-w8*.log`, `tests/test_exaone_decode_batch.c` |
| nsys attribution | `scratch/matrix/rowbatch-w8.nsys-rep`, `rowbatch-w8-kernels.txt` |
| harnesses | `benchmarks/rowbatch_identity.py`, `scratch/matrix/matrix_bench.py` (`--disjoint-frontiers`) |
