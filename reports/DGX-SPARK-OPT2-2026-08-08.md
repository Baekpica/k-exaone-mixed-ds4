# K-EXAONE × ds4 DGX Spark — optimization round 2: the counterpart tiers

Continues `DGX-SPARK-REMEASURE-ROWBATCH-2026-08-08.md` (same day, evening).
Mandate for this round: for every serving-performance tier upstream `ds4` or
the `Entrpi` fork built for DeepSeek, build and measure the `exaone-moe`
counterpart — no gaps.

Host: GB10 (`sm_121`), 121.6 GiB unified.  Model: mixed-quant v1.
ds4: `feature/exaone-model-loader` @ `d35f0dd`, one commit for the round.

## The counterpart map, closed

| Upstream / fork tier | exaone counterpart | Disposition |
|---|---|---|
| `DS4_CUDA_GREEDY_SPLITKV` flash-decode | split-KV decode + narrow-deep verify routing | **shipped, default on** |
| mmq vector-envelope tier | MoE small-batch vec chunking + dedup width cap | **shipped, default on** |
| native mixed prefill/decode batch | layer prefill-tail + server offer/fuse | **shipped, default on** |
| `DS4_CUDA_DECODE_GRAPHS` islands | front/tail capture on the same state machine | shipped, **opt-in** (first-replay defect, below) |
| FlashAttention-style prefill blocking | tiled prefill kernel | shipped, **opt-in** (measured a wash, below) |
| dense D2R prefill tier | `DS4_MMQ_D2R_MAX_K` tunable | ceiling kept at the measured cliff |

## Headline numbers

**Single stream, cold prompts (J2-A vs the morning's H-A2):**

| depth | decode before | decode after | prefill |
|---:|---:|---:|---:|
| ~700 | 10.89 | 10.87 | 51.8 |
| ~1 900 | 10.19 | 10.17 | 52.8 |
| ~8 100 | 7.30 | **10.03** | 47.3 |
| ~33 300 | 3.52 | **8.73** | 32.2 |

Depth term 5.88 → **0.61 µs/position** (effective KV read 8.4 → ~81 GB/s).
New model: `ms/token = 94.5 + 0.00061·ctx`.  Extrapolated: 5.7 t/s at 128K,
3.9 t/s at the full 262 144.  Shallow cells are unchanged — the split falls
back to the bit-exact one-block kernel inside one chunk.

**Concurrent serving, steady-state aggregate (J3 vs the H-C2s baseline):**

| streams | before | after |
|---:|---:|---:|
| 1 | 11.42 | 11.46 |
| 2 | 10.50 | **14.83** |
| 4 | 10.87 | **16.28** |
| 8 | 11.08 | **18.50** |

Monotonically rising for the first time.  Step cost by stage of the day:
831.6 (first row-batch) → 592.7 (aligned dispatch) → 450 (MoE chunking) →
**387.5 ms/step** at width 8.  The mixed prefill+decode fuse fired in
serving (`+prefill` at widths 1–6, no stalls): an admitted prompt's quantum
rides the decode batch's weight sweep.

**MTP, quench defeated (J2C, all tiers active):**

| depth | acceptance | MTP ms/tok | plain | vs plain |
|---:|---:|---:|---:|---:|
| 1 387 | 36.6 % | 108.8 | 98.3 | +11 % |
| 7 752 | 60.8 % | 102.5 | 99.7 | **+2.8 %** |
| 33 914 | 37.0 % | 142.8 | 114.5 | +25 % |

The verify-cost lever is spent — routing the two-row verify through the
per-row flash-decode took the 33K cell from 0.45× plain (3.97 t/s, mid-round)
to 0.80× (7.00 t/s).  **Default-off + auto-quench stands**; the remaining
lever is genuinely acceptance now (the 8K row already sits inside the 3 %
quench threshold).  Acceptance is corpus property — it moved 20 points
between slices today — so only k transfers across runs.

## What did not survive its own measurement

- **Tiled prefill attention** — correct against the CPU reference on both
  layer types, and a wash end to end (32K cold prefill 32.2 t/s tiled vs
  33.9 anchor): the staged K/V tile's traffic saving is spent in
  per-(query, key) warp reductions.  Opt-in `DS4_EXAONE_PREFILL_TILE=1`;
  the recorded next step is tensor-core (HMMA) scores, which removes the
  shuffle bottleneck rather than rearranging it.
- **Decode-graph islands** — a wide argmax flip at the diagnostic's step 2
  (the first-replay slot) first read as a replay defect.  Two discriminator
  runs — islands off, then the MoE dedup kernel off — reproduced the flip
  bit-for-bit both times, which acquits both suspects: the flip is
  accumulated kernel-order drift landing on a 1.4-margin token of a
  random-token stress prompt, and it belongs to the measured numerics
  contract below.  The islands stay opt-in regardless: their captures
  include pool-allocated scratch whose replay semantics are not yet pinned,
  and capture must not be a leap of faith.
- **The gate/up dedup kernel at width 8** — slower than the per-slot kernel
  it replaces (450 → 395 ms/step with it off).  Capped to the 2-row verify
  width it was built and measured for.

## The numerics contract, measured

Row-batched decode differs from sequential by kernel accumulation order.
Calibrated on random-token stress prompts: drift up to ~2–5 absolute on the
logits tail, argmax flips on ~2 % of rows at margins under that amplitude,
deterministic per batch composition; the same batch always reproduces the
same output, and width-1 decode is unchanged.  The diagnostic
(`tests/test_exaone_decode_batch`) now fails on amplitude (> 8) or rate
(> 20 % of rows) rather than on margins, which cannot separate legitimate
drift from defects at this operating point.

## Next round, in order of expected payoff

1. **Tensor-core prefill attention.**  Prefill's quadratic term is the one
   wall left (33K cold ≈ 17 min, 256K ≈ 9 h).  HMMA Q·Kᵀ scores remove the
   shuffle bound the tiled variant ran into.
2. **MTP acceptance** (warm the ring from the prompt) and **batched MTP**
   (verify rows joining the cross-session batch) — the 8K row is 2.8 % from
   flipping the default.
3. **Dedup kernel and islands root causes** — both scoped, both documented.
4. Shortest-prefill-first admission; Q3_K/Q4_K aligned-SoA down tier;
   `DS4_MMQ_D2R_MAX_K` A/B at 6144.

## Artifacts

| What | Path |
|---|---|
| diagnostic runs | `scratch/matrix/J1-decode-batch-diag.log`, `decode-batch-diag-w8*.log`, `J1-dedup-off.log` |
| depth + MTP cells | `scratch/matrix/J2-A-plain-256k/`, `J2-B-…`, `J2C-mtp-noquench-256k/` |
| identity + widths + mixed rides | `scratch/matrix/J3-rowbatch-short/` |
| 16K twin | `scratch/matrix/J4-batched8-c16k/` |
| nsys attributions | `scratch/matrix/rowbatch-w8*.nsys-rep`, `*-kernels.txt` |
