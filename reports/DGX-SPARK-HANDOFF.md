# DGX Spark (`sm_121`) Handoff

Everything below was produced on the RTX PRO 6000 (`sm_120`) development host.
None of it has been validated on GB10. This document says what is done, what is
known, and what the `sm_121` stage has to do first.

## What ships

| Item | Where |
|---|---|
| Mixed-quant artifacts + provenance | `Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF` (public) |
| Converter, recipe, verifier, benchmarks | `k-exaone-mixed-ds4` |
| ds4 K-EXAONE loader | `Baekpica/ds4`, branch `feature/exaone-model-loader` |
| Reference fixtures + expected outputs | `fixtures/`, `benchmarks/results/` |
| This bundle | private HF bucket |

## Pinned revisions

| Component | Pin |
|---|---|
| K-EXAONE safetensors | `61e6d578eb102b578e5704e2916ac841df9eca0a` |
| K-EXAONE GGUF | `5bd0394e4f42c00df63e207b9c434387523a6b77` |
| BF16 GGUF sha256 | `73be2da8653976df036bf9b6466b011f86cb10f78bab30a47025638ec999d3f8` — **verified locally, matches** |
| llama.cpp | `6a32c29a746a2e44de463de647f9f6661eb5086b` (`b10295`) |
| ds4 upstream base | `b0309611041655f4e45671cfd9c9886aff161406` |

## Artifacts

| Variant | Size | sha256 | Verified against recipe |
|---|---:|---|---|
| pilot (`Q2_K` experts, no imatrix) | 87.84 GiB | `2d840ee44b0e10cb2e14ec7cf58d2e7849615de1a92f58b1220790f42310ce39` | yes, clean |
| v1 (`IQ2_XXS` gate/up, `Q3_K` down) | 85.5 GiB projected | see `*.manifest.json` | see `verify-v1.json` |

Both preserve all 128 routed experts, the shared expert, and the 1-layer MTP
block at `blk.48` (15 tensors, `Q8_0`).

## Memory budget — the first thing to check on GB10

Sizing targets a single 128 GB unified-memory device but **has not been measured
there**. What is known:

- Artifact on disk: 87.84 GiB (pilot) / ~85.5 GiB (v1).
- That leaves roughly 40 GiB of the 128 GB for the runtime workspace, logits,
  CUDA graphs, KV cache, MTP state, and the operator safety floor — before
  accounting for whatever the OS and driver hold.
- KV cache is not small: 48 layers, 8 KV heads, head_dim 128. Full-attention
  layers (every 4th — indices 3, 7, … 47, so 12 of them) hold the whole context;
  the other 36 are windowed at 128 tokens. At f16 that is
  `12 × 2 × 8 × 128 × 2 bytes = 48 KiB per token` for the full layers plus a
  fixed `36 × 2 × 8 × 128 × 128 × 2 bytes = 18 MiB` for the sliding ones.
  8 K context ≈ 0.4 GiB; 128 K ≈ 6 GiB. The LLLG schedule is what makes long
  context affordable here — measure it, do not assume it.

**Do this first:** load the pilot artifact and record actual resident unified
memory before anything else. If it does not fit with a safety floor, the size
ladder in `quant-recipe-v1.yaml` (`size_reduction_ladder`) is the intended
response — embeddings and output to `Q6_K` first, never the router or norms.

## ds4 state — read before planning kernel work

`feature/exaone-model-loader` contains **Phase D2-1 only**: architecture
detection, hparam validation, tensor binder, layout validation. It compiles
clean. **It cannot run inference.** What is missing, in order:

1. **GQA attention.** This is the big one. All four existing ds4 shapes set
   `n_head_kv = 1` and use MLA with a sparse DSA indexer. K-EXAONE is plain GQA
   — 64 query heads over 8 KV heads at head_dim 128, with per-head RMSNorm on Q
   and K before RoPE. No GQA path exists in ds4, reference or CUDA. Neither the
   attention kernels nor the KV cache layout can be reused from DeepSeek4/GLM.
2. **LLLG KV cache.** Sliding layers need only the last 128 tokens; full layers
   need everything. The two must be separate cache types, correctly forked and
   copied per session.
3. **MoE forward.** This part reuses well — `n_ff_exp` 2048, one shared expert
   and `expert_weight_scale` 2.5 are identical to GLM 5.2. Router differences to
   respect: sigmoid gating (`expert_gating_func` 2), top-8, normalized top-k
   probabilities, and a per-layer `exp_probs_b` score-correction bias.
4. **The MTP block is dense.** `blk.48` has `ffn_gate/up/down` at 18432, not
   expert tensors — the binder already handles this, the forward path must too.
   It has no embedding or LM head of its own; it shares the base model's.
5. **`Q3_K` CPU reference dequant.** `DS4_TENSOR_Q3_K` is now declared and
   `cuda/mmq/` already vendors llama.cpp's Q3_K kernel, but ds4's CPU reference
   path has no Q3_K block struct. v1 needs it for routed expert down.

## Why `Q3_K` for routed expert down

The work order specified `Q2_K`. That lands at 79.54 GiB, 2.5 GiB *under* the
82–90 GiB target. `Q4_K` lands at 93.3 GiB, past the 92 GiB hard limit. `Q3_K`
is the only type that puts the artifact in the target band, and it spends the
headroom on the tensor the work order's own reasoning singles out. If GB10
memory turns out tighter than expected, reverting down to `Q2_K` is the cheapest
6 GiB available and costs no engine work.

## Calibration and imatrix

The v1 imatrix was built from a 20.6 MB corpus covering all six languages the
model serves, Korean weighted heaviest. Coverage was checked, not assumed:

```
475 chunks, 243,200 tokens
experts never activated: 0 / 36,096 expert-slots
lowest per-expert activation count: 589   mean: 15,170
```

Every one of 128 experts in all 47 MoE layers was activated. Re-run
`converter/tools/check_imatrix.py` if the corpus or chunk count changes — a
silently zero imatrix row produces a silently bad artifact.

## Measured on `sm_120` (development numbers, not release numbers)

| | |
|---|---|
| Pilot load in `llama-server`, 4 GPUs | 10.2 s (page cache warm) |
| Korean decode, single stream | 78.1 tok/s |
| Broken-jamo ratio | 0.000 |

These are llama.cpp numbers on discrete GDDR7 GPUs. They say the artifact is
sound; they say nothing about GB10 throughput, and must not be quoted as such.

## Not done

- Phase C is partial: the fixture harness exists and the pilot was mid-run when
  it was stopped to free GPUs for the imatrix. No Q4_K_M baseline comparison yet.
- No `sm_121` build has been attempted. The build target is defined in
  `build-manifest.json`; nothing has compiled for GB10.
- No ds4 forward path, so no MTP speculative decode, no continuous batching, no
  OpenAI-compatible serving through ds4.
- Phases F and G are entirely unstarted — they are DGX-Spark-only by definition.
