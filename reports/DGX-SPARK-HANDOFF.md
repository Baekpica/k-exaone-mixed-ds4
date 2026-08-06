# DGX Spark (`sm_121`) Handoff

Everything below was produced on the RTX PRO 6000 (`sm_120`) development host.
None of it has been validated on GB10. This document says what is done, what is
known, and what the `sm_121` stage has to do first.

## What ships

| Item | Where |
|---|---|
| Mixed-quant artifacts + provenance | `Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF` (public) |
| Converter, recipe, verifier, benchmarks | `k-exaone-mixed-ds4` |
| ds4 K-EXAONE loader + reference forward | `Baekpica/ds4`, branch `feature/exaone-model-loader` |
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

`feature/exaone-model-loader` now runs **Phase D2-1 and D2-2**: architecture
detection, hparam validation, tensor binder, layout validation, Q3_K
dequantization, and a complete CPU reference forward. A K-EXAONE GGUF loads and
produces logits. `tests/test_exaone_ref` is the harness.

### Architecture details that are not in the config

These came from reading llama.cpp's exaone-moe graph. Each would have produced
plausible-looking garbage, and each is already handled in the reference path —
**the CUDA kernels have to reproduce all of them**:

1. **RoPE is applied on sliding-window layers only.** Every fourth layer is
   full attention *and* has no positional encoding at all. LLLG is
   local+RoPE / global+NoPE, not merely a window change.
2. **RoPE is NeoX-style**: element *i* pairs with *i + n_rot/2*, not *i+1*.
3. **QK-norm runs before RoPE**, per head over `head_dim` (128), not `n_embd`.
4. **`exp_probs_b` steers top-k selection only.** Expert weights are gathered
   from the *unbiased* sigmoid probabilities, then normalized (clamped at
   6.103515625e-5), then scaled by 2.5. Folding the bias into the weights keeps
   the same experts and still changes every output.
5. **The MTP block is dense.** `blk.48` has `ffn_gate/up/down` at 18432, not
   expert tensors, and shares the base model's embedding and LM head.

### Validation status

Against llama.cpp on the same Q8_0 model and tokens:

- greedy next token matches (argmax 46862 in both)
- `attn_norm-0` matches to four decimals — embedding and RMSNorm are exact
- residual difference is a numerical path difference, not architectural: the
  layer-0 attention output differs by a *constant ~2e-3 absolute* regardless of
  element magnitude, the signature of 8-bit activation quantization. Enabling it
  (`DS4_EXAONE_QUANT_ACT=1`) cuts that error by 3–13× per element. It does not
  converge fully because llama.cpp's CUDA path uses Q8_1 activations, not Q8_0.
- what is left shows up as rank swaps between experts whose sigmoid
  probabilities are nearly tied — 14 of 47 layers match the top-8 in exact
  order, 15 more match the same set in a different order.

Rejected hypotheses, recorded so they are not retried: f16 KV cache (matched
now, changes nothing at position 0 where attention is trivial) and ds4's
`f16_to_f32` (an exact IEEE conversion).

**Position 0 does not exercise RoPE or QK-norm** — at pos 0 the rotation is
identity and softmax over a single position is 1.0 regardless of Q·K. The
multi-token logits comparison does exercise them; the 128-token sliding window
does not yet, and needs a prompt longer than 128 tokens.

### Still missing

1. **CUDA kernels.** Nothing of the above is on the GPU. Both `sm_120` and
   `sm_121` are unwritten.
2. **Batching, KV session management, serving.** The reference path is
   single-token, allocates per call, and has its own cache struct rather than
   ds4's session-managed one.
3. **MTP speculative decode.** The block is bound; nothing runs it.
4. **`Q3_K` activation-quantized dot.** The f32 path exists and is verified;
   the q8-activation fast path skips Q3_K.

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
- ds4 has a CPU reference forward but no CUDA kernels, no batching, no MTP
  speculative decode and no OpenAI-compatible serving.
- Phases F and G are entirely unstarted — they are DGX-Spark-only by definition.
