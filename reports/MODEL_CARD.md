---
license: other
license_name: k-exaone
license_link: LICENSE
base_model: LGAI-EXAONE/K-EXAONE-236B-A23B
base_model_relation: quantized
language: [en, ko, es, de, ja, vi]
pipeline_tag: text-generation
library_name: gguf
tags: [gguf, k-exaone, exaone, moe, mixed-quantization, mtp, dgx-spark]
---

# K-EXAONE-236B-A23B — Mixed-Quant GGUF

Mixed-precision GGUF builds of `LGAI-EXAONE/K-EXAONE-236B-A23B`, quantized per
module role rather than uniformly, so a 237 B-parameter MoE fits a single
128 GB unified-memory device while keeping the parts that matter most at 8 bit.

**All 128 routed experts, the shared expert, and the original 1-layer MTP block
are preserved.** No experts were pruned or merged.

## Variants

| File | Size | Routed gate/up | Routed down | Needs imatrix |
|---|---:|---|---|---|
| `…-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1.gguf` | 85.5 GiB | `IQ2_XXS` | `Q3_K` | yes (built with one) |
| `…-MXQ-Q2K-Q4Edge-Q8Dense-MTPQ8-pilot-v1.gguf` | 87.8 GiB | `Q2_K` | `Q2_K` | no |

The pilot exists because llama.cpp treats `IQ2_XXS` without an importance
matrix as a hard error. It is a complete, usable artifact — slightly larger and
built without calibration data — not a preview of the other one.

## Recipe

Quantization is assigned by what each tensor does, not by a global bit budget.

| Tensor group | Type | Why |
|---|---|---|
| Token embedding, LM head | `Q8_0` | multilingual token fidelity; logit sensitivity |
| All norms (incl. QK-norm) | `F32` | tiny, and error accumulates through them |
| Router (`ffn_gate_inp`, `exp_probs_b`) | `F32` | a wrong expert choice costs more than any bit saved |
| Attention Q/K/V/O | `Q8_0` | long-context stability |
| Dense layer 0 MLP | `Q8_0` | every token passes through it |
| Shared expert | `Q8_0` | every token passes through it |
| Routed expert gate/up | `IQ2_XXS` (pilot: `Q2_K`) | ~64 % of all parameters; where the compression has to come from |
| Routed expert down | `Q3_K` (pilot: `Q2_K`) | weighted accumulation, kept more conservative than gate/up |
| Edge MoE layers 1–4, 44–47 | `Q4_K` | first and last sparse blocks protected |
| MTP block (`blk.48`) | `Q8_0` | draft quality drives speculative acceptance |

Full recipe: `quant-recipe-v1.yaml`. Exact per-tensor assignments as fed to
`llama-quantize`: `*.tensor-types.txt`. Per-tensor verification against the
recipe: `verify-*.json`.

### Calibration (v1 only)

The importance matrix was built from a corpus covering all six languages the
model serves — Korean weighted heaviest, since routed gate/up goes to the most
aggressive quant in the recipe and Korean capacity is what this artifact exists
to protect. Sources: `nvidia/Nemotron-SFT-Multilingual-v2` (ko, ja),
`-v1` (es, de), `Nemotron-Cascade-SFT-Stage-1`,
`Nemotron-SFT-Instruction-Following-Chat-v2`,
`Nemotron-SFT-Competitive-Programming-v2`, and Wikipedia for Vietnamese —
neither Nemotron release covers it. Records are rendered with K-EXAONE's own
chat template so the activations the matrix sees match serving time.
Composition: `calibration.composition.json`.

## Provenance

| | |
|---|---|
| Source model | `LGAI-EXAONE/K-EXAONE-236B-A23B` @ `61e6d578eb102b578e5704e2916ac841df9eca0a` |
| Source GGUF | `LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF` @ `5bd0394e4f42c00df63e207b9c434387523a6b77` |
| BF16 GGUF sha256 | `73be2da8653976df036bf9b6466b011f86cb10f78bab30a47025638ec999d3f8` |
| llama.cpp | `6a32c29a746a2e44de463de647f9f6661eb5086b` (build `b10295`) |
| Converter | [`k-exaone-mixed-ds4`](https://github.com/Baekpica/k-exaone-mixed-ds4) |

Artifact sha256 and build parameters: `*.manifest.json`.

## Model structure

48 transformer layers plus one MTP block stored as `blk.48`, so
`exaone-moe.block_count` is 49. Hidden 6144, vocab 153 600, context 262 144.
Attention is GQA — 64 query heads over 8 KV heads at head_dim 128 — with
per-head RMSNorm on Q and K, on an LLLG sliding-window schedule (window 128;
every fourth layer is full attention). Layer 0 is dense (18 432); layers 1–47
are MoE with 128 routed experts, top-8, sigmoid gating with normalized top-k
probabilities, `routed_scaling_factor` 2.5, plus one 2048-wide shared expert.
The MTP block is a dense layer with its own attention and `eh_proj`; it shares
the base model's embedding and LM head.

## Usage

```bash
llama-server -m K-EXAONE-236B-A23B-MXQ-*.gguf -ngl 99 -c 8192
```

A mixed-quant GGUF needs no special runtime: GGUF stores a type per tensor and
ggml dispatches per tensor, which is how `Q4_K_M` — itself a mixture of Q4_K,
Q6_K and Q8_0 — already works. This recipe just assigns that mixture more
aggressively, and `llama-quantize` is what produced the file.

Measured, not assumed: the pilot artifact loaded in `llama-server` on 4 × RTX
PRO 6000 in **10.2 s** and generated 384 tokens of Korean at **78.1 tok/s** with
a broken-jamo ratio of **0.000**.

Two real caveats:

- llama.cpp **ignores the MTP tensors**. They are preserved in the artifact, not
  executed. Speculative decoding through the MTP block is engine work.
- **ds4 cannot serve this yet.** The K-EXAONE model family in
  [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader)
  currently has metadata validation and the tensor binder; the forward path is
  not implemented. ds4 is an MLA-only engine and K-EXAONE is plain GQA, so that
  attention path has to be written. Until then llama.cpp is the way to run these
  files.

## Limitations

- `IQ2_XXS` on routed gate/up is aggressive. The recipe protects embeddings,
  attention, router, shared expert, dense layer 0, and the edge MoE layers
  specifically to offset it, but expect degradation relative to `Q4_K_M` on
  tasks that lean on rarely-activated experts.
- Evaluation to date is a 32-prompt fixture smoke test plus token-fidelity
  comparison against a higher-precision reference, not a full benchmark suite.
  Results ship in the converter repository, including the cases that failed.
- Not yet validated on DGX Spark / `sm_121`. Sizing targets that device, but
  the resident-memory, context and throughput numbers there are not measured.

## License and attribution

Base model © LG Management Development Institute, under the
**K-EXAONE AI Model License Agreement** (`LICENSE`, shipped alongside).

Per §2.1, this Agreement is distributed with the artifact and the derivative
name begins with "K-EXAONE". Note §2.2: distributing or sublicensing the model
or derivative works **to third parties for commercial purposes requires a
separate agreement with the Licensor**. This repository is published as a
research artifact; commercial redistribution is not granted by it.
