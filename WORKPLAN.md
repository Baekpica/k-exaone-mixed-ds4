# K-EXAONE Mixed-Quant GGUF + ds4 — Work Plan

Scope for this session: **everything up to, but not including, the DGX Spark
stage.** All work runs on the RTX PRO 6000 (`sm_120`) development host. Every
DGX-Spark-only step is prepared and packaged as a handoff bundle rather than
executed.

## Delivery targets

| Deliverable | Destination |
|---|---|
| Mixed-quant GGUF artifacts | `Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF`, **public** |
| ds4 K-EXAONE support | fork `Baekpica/ds4` (+ PR to `Entrpi/ds4` where it lands cleanly) |
| DGX Spark handoff bundle | **private** HF bucket |

The handoff bundle carries everything the `sm_121` stage needs and cannot
regenerate cheaply: pinned SHAs, the artifact manifest, reference fixtures and
expected checksums, `sm_120` microbenchmark baselines, and the open items from
Phases E–G.

## Host

| | |
|---|---|
| GPU | 4 × RTX PRO 6000 Blackwell 96 GiB (`sm_120`), P2P OK across all pairs |
| CPU / RAM | 2 × AMD EPYC 9355 (128 threads) / 2.0 TiB |
| CUDA | 13.0.88, driver 580.159.03 |
| `/workspace` | MooseFS FUSE, ~3 TB usable, 1.1 GB/s write / 685 MB/s read |
| `/` | local overlay, ~1 TB usable, 3.8 GB/s write |
| Absent | DGX Spark (`sm_121`) — not reachable from this host |

Storage budget: BF16 GGUF 442 + safetensors 442 + Q4_K_M 134 + artifacts ~270
≈ 1.29 TB of the 3 TB.

## Pinned revisions

| Component | Pin |
|---|---|
| K-EXAONE safetensors | `61e6d578eb102b578e5704e2916ac841df9eca0a` |
| K-EXAONE GGUF | `5bd0394e4f42c00df63e207b9c434387523a6b77` |
| BF16 GGUF sha256 | `73be2da8653976df036bf9b6466b011f86cb10f78bab30a47025638ec999d3f8` (474 281 148 160 B, single file) |
| llama.cpp | `6a32c29a746a2e44de463de647f9f6661eb5086b` (`b10295`, ≫ required `b7737`) |
| ds4 | `b0309611041655f4e45671cfd9c9886aff161406` |

## Model facts established from `config.json` + safetensors headers

Confirms every shape in the work order, plus three things the work order does
not mention:

- **QK-norm** — `self_attn.q_norm` / `k_norm` on all 48 layers *and* on the MTP block.
- **`e_score_correction_bias`** — a per-layer router score-correction bias (47), GGUF `blk.N.exp_probs_b.bias`.
- **The MTP block is dense, not MoE** — one attention layer (with QK-norm), a
  dense `intermediate_size` MLP, `fc` (eh_proj), and three norms. It has no
  embedding or LM head of its own; it shares the base model's.

Totals: 18 688 source tensors → 781 GGUF tensors, 237.10 B parameters,
441.63 GiB at BF16 (projection lands within 0.02 % of the real 441.71 GiB file,
which validates the inventory).

## Quant recipe decisions

`manifests/quant-recipe-v1.yaml` is the source of truth.

**`routed_expert_down`: Q2_K → Q3_K.** The work order's own recipe projects to
79.54 GiB — 2.5 GiB *under* the 82–90 GiB target. Q3_K spends that headroom on
the tensor the work order's stated principle already singles out ("weighted
accumulation quality handled more conservatively than gate/up"): **85.48 GiB,
in target.** Preferred over widening the edge-layer set, which would instead
change a parameter the work order fixes explicitly.

**`v1-pilot-noimatrix`.** llama.cpp's `tensor_requires_imatrix` makes IQ2_XXS a
hard error without an importance matrix (Q2_K does not require one unless the
file ftype is `Q2_K_S`). Work order B3 option 3 applies: the pilot substitutes
Q2_K for gate/up and keeps Q2_K down → **87.77 GiB, also in target.** It is a
usable artifact and the Phase C/D baseline, not a throwaway.

## ds4 findings that shape Phase D

- Model family is a **runtime** value — `g_ds4_shape.family`, enum
  `{DEEPSEEK4=0, GLM_DSA=1}` — reached through the `DS4_MODEL_FAMILY` macro.
  GLM 5.2 was added as a static `ds4_shape` constant plus ~796 conditional
  branches in `ds4.c`. That is the precedent for `EXAONE_MOE`.
- **ds4 is an MLA-only engine.** All four shapes set `n_head_kv = 1` and carry
  MLA/LoRA compression plus a sparse DSA indexer. K-EXAONE is standard **GQA**
  (64 Q / 8 KV heads, `head_dim` 128) with QK-norm and an LLLG sliding-window
  schedule. **No GQA attention path exists** — reference and CUDA both need to
  be written. This is the single largest item in Phase D.
- The MoE side reuses well: `n_ff_exp` 2048 and `expert_weight_scale` 2.5 are
  **identical** to GLM 5.2, which also uses 1 shared expert.
- `n_nextn_predict` already exists in the shape struct (GLM 5.2 sets it to 1)
  and `ds4.c` carries ~875 NextN/MTP references, so MTP tensors are already
  excluded from the normal layer loop. Useful groundwork for Phase F5.
- Every recipe quant type is recognized: `gguf_types[]` covers `q8_0`, `q4_k`,
  `q3_k`, `q2_k`, `iq2_xxs`; `cuda/mmq/` vendors llama.cpp's MMQ/MMVQ suite.

## Order of work

1. **A** — sources, pins, tensor inventory ✱ inventory done ahead of download
2. **B** — recipe → mixed artifact (pilot first, then imatrix → v1)
3. **C** — baseline comparison against Q4_K_M and the BF16 reference
4. **D** — ds4 fork: exaone-moe metadata, tensor binder, reference forward
5. **E (`sm_120` half)** — CUDA kernels, correctness, microbenchmarks
6. **Handoff** — bundle for the `sm_121` stage → private HF bucket

Phases F and G are DGX-Spark-only and are specified, not executed, here.
