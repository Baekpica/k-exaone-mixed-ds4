# K-EXAONE Mixed-Quant v1 — Build and Phase C Report

Development host: 4 × RTX PRO 6000 Blackwell 96 GiB (`sm_120`), CUDA 13.0.88.
**No DGX Spark numbers appear anywhere in this report.** Everything here is a
development measurement on discrete GDDR7 GPUs and must not be quoted as
`sm_121` serving performance.

## Artifacts

| Variant | Projected | Actual | Verified | sha256 |
|---|---:|---:|---|---|
| pilot (`Q2_K` gate/up + down) | 87.84 GiB | **87.84 GiB** | clean | `2d840ee4…10ce39` |
| v1 (`IQ2_XXS` gate/up, `Q3_K` down) | 85.55 GiB | see manifest | see `verify-v1.json` | see manifest |

Source: `K-EXAONE-236B-A23B-BF16.gguf`, 441.71 GiB, sha256 verified locally as
`73be2da8…99d3f8` — matches the value published on the Hub.

The pilot landed within 0.01 GiB of the projection computed from the BF16 tensor
table before it was built, which is the check that the recipe resolution in
`build_mixed.py` and the projection in `project_from_gguf.py` agree with what
`llama-quantize` actually does.

### Type distribution (pilot, verified per tensor)

| Type | Tensors | What |
|---|---:|---|
| `Q8_0` | 346 | attention, dense layer 0, shared expert, embedding, output, MTP block |
| `F32` | 294 | all norms (incl. QK-norm), router `ffn_gate_inp`, `exp_probs_b` |
| `Q2_K` | 117 | routed experts, 39 non-edge MoE layers × 3 |
| `Q4_K` | 24 | routed experts, 8 edge layers × 3 |

781 tensors total. The MTP block at `blk.48` is present with all 15 tensors.
llama.cpp logs `unused tensor blk.48.nextn.*` on load — preserved, not executed,
exactly as intended.

## Two findings that changed the plan

**`routed_expert_down`: `Q2_K` → `Q3_K`.** The work order's recipe projects to
79.54 GiB, 2.5 GiB *under* the 82–90 GiB target. `Q4_K` overshoots to 93.3 GiB,
past the 92 GiB hard limit. `Q3_K` is the only type that lands in the band, and
it spends the headroom on the tensor the work order's own reasoning singles out.
Cost: ds4 has no `Q3_K` CPU reference dequant yet (the CUDA kernel is already
vendored).

**`IQ2_XXS` requires an importance matrix.** llama.cpp's
`tensor_requires_imatrix` makes it a hard error, while `Q2_K` needs none unless
the file ftype is `Q2_K_S`. Work order B3 option 3 applies — the pilot ships
`Q2_K` gate/up and completes the pipeline end to end at 87.84 GiB, in target,
while the imatrix is produced.

## Importance matrix

Corpus: 20.6 MB, 16.2 M characters, our own composition following the bucket
structure of `Baekpica/Solar-Open2-120B-A15B-REAM-148E-Healing-Mix` but
retargeted to K-EXAONE's language set. ko/ja from
`nvidia/Nemotron-SFT-Multilingual-v2`, es/de from `-v1` (v2 has neither), vi
from Wikipedia — no Nemotron release covers Vietnamese. Instruction, reasoning
and code from Cascade Stage-1, IF-Chat-v2 and CompProg-v2. Rendered with
K-EXAONE's own chat markers so the activations match serving.

Korean was boosted 4.3× in a second pass (386 K → 1.65 M Hangul characters).
Routed gate/up goes to the most aggressive quant in the recipe, and Korean
capacity is what this artifact exists to protect.

Coverage was checked rather than assumed:

```
775 chunks × 512 tokens = 396,800 tokens
experts never activated: 0 / 36,096 expert-slots
lowest per-expert activation count: 1,159    mean: 24,770
```

Every one of 128 experts in all 47 MoE layers was activated. This matters: a
never-activated expert has a zero imatrix row, and `IQ2_XXS` then quantizes it
as if unweighted — precisely the experts holding rare capability, failing
silently.

## Phase C — pilot artifact, 32 fixtures

Greedy (`temperature=0`, `top_k=1`), reasoning off, `max_tokens` 768,
`n_ctx` 32768, 4 GPUs, `-sm layer`. Load: 20.6 s.

| Category | n | decode tok/s | prefill tok/s | jamo | rep-3gram | truncated |
|---|--:|--:|--:|--:|--:|--:|
| ko_general_qa | 8 | 63.7 | 262 | 0.0003 | 0.0105 | 6 |
| ko_writing | 4 | 69.9 | 329 | 0.0005 | 0.0174 | 1 |
| en_reasoning | 4 | 60.8 | 308 | 0.0000 | 0.0051 | 3 |
| math | 4 | 61.5 | 334 | 0.0000 | 0.0209 | 2 |
| code | 4 | 61.4 | 334 | 0.0000 | 0.0131 | 2 |
| json_tool | 4 | 79.7 | 354 | 0.0000 | 0.0546 | 0 |
| long_context | 4 | 71.0 | 3228 | 0.0000 | 0.0415 | 0 |

**Overall: 32/32 completed, mean decode 66.5 tok/s, mean jamo 0.0001, mean
rep-3gram 0.0217, JSON parse 4/4, needle retrieval 3/4, invalid UTF-8 0.**

No Hangul collapse, no repetition loops, no malformed UTF-8 — the failure modes
the work order flags for low-bit Korean models did not appear at `Q2_K` experts.

The 14 truncations are long-form answers hitting the 768-token cap, not
defects. Prefill on 5 774-token prompts runs at 3 228 tok/s against ~300 tok/s
on short prompts, which is fixed per-request overhead dominating the short case,
not a long-context penalty.

### A harness bug worth recording

The first run scored **JSON 1/4 and needle 2/4** and looked like a quantization
failure. It was not. K-EXAONE is a reasoning model; llama.cpp returns its
reasoning in `message.reasoning_content`, and at `max_tokens=384` those prompts
hit the cap mid-thought with `content` still empty. `finish_reason` was
`length` on exactly the failures and `stop` on exactly the passes — a single
probe with `max_tokens=1500` returned 5 589 characters of reasoning and no
content at all.

The chat template exposes `enable_thinking`; passing it `false` emits a closed
`<think></think>` and the model answers directly — 60 tokens, valid JSON,
`finish_reason=stop`. Re-running gave 4/4 and 3/4.

Recorded because the mistake is cheap to repeat: **on a reasoning model, a
format-compliance metric measured under a token cap measures the cap.**

## Not measured

- No Q4_K_M or BF16 baseline comparison yet. The Q8_0 greedy reference is being
  generated; first-divergence and token-agreement numbers against it are not in
  this report.
- No v1-vs-pilot quality comparison — v1 finished building after these runs.
- Nothing on DGX Spark: no resident memory, no `sm_121` correctness, no serving
  throughput, no MTP speculative decode. Those are Phases E-`sm121`, F and G.
