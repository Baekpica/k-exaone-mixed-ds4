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

### The point of this artifact

**A 237-billion-parameter model, structurally intact, resident on one 128 GB
DGX Spark — with its full 262 144-token context.**

Not a distillation. Not a pruned or expert-dropped variant. Not a
layer-truncated one. Every one of the **128 routed experts** is present in every
one of the 47 MoE layers, alongside the shared expert, the dense layer 0, and
the original 1-layer MTP block. The tensor count matches the BF16 source:
**781 tensors, 237.10 B parameters**. The only thing that changed is the number
of bits each tensor is stored in — assigned by *what the tensor does*, not by a
global bit budget.

The 250 B-class weight class normally implies a multi-GPU host. This artifact
fits **85.56 GiB** of weights and **12.02 GiB** of 256K KV cache into a single
GB10's unified memory, measured at **103.62 GiB of 121.6 GiB resident** and
serving over an OpenAI-compatible API. That is the result this repository
exists to demonstrate.

| | |
|---|---:|
| Parameters | 237.10 B (A23B active) |
| Routed experts kept | **128 / 128**, all 47 MoE layers |
| Tensors | 781 — identical to the BF16 source |
| BF16 size | 441.63 GiB |
| **This artifact (v1)** | **85.56 GiB** — 5.16× smaller |
| Context served on one GB10 | **262 144 tokens** |
| Resident at 256K, measured | **103.62 GiB / 121.6 GiB** |

Mixed-precision GGUF builds of `LGAI-EXAONE/K-EXAONE-236B-A23B`, quantized per
module role rather than uniformly, keeping the parts that matter most at 8 bit.

## Variants

| Variant | Size | Routed gate/up | Routed down | Built with imatrix |
|---|---:|---|---|---|
| **v1** `…-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1` | 85.56 GiB | `IQ2_XXS` | `Q3_K` | yes |
| **pilot** `…-MXQ-Q2K-Q4Edge-Q8Dense-MTPQ8-pilot-v1` | 87.84 GiB | `Q2_K` | `Q2_K` | no |

Each is published as three shards (`-00001-of-00003` …) because the Hub caps
individual files at 50 GB. Point llama.cpp at the **first** shard; it loads the
rest automatically. No merge step is needed:

```bash
llama-server -m K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  -ngl 99 -c 8192
```

The pilot exists because llama.cpp treats `IQ2_XXS` without an importance
matrix as a hard error, so it substitutes `Q2_K` and needs no calibration data.
**v1 is the better artifact on both axes** — 2.3 GiB smaller *and* closer to the
Q8_0 reference (see below) — so prefer it unless you specifically want an
artifact built without calibration data.

| sha256 | |
|---|---|
| v1 (unsplit) | `0e93f4bc41db6eb53c3520352ff7ec0be40749948a6608deb4cc2ad0818c94a1` |
| pilot (unsplit) | `2d840ee44b0e10cb2e14ec7cf58d2e7849615de1a92f58b1220790f42310ce39` |

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

## How to run it

Two runtimes serve these files, and they are not interchangeable.

| | **llama.cpp** | **ds4** (`Baekpica/ds4`, `feature/exaone-model-loader`) |
|---|---|---|
| Runs the artifact | yes, unmodified | yes, unmodified |
| MTP block `blk.48` | **ignored** — stored, never executed | **executed**, target-verified speculative decoding |
| 262 144-token context on one 128 GB device | not measured here | **measured — 103.62 GiB resident** |
| Server API | llama.cpp HTTP API | OpenAI / Responses / Anthropic-compatible |
| Validated on GB10 / `sm_121` | no | **yes** — see below |

### llama.cpp

```bash
llama-server -m K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  -ngl 99 -c 8192
```

Point it at the **first** shard; it loads the other two automatically. A
mixed-quant GGUF needs no special runtime: GGUF stores a type per tensor and
ggml dispatches per tensor, which is how `Q4_K_M` — itself a mixture of Q4_K,
Q6_K and Q8_0 — already works. This recipe just assigns that mixture more
aggressively, and `llama-quantize` is what produced the file.

Measured, not assumed: the pilot artifact loaded in `llama-server` on 4 × RTX
PRO 6000 in **10.2 s** and generated 384 tokens of Korean at **78.1 tok/s** with
a broken-jamo ratio of **0.000**.

The caveat is the MTP block: llama.cpp **ignores** those tensors. They are
preserved in the artifact, not executed.

### ds4 — the engine this artifact was sized for

> **This section describes software outside this repository.** Everything below
> requires
> **[`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader)**,
> branch `feature/exaone-model-loader`. Pin the commit below; the branch moves.

The lineage matters, because almost none of the engine is ours:

| Layer | Repository | What it provides |
|---|---|---|
| Engine | [`antirez/ds4`](https://github.com/antirez/ds4) | the whole runtime — GGUF loader, sessions, KV, CUDA backend, MoE routing, the OpenAI/Responses/Anthropic server, and the NextN/MTP scheduling contract |
| GB10 port | [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) | the `sm_121` build target and the **aligned-artifact tier** that makes mixed-quant MoE weights fast on unified memory |
| This work | [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader) | the `exaone-moe` model family: GQA + QK-norm attention, the LLLG sliding-window schedule, sigmoid/top-8 routing, and the `blk.48` MTP graph |

ds4 was an MLA-only engine; K-EXAONE is plain GQA, so that attention path had to
be written. Neither `antirez/ds4` nor `Entrpi/ds4-on-spark` serves this model as
shipped — use the branch above.

## Serving on DGX Spark (GB10 / `sm_121`) with ds4

Measured on a DGX Spark: NVIDIA GB10, `sm_121`, **121.6 GiB unified memory**,
driver 595.71.05, CUDA 13.3, Linux 6.17.

| | |
|---|---|
| Engine | [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader) |
| Branch | `feature/exaone-model-loader` |
| Commit | `8c45d39956f0edcc88834d9ec93dd026ff32f69d` |
| Weights | [`Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF`](https://huggingface.co/Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF), variant **v1** |
| Converter / reports | [`Baekpica/k-exaone-mixed-ds4`](https://github.com/Baekpica/k-exaone-mixed-ds4) |

**1 — get the weights** (85.56 GiB across three shards):

```bash
hf download Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF \
  --include 'K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-*.gguf' \
  --local-dir ./K-EXAONE-mixed
```

**2 — build the engine.** `make cuda-spark` is the GB10 target; it forces
`CUDA_ARCH=sm_121` across every binary. Building for the wrong architecture is
the single most common way to get wrong kernel results here.

```bash
git clone https://github.com/Baekpica/ds4
cd ds4
git checkout 8c45d39956f0edcc88834d9ec93dd026ff32f69d
make cuda-spark
```

**3 — serve.** This is the exact command validated below:

```bash
./ds4-server \
  -m ../K-EXAONE-mixed/K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  --cuda \
  -c 262144 \
  --host 0.0.0.0 \
  --port 8001
```

**The full 262 144-token context fits on one GB10 with the model resident.**
Cold start to `listening` is about 3 min 45 s, dominated by the one-time
alignment repack. Give the machine ~119 GiB free before starting: the loader
peaks higher than its steady state.

**4 — call it.** Any OpenAI client works; point `base_url` at
`http://<host>:8001/v1` and use any model name (`/v1/models` advertises
`deepseek-v4-flash` and `deepseek-v4-pro`; both serve the loaded GGUF).

```bash
curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "K-EXAONE-236B-A23B",
    "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
    "thinking": {"type": "disabled"},
    "temperature": 0,
    "max_tokens": 64
  }'
```

Useful flags: `--batched-session N` keeps N resident sessions and batches
decode-ready requests (concurrency); `--exaone-mtp` / `--exaone-mtp-timing`
enable the MTP path; `--kv-disk-dir` enables disk KV checkpoints — **not**
recommended for this model, keep K-EXAONE on in-memory KV.

### Resident memory at `-c 262144`

The 85.56 GiB GGUF is mapped once and left **unpinned**; ds4 then materialises
the weights the CUDA backend actually reads. Those two are alternatives, not
additions — the mapping's pages are handed over, not duplicated:

| Component | Size |
|---|---:|
| Aligned CUDA artifacts (repacked at load) | 39.09 GiB — 78 `IQ2` tensors 30.16 GiB + 345 `Q8` tensors 8.93 GiB |
| Raw expert cache payload | 45.39 GiB |
| **Weights resident on device** | **84.48 GiB** |
| KV cache, 262 144 tokens (12 full + 36 sliding layers) | 12.02 GiB |
| Graph workspace | 1.60 GiB |
| Context buffers (`prefill_chunk` 2048) | 104.22 MiB |
| **`nvtop` GPU Mem, idle and ready at 256K** | **103.62 GiB / 121.6 GiB** |

That leaves roughly 18 GiB of headroom on a 121.6 GiB machine with the largest
context the model supports already allocated.

The LLLG schedule is what makes 256K affordable: only 12 of 48 layers keep a
full-context KV, the other 36 keep a 128-position sliding window, so KV costs
**48 KiB/token** instead of the ~192 KiB/token a fully global GQA stack would
need.

### Reading the memory numbers

GB10 is a coherent unified-memory device, so "GPU memory" and "host memory" are
the same physical pool and the usual tools disagree about who owns it:

- `nvidia-smi --query-gpu=memory.used` reports **`[N/A]`** on GB10. Use
  `nvtop -s` and read the process's `gpu_mem_bytes_alloc`.
- Process `VmRSS` **understates** residency by design: ds4 leaves the 85.56 GiB
  model mmap unpinned and hands the pages to the CUDA cache, so most of the
  footprint is CUDA-owned rather than process-anonymous.
- A small-context run is not comparable to a 256K run. The same build with two
  127-token sessions peaks near **90.17 GiB**; the 256K server sits at
  **103.62 GiB**. The difference is almost entirely the 12.02 GiB 256K KV.
- After a clean process exit the NVIDIA 595.71.05 driver may hold the unified
  allocation for a while. `nvtop` showing no process is **not** proof that
  system memory came back — check `free`/`btop` before starting another large
  model, and drop caches if it has not.

### OpenAI-compatible API

`/v1/chat/completions`, `/v1/completions`, `/v1/responses` and `/v1/messages`
are served; `/v1/models` advertises the `deepseek-v4-flash` and
`deepseek-v4-pro` aliases, both of which serve the loaded GGUF. There is no
llama.cpp-style `/health` or `/props` — probe `/v1/models` plus a real
completion. Validated on this host, greedy (`temperature: 0`):

| Check | Result |
|---|---|
| `GET /v1/models` | serves both aliases |
| non-streaming chat completion | `finish_reason=stop`, correct Korean answer, usage populated |
| streaming chat completion | SSE chunks, `finish_reason`, and — with `stream_options: {"include_usage": true}` — a final usage chunk |
| streamed text == non-streamed text | identical under `temperature: 0` |
| thinking mode | `reasoning_content` arrives in its own delta field, never inlined into `content` |
| four sequential requests | no state carried between them; repeating the first request reproduces it byte-for-byte |

Two behaviours worth knowing before you benchmark:

- Thinking is **on by default** for chat requests. With `max_tokens: 64` the
  budget is spent inside `reasoning_content` and `content` comes back empty —
  that is correct, not a hang. Send `"thinking": {"type": "disabled"}` (or
  `"think": false`) for short factual answers.
- Streaming usage follows the OpenAI rule: no `stream_options.include_usage`,
  no usage chunk.

### Multi-token prediction (`blk.48`)

ds4 executes the trained MTP block from this same GGUF — no separate draft
model, no second weight copy. It is **opt-in and off by default**:
`--exaone-mtp` enables it, `--exaone-mtp-timing` adds per-cycle counters.

- Input ordering is the trained one,
  `enorm(embed(x[p+1])) || hnorm(target_hidden[p])`, with the decoder position
  explicitly shifted to `p + 1`.
- Every draft is **verified against the target model's own argmax** and
  committed only on an exact token-ID match, so speculation cannot change
  greedy output. A 64-token identity test passes with `plain == MTP`,
  `mismatch = -1`.
- Extra runtime state is 0.50 MiB (a 128-row private f16 KV ring).
- Speculation runs for greedy requests only (`temperature: 0`).
- **It is slower on this hardware.** On a structured prompt all 12 drafts were
  accepted and end-to-end throughput was still ~10.5 % *lower* than plain
  decode: 2429.9 ms of measured MTP work against a 1957.2 ms target-only
  estimate. The verifier/draft cost, not draft quality, is the problem.
- An automatic loss quench therefore watches the first 12 verifier cycles and
  disables speculation for the rest of the session when measured MTP work runs
  more than 3 % slower. `DS4_EXAONE_MTP_NO_QUENCH=1` defeats it, for
  measurement only.

Keep MTP off for serving until a different verifier/draft schedule shows a
measured gain.

## Measured quality

32 fixtures, greedy (`temperature=0`, `top_k=1`), reasoning off,
`max_tokens` 768, compared against the same fixtures run on the official
**`Q8_0`** build (234.7 GiB) as reference.

| | pilot `Q2_K` · 87.84 GiB | **v1 `IQ2_XXS`+`Q3_K` · 85.56 GiB** |
|---|--:|--:|
| word-agreement vs `Q8_0`, mean | 0.139 | **0.183** |
| — json / tool-call | 0.250 | **0.681** |
| — long-context retrieval | 0.364 | **0.450** |
| identical outputs | 2/32 | 3/32 |
| JSON parses | 4/4 | 4/4 |
| needle retrieved | 3/4 | 3/4 |
| broken-jamo ratio | 0.0001 | 0.0003 |
| repetition (3-gram) | 0.022 | 0.022 |
| decode, 4 × RTX PRO 6000 | 66.5 tok/s | 77.6 tok/s |

v1 tracks the `Q8_0` reference more closely than the pilot **while being
smaller** — the importance matrix and `Q3_K` down are doing real work, most
visibly on structured output. Absolute agreement is low for both because greedy
long-form generation diverges after a single differing token; the pair track for
about 11 words on average before separating. The task-level outcomes (JSON
validity, retrieval, no jamo collapse, no repetition loops) match the `Q8_0`
reference's own scores.

## Limitations

- `IQ2_XXS` on routed gate/up is aggressive. The recipe protects embeddings,
  attention, router, shared expert, dense layer 0, and the edge MoE layers
  specifically to offset it, but expect degradation relative to `Q4_K_M` on
  tasks that lean on rarely-activated experts.
- Evaluation is a 32-prompt fixture set plus the token-fidelity comparison
  above, not a full benchmark suite. Raw results, including the failures, ship
  in the converter repository.
- **The MTP block only runs under ds4**, on the pinned branch and commit above.
  Under llama.cpp it is inert. There is no third runtime that executes it.
- **MTP is slower than plain decode on GB10** even at 100 % draft acceptance,
  so it ships off by default and auto-quenches when enabled.
- **MTP does not run under `--batched-session`.** ds4 disables speculative
  decoding whenever native session batching is active, so concurrency > 1 is
  plain decode regardless of the MTP flags.
- **`nvtop` process disappearance does not mean memory came back.** Driver
  595.71.05 can retain the unified allocation after a clean exit; verify with
  `free`/`btop` before loading another large model.

## Acknowledgements

This artifact is only interesting because there is an engine that serves it, and
that engine is almost entirely other people's work.

- **[`antirez/ds4`](https://github.com/antirez/ds4)** — Salvatore Sanfilippo's
  DwarfStar/ds4, the original engine. Everything here is downstream of it: the
  GGUF loader, the session and KV machinery, the OpenAI/Responses/Anthropic
  server, the CUDA backend, the MoE routing path, and the NextN/MTP scheduling
  contract that the `blk.48` work slots into. The `exaone-moe` family is a new
  model family added to *his* architecture, not a new engine.
- **[`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark)** — the
  DGX Spark fork. The GB10 story rests on it: the `sm_121` build target, and
  above all the **aligned-artifact tier** — repacking `IQ2`/`Q8` tensors into
  alignment-correct CUDA artifacts — which is what makes a mixed-quant MoE
  actually run at speed on GB10's unified memory. That tier was adopted from
  this fork, along with its warning that single-stream MTP can lose on Spark,
  which our own measurements then confirmed.
- **[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) and GGML** —
  Georgi Gerganov and contributors. GGUF, `llama-quantize`, the importance-matrix
  tooling, and the `IQ2_XXS`/`Q3_K`/`Q4_K`/`Q8_0` formats are all theirs; this
  artifact is a `llama-quantize` output, and llama.cpp is still the reference
  runtime for it.
- **[`LGAI-EXAONE`](https://huggingface.co/LGAI-EXAONE)** — K-EXAONE-236B-A23B
  itself, including the trained MTP block that `blk.48` executes.

Mistakes in the recipe, the `exaone-moe` port, and the measurements here are
ours, not theirs.

## License and attribution

Base model © LG Management Development Institute, under the
**K-EXAONE AI Model License Agreement** (`LICENSE`, shipped alongside).

Per §2.1, this Agreement is distributed with the artifact and the derivative
name begins with "K-EXAONE". Note §2.2: distributing or sublicensing the model
or derivative works **to third parties for commercial purposes requires a
separate agreement with the Licensor**. This repository is published as a
research artifact; commercial redistribution is not granted by it.
