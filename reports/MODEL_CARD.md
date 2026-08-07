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
| llama.cpp (quantizer) | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) @ `6a32c29a746a2e44de463de647f9f6661eb5086b` (build `b10295`) |
| Converter | [`Baekpica/k-exaone-mixed-ds4`](https://github.com/Baekpica/k-exaone-mixed-ds4) |
| Serving engine (measured below) | [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader) @ `8c45d39956f0edcc88834d9ec93dd026ff32f69d` |
| — upstream engine | [`antirez/ds4`](https://github.com/antirez/ds4) |
| — DGX Spark port | [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) |

Artifact sha256 and build parameters: `*.manifest.json`. Tensor-level
verification against the recipe (`verify-v1.json`): **781 tensors, 85.558 GiB,
0 errors, 0 warnings**, matching the BF16 source's own tensor count.

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

**Budget `--batched-session` carefully.** Each resident session costs its own KV
*and* its own 1.60 GiB graph workspace — the workspace is not shared across
slots — so `N` sessions cost `N × (KV + 1.60 GiB)`. With 84.48 GiB of weights
resident there is roughly 35 GiB to spend:

| ctx per slot | KV/slot | + workspace | 8 slots | observed total |
|---:|---:|---:|---:|---:|
| 16 384 | 0.75 GiB | 2.35 GiB | 18.8 GiB | **108.8 GiB**, 9 GiB free |
| 40 960 | 1.89 GiB | 3.49 GiB | 27.9 GiB | **117.9 GiB**, 0.5 GiB free — too tight |

`--batched-session 8 -c 40960` boots but leaves the machine with half a gigabyte
of headroom. Prefer more context per slot or fewer slots, not both.

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
- **After a clean exit the driver keeps the memory, and that is fine.** With
  595.71.05, `free` reports roughly 14 GiB available after `ds4-server` exits,
  and it stays there: the whole of `/proc/meminfo` accounts for only ~17.7 GiB
  of the 127.5 GiB total, so the ~110 GiB is held by the NVIDIA kernel module,
  not by page cache. Dropping caches cannot reclaim it — there is nothing in
  the page cache to drop.

  It also does not need reclaiming. The next CUDA process reuses the driver's
  pool directly: a second 256K server booted normally in 230 s with
  `MemAvailable` still showing 14 GiB. **The precondition for booting is that
  no other `ds4-server` is running — not a `MemAvailable` threshold.** A
  readiness check that waits for free memory will wait forever.

### Measured throughput

Greedy (`temperature: 0`), thinking disabled, 128 generated tokens per request,
one **cold** prompt per measurement over `/v1/chat/completions` with streaming.
`prefill t/s` is `prompt_tokens / time-to-first-token` — what a client actually
waits for. `decode t/s` is measured between the first and last content chunk.
Raw per-request records ship in the converter repository.

| Prompt tokens | Prefill t/s | Decode t/s | Time to first token |
|---:|---:|---:|---:|
| 1 451 | 53.0 | 10.51 | 27.4 s |
| 3 941 | 51.6 | 9.05 | 76.4 s |
| 8 222 | 47.9 | 7.38 | 171.6 s |
| 16 376 | 42.3 | 5.42 | 387.1 s |

Both curves are clean enough to fit and extrapolate.

**Decode cost is linear in context depth:**

```text
ms per token = 86.6 + 0.00597 × context_tokens      (residuals < 0.4 ms)
```

**Prefill cost is quadratic in prompt length**, because the marginal cost of the
next 2048 tokens grows linearly with the depth they start at:

```text
seconds per 2048-token chunk = 37.5 + 0.00145 × depth_tokens   (16 points, residuals < 0.25 s)
```

Which gives, for depths beyond what was measured directly:

| Context | Marginal prefill t/s | Cold prefill of a full prompt | Decode t/s |
|---:|---:|---:|---:|
| 8 192 | 41.5 | 2.7 min | 7.4 (measured) |
| 32 768 | 24.1 | 16 min | 3.5 |
| 65 536 | 15.5 | 45 min | 2.1 |
| 131 072 | 9.0 | 2.3 h | 1.2 |
| 262 144 | 4.9 | 8.0 h | 0.6 |

### What that means in practice

The 262 144-token context **fits, is allocated, and is resident** — that is a
memory result and it holds. It is not a throughput result. A cold 256K prompt
would take hours to prefill on this hardware, and decode at that depth runs
below 1 token/s. **Useful working depths on one GB10 today are roughly 2K–32K.**

**Multi-turn chat re-pays the whole prefill.** This was measured, not assumed,
and it is the most important thing to know before building on this:

| Turn | Prompt tokens | Time to first token |
|---|---:|---:|
| 1 — cold, ~7K document + question | 6 978 | 166.2 s |
| 2 — same history + the assistant's own reply + a follow-up | 7 086 | **143.9 s** |
| 3 — a different document, cold | 6 725 | 136.5 s |

Turn 2 is a normal continuation and costs the same as a cold prompt. The server
log shows why: the live checkpoint held 7 074 tokens, the new prompt shared
**6 984** of them, and all of it was discarded. ds4 reuses session KV only when
the new prompt contains the *entire* checkpoint
(`prompt->len >= checkpoint.len && ds4_tokens_starts_with(...)`), and the
checkpoint includes the tokens the model itself generated. Replaying the
assistant's reply as text re-tokenises a few of those differently, the
all-or-nothing test fails, and 98.6 % of a valid prefix is thrown away.

Plan for it: at these prefill rates a 7K-token chat turn costs about 2.5
minutes each time. Keep conversational context small, or keep the transcript
short enough that re-prefilling is affordable.
Concurrency, however, does **not** help today. With `--batched-session 8` and
short prompts so prefill cannot interfere, aggregate decode throughput is flat:

| Concurrent streams | Summed decode t/s | Per stream |
|---:|---:|---:|
| 1 | 11.12 | 11.12 |
| 2 | 9.80 | ~4.9 |
| 4 | 10.03 | ~2.5 |
| 8 | 10.80 | ~1.35 |

Some of that is inherent — in a top-8-of-128 MoE, concurrent tokens route to
largely disjoint experts, so routed-expert weight reads do not amortise across
a batch. The shared components should still amortise and do not appear to.
With **cold** prompts it is worse than flat: prefill is serialised across slots,
so `N` concurrent long requests behave like `N` sequential ones and wall
throughput *falls* (3.18 t/s at one stream to 2.01 t/s at eight).

Batching itself is free — at concurrency 1 the batched server matches the plain
one (10.39 vs 10.51 t/s at 2K, 7.37 vs 7.38 at 8K) — so `--batched-session` is
worth using for fairness and slot residency, just not for throughput.

### Where the time goes

The 12 full-attention layers hold **49 152 bytes of KV per context position**
(GQA, 8 KV heads × 128 dims, K and V, f16). Decode adds **5.97 µs per context
position**, which is an effective **8.2 GB/s** on a part with roughly 273 GB/s
of memory bandwidth — about **3 %**. The depth-dependent part of decode is
therefore not bandwidth-bound. The `exaone-moe` GQA decode attention path is the
limiter, and it is the obvious first optimisation target.

For scale: ds4's tuned MLA path on DeepSeek V4 Flash reaches 825 t/s prefill and
18 t/s decode on this same GB10 (`speed-bench/gb10.csv` in the engine repo).
That model has far fewer active parameters, so the absolute numbers are not
comparable — but the *shape* is. MLA prefill is nearly flat with depth
(825 → 823 t/s from 2K to 64K) where `exaone-moe` roughly halves every 4×.

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
- An automatic loss quench watches the first 12 verifier cycles and disables
  speculation for the rest of the session when measured MTP work runs more than
  3 % slower. `DS4_EXAONE_MTP_NO_QUENCH=1` defeats it, for measurement only.

**It loses at short context and reaches break-even at long context.** Measured
with the quench defeated, so the whole generation is speculative:

| Context | Draft acceptance | Cycle | MTP ms/token | Plain ms/token | vs plain |
|---:|---:|---:|---:|---:|---:|
| 1 451 | 69.3 % | 204.1 ms | 120.5 | 95.6 | +26 % |
| 7 924 | 44.3 % | 241.7 ms | 167.5 | 133.7 | +25 % |
| 32 995 | 34.0 % | 390.3 ms | 291.2 | 281.2 | **+4 %** |

The mechanism is a single ratio. A cycle runs one draft pass (~13 ms, roughly
constant) plus one **two-row** target verify pass. Write **k** for the cost of
that two-row pass relative to an ordinary one-row decode, and **a** for draft
acceptance; a cycle commits `1 + a` tokens, so speculation wins exactly when

```text
k < 1 + a
```

Measured, k falls with depth — 2.06, 1.76, **1.36** — because the KV read the
two rows share amortises as the context grows, while acceptance falls — 69.3 %,
44.3 %, 34.0 % — because the MTP block's private KV ring is only 128 rows and
starts cold. At 32 995 tokens the two sides are 1.36 against 1.340: break-even
to within measurement noise, and still improving with depth.

So `k` is the lever, not draft quality. An ideal two-row pass would share the
weight streaming between its rows and cost `k ≈ 1.05`, at which point even 34 %
acceptance turns into roughly a 25 % speedup. That k is 1.36 rather than 1.05
is the same finding as the decode-attention result above: this path is bound by
per-row cost, not by bandwidth.

**Keep MTP off for serving today** — it is a loss below ~32K and a wash above.
It becomes worth enabling if either the two-row verify pass gets cheaper or the
MTP KV is warmed from the prompt instead of starting cold.

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
- **256K is a memory result, not a throughput result.** The context is allocated
  and resident, but a cold prompt at that depth takes hours to prefill and
  decodes below 1 token/s. Plan for 2K–32K of working context on one GB10.
- **Prefill is quadratic and decode is linear in context depth**, both steeper
  than ds4's MLA models on the same hardware. The measured decode attention
  path reaches only ~3 % of the device's memory bandwidth, so this is an
  engine-side limit with real headroom, not a property of the artifact.
- **The MTP block only runs under ds4**, on the pinned branch and commit above.
  Under llama.cpp it is inert. There is no third runtime that executes it.
- **MTP is a loss below ~32K of context and a wash above it**, so it ships off
  by default and auto-quenches when enabled. The limit is the cost of the
  two-row verify pass, not draft quality.
- **MTP does not run under `--batched-session`.** ds4 disables speculative
  decoding whenever native session batching is active, so concurrency > 1 is
  plain decode regardless of the MTP flags.
- **Host memory accounting is not usable as a readiness signal.** Driver
  595.71.05 retains the unified allocation after a clean exit and does not
  return it to the kernel. Gate a restart on "no `ds4-server` process", not on
  `free`.

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
