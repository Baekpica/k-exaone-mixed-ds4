# K-EXAONE × ds4 DGX Spark — checklist items 4–6

Continues `DGX-SPARK-HANDOFF-2026-08-07-1626-KST.md`, which finished items 1–3
and handed off items 4–6 unrun.

Host: `thinkstationpgx-8abc` — NVIDIA GB10 (`sm_121`), 121.6 GiB unified RAM,
driver 595.71.05, CUDA 13.3, Linux 6.17.

ds4: `feature/exaone-model-loader` @ `8c45d39956f0edcc88834d9ec93dd026ff32f69d`,
worktree clean, built with `make cuda-spark`. The branch has since been **pushed
to `origin`** (it was 11 commits ahead at handoff time).

Model: mixed-quant v1, 85.56 GiB, 3 shards.

## Disposition

| Item | State |
|---|---|
| 4 — plain/MTP context × concurrency matrix | run with a reduced frontier set; see *Why the matrix was reduced*. Deep and concurrent cells are still landing as this is written. |
| 5 — Model Card edit / upload / remote verify | done |
| 6 — final 256K serving + API validation | done, all 16 checks pass |

## Item 6 — 256K serving and API validation

Command, exactly as handed off:

```bash
cd /home/sunghoon/workspace/ds4-exaone/ds4
./ds4-server \
  -m ../models/K-EXAONE-236B-A23B-Mixed-Quant-GGUF/K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  --cuda -c 262144 --host 0.0.0.0 --port 8001
```

Cold start to `listening`: **3 min 46 s**. Startup breakdown from the server log:

```text
mapped 3 GGUF shards (85.56 GiB) into one 85.56 GiB address range, 781 tensors
iq2 aligned repack base:  78 tensors 30.16 GiB in 68.2s (threads=6)
q8  aligned repack base: 345 tensors  8.93 GiB in 17.1s (threads=6)
aligned artifacts (records): 423 built, 39.09 GiB device, 85.3s
exaone expert residency: 45.39 GiB raw CUDA-owned payload added
exaone final CUDA residency: 39.09 GiB aligned artifacts + 45.39 GiB raw cache payload
exaone graph: ctx 262144, prefill chunk 2048, KV 12.02 GiB (12 full + 36 sliding layers), workspace 1.60 GiB
```

`nvtop -s` at idle-and-ready: **111 261 696 000 B = 103.62 GiB** of 121.6 GiB.
This matches the 103.523 GiB recorded for the previous 256K server.

### Validation results

All 16 checks pass (`scratch/matrix/A-plain-256k-c1/api/api-validation.json`):

| Check | Result |
|---|---|
| `GET /v1/models` | `deepseek-v4-flash`, `deepseek-v4-pro` |
| chat completion, default (thinking) mode | `finish_reason=length`, 64 completion tokens |
| chat completion, `thinking.type=disabled` | `finish_reason=stop`, `"대한민국의 수도는 서울입니다."` |
| `usage.prompt_tokens` populated | yes |
| streaming: SSE chunks | 11 chunks |
| streaming: `finish_reason` | `stop` |
| streaming: usage chunk | present **only with `stream_options.include_usage`** |
| streamed text == non-streamed text | identical under `temperature: 0` |
| thinking mode `reasoning_content` separation | 625 reasoning chars, `content` empty, no `<think>` leakage |
| 4 sequential requests answered | yes |
| no state leaked between requests | request C never mentions request A's answer |
| greedy repeat reproducible | request D reproduces request A byte-for-byte |

### Two behaviours that are correct but surprising

1. **Streaming usage requires `stream_options: {"include_usage": true}`.** The
   first validation pass recorded this as a failure; it is not. `sse_usage_chunk`
   returns early without the flag, which is the OpenAI contract. Any harness
   that reads `usage` off the stream must send the flag.
2. **Thinking is on by default for chat requests.** With `max_tokens: 64` the
   whole budget is spent inside `reasoning_content` and `content` comes back
   empty. The handoff's own curl hits this. Send
   `"thinking": {"type": "disabled"}` for short factual answers.

## Item 4 — benchmark matrix

### Why the matrix was reduced

The handoff asked for context `2K…256K` × concurrency `1,2,4,8` × mode
`plain,MTP`. Measured prefill throughput makes the full grid unaffordable, and
the reason is worth recording because it is also the main optimisation target.

ds4 reuses a session's KV **only when the new prompt is an exact token-prefix
extension of the live checkpoint** (`ds4_session_sync_internal`:
`prompt->len >= checkpoint.len && ds4_tokens_starts_with(prompt, &checkpoint)`).
A benchmark that grows a prompt and also generates tokens breaks that condition
on every step, so every matrix cell pays a **cold full-length prefill**. At the
measured rate a single 256K prefill costs roughly 8 hours, and one concurrency-8
cell at 32K costs 8 × 32K tokens of prefill.

The matrix was therefore cut to what fits, and the omitted cells are named
rather than silently dropped. See *Cells not run*.

### Method

- Driver: `scratch/matrix/matrix_bench.py`, stdlib only, over
  `/v1/chat/completions` with `stream: true` and
  `stream_options.include_usage`.
- Every request: `temperature: 0`, `thinking: {"type": "disabled"}`,
  `max_tokens: 128`.
- Prompt: a slice of `speed-bench/promessi_sposi.txt` plus a fixed Korean
  instruction. Each concurrency stream reads a **disjoint** slice so streams
  cannot share a KV prefix.
- `prefill t/s` = `prompt_tokens / TTFT`, where TTFT is measured client-side to
  the first content delta. It therefore includes queueing and the first decode
  step, and is the number a client actually experiences.
- `decode t/s` = `(completion_tokens − 1) / (t_last_chunk − t_first_chunk)`.
- Memory is sampled from `/proc/meminfo`, `/proc/<pid>/status` and `nvtop -s`
  before and after every cell.
- One JSON record per cell in `cells.jsonl`, with the full per-request detail,
  alongside an immutable `run-meta.tsv` per server boot (commit, dirty-file
  count, model path, driver, kernel, exact command line, boot seconds).

Two earlier harness attempts were discarded and are kept for audit:

- `/v1/completions` on raw text — at some corpus offsets the greedy next token
  is EOS, so `completion_tokens` came back 0 and there was no decode phase.
- the SFT calibration corpus — it is a multiple-choice set, so slices ending in
  `"Answer: "` made the model emit one letter and stop.

### Results — single stream, plain

| Prompt tokens | TTFT | Prefill t/s | Decode t/s | GPU Mem |
|---:|---:|---:|---:|---:|
| 1 451 | 27.4 s | 53.0 | 10.51 | 103.62 GiB |
| 3 941 | 76.4 s | 51.6 | 9.05 | 103.62 GiB |
| 8 222 | 171.6 s | 47.9 | 7.38 | 103.62 GiB |
| 16 376 | 387.1 s | 42.3 | 5.42 | 103.62 GiB |
| 33 068 | 971 s | 34.0 | 3.56 | 103.62 GiB |
| 65 662 | 2 700 s | 24.3 | 2.11 | 103.62 GiB |

### Depth models, and the fact that they were predictive

```text
decode:  ms per token             = 87.098 + 0.005883 x context_tokens   (6 points, residuals < 1.0 ms)
prefill: seconds per 2048 tokens  = 37.303 + 0.0014555 x depth_tokens    (61 points, residuals < 0.9 s)
```

Both models were fitted on the shallow cells and then **tested against cells
that had not run yet**:

| Cell | Predicted | Measured | Error |
|---|---:|---:|---:|
| decode @ 32 768 | 3.58 t/s | 3.56 t/s | 0.6 % |
| decode @ 65 536 | 2.12 t/s | 2.11 t/s | 0.5 % |
| cold prefill @ 32 768 | 953 s | ~971 s | 1.9 % |
| cold prefill to 40 960 | 1 311 s | 1 308.8 s | 0.2 % |
| cold prefill @ 65 536 | 2 669 s | ~2 700 s | 1.2 % |

That is what licenses the two rows below that were never run:

| Context | Marginal prefill t/s | Cold prefill | Decode t/s |
|---:|---:|---:|---:|
| 32 768 | 24.1 | 15.9 min | 3.56 (measured) |
| 65 536 | 15.4 | 44.5 min | 2.11 (measured) |
| 131 072 | 9.0 | 2.3 h | 1.17 |
| 262 144 | 4.9 | 8.1 h | 0.61 |

### Results — MTP, single stream, quench defeated

`DS4_EXAONE_MTP_NO_QUENCH=1`, so the whole 128-token generation is speculative.

| Context | Acceptance | Cycle | MTP ms/token | Plain ms/token | vs plain | k = 2-row / 1-row |
|---:|---:|---:|---:|---:|---:|---:|
| 1 451 | 69.3 % | 204.1 ms | 120.5 | 95.6 | +26 % | 2.06 |
| 7 924 | 44.3 % | 241.7 ms | 167.5 | 133.7 | +25 % | 1.76 |
| 32 995 | 34.0 % | 390.3 ms | 291.2 | 281.2 | +4 % | 1.36 |

This **changes the item-3 conclusion**, which was drawn at 64 tokens of context
on a structured integer prompt and reported 100 % acceptance with a 10.5 % loss.
On natural prose, acceptance is much lower and falls with depth — but the loss
falls faster, and at 33K MTP is at break-even.

A cycle runs one draft pass (~13 ms, roughly constant) plus one **two-row**
target verify pass. With `k` the cost of that two-row pass relative to an
ordinary one-row decode and `a` the acceptance rate, a cycle commits `1 + a`
tokens, so speculation wins exactly when

```text
k < 1 + a
```

| Context | k | 1 + a | verdict |
|---:|---:|---:|---|
| 1 451 | 2.06 | 1.693 | lose |
| 7 924 | 1.76 | 1.443 | lose |
| 32 995 | **1.36** | **1.340** | break-even |

`k` falls with depth because the KV the two rows share amortises; `a` falls
because the MTP block's private KV ring is 128 rows and starts cold. They
converge at ~33K.

**`k` is the lever, not draft quality.** A two-row pass that shared its weight
streaming between rows would cost `k ≈ 1.05`; at that k, even 34 % acceptance
is a ~25 % speedup. That k is 1.36 instead is the same defect as the decode
attention result in *Findings*: this path is bound by per-row cost, not by
bandwidth. The second lever is acceptance — warming the MTP KV from the prompt
instead of starting cold should recover much of the 69 % seen at shallow depth.

### Cells not run, and why

| Cell | Status | Reason |
|---|---|---|
| 256K, any concurrency | **not run** | one cold 256K prefill is ~8 h at the measured rate; predicted from the fitted model instead |
| 128K, any concurrency | **not run** | ~2.3 h per prefill; same treatment |
| 64K, concurrency 2/4/8 | **not run** | 2–8 × 44.5 min of serialised prefill per cell |
| MTP at concurrency 2/4/8 | **not applicable** | ds4 disables speculative decoding whenever `--batched-session` is active (`ds4_server.c`: the speculative branch is gated on `!s->batched_mode`, and the server logs "MTP speculative decoding is disabled while native session batching is active"). Reporting these as MTP results would be reporting plain decode under an MTP label. |

MTP cells are run with `DS4_EXAONE_MTP_NO_QUENCH=1`. Without it the quench
resets on every prompt sync and fires again after 12 cycles, so a 128-token
generation would be ~12 speculative cycles followed by ~104 plain ones —
measuring the quench policy rather than MTP itself. The quench policy was
already validated in item 3; what is open is whether MTP wins at *any* depth.

## Item 5 — Model Card

`reports/MODEL_CARD.md` was rewritten, not patched. The two stale claims the
handoff named are gone:

- "ds4 cannot serve this yet … the forward path is not implemented" — removed.
- "Not yet validated on DGX Spark / `sm_121`" — removed.

What replaced them, plus what the handoff asked for:

| Handoff requirement | Where it landed |
|---|---|
| GB10 resident model breakdown and 256K KV budget | *Resident memory at `-c 262144`* |
| `nvtop` GPU Mem versus Host/RSS caveat | *Reading the memory numbers* |
| exact ds4 build and serving command | *Serving on DGX Spark*, steps 1–4, with the commit pinned |
| OpenAI API behaviour | *OpenAI-compatible API*, plus the two surprising defaults |
| MTP identity, acceptance, throughput loss, auto-quench, default-off | *Multi-token prediction (`blk.48`)* |
| native-batch MTP and NVIDIA UMA cleanup limitations | *Limitations* |

Three things were added beyond the handoff list, on the user's instruction:

1. The card now **leads with what the artifact is for** — a 237 B model kept
   structurally intact (128/128 routed experts, 781 tensors, 0 verification
   errors) and resident on one 128 GB device with its full context — rather
   than with the quantization recipe.
2. The card **names its serving dependency** and walks through
   download → `make cuda-spark` → serve → call, because nothing in it is
   reproducible without that specific engine branch.
3. **Acknowledgements**: [`antirez/ds4`](https://github.com/antirez/ds4) for the
   runtime the whole thing sits on, and
   [`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) for the
   `sm_121` target and the aligned-artifact tier the GB10 result depends on,
   alongside llama.cpp/GGML and LGAI-EXAONE.

The same lineage and acknowledgement went into the converter repo README and
into the ds4 fork's own README, whose *Model Weights* section still claimed the
engine only works with DeepSeek V4 and GLM 5.2 GGUFs.

## Findings that are not on the checklist

### 1. The decode attention path is the bottleneck, with ~30× headroom

The 12 full-attention layers hold 49 152 bytes of KV per context position
(GQA, 8 KV heads × 128 dims, K and V, f16 — confirmed against the engine's own
"KV 12.02 GiB (12 full + 36 sliding layers)" at `ctx=262144`). Decode adds
**5.966 µs per context position**, i.e. an effective **8.2 GB/s** against
roughly 273 GB/s of available bandwidth — about **3 %**.

So the depth-dependent half of decode is not bandwidth-bound. By contrast the
depth-*independent* half (86.6 ms/token) is roughly consistent with streaming
the ~8 GiB of active mixed-quant weights, so the MoE weight path is fine. The
GQA decode attention kernel is the thing to fix.

### 2. Prefill does not reuse a prefix once anything has been generated

`ds4_session_sync_internal` reuses KV only when
`prompt->len >= checkpoint.len && ds4_tokens_starts_with(prompt, &checkpoint)`,
and the checkpoint accumulates generated tokens during decode. Any client that
extends a prompt *without* replaying the assistant's own output verbatim pays a
full cold prefill. This is correct behaviour, but it is a sharp edge worth
documenting for agent frameworks that rebuild prompts.

### 3. Two API defaults silently break naive benchmarks

- Streaming omits `usage` unless `stream_options: {"include_usage": true}` is
  sent. A harness that reads token counts off the stream will see `null`.
- Chat defaults to thinking mode, so a small `max_tokens` is consumed entirely
  inside `reasoning_content` and `content` returns empty with
  `finish_reason=length`.

### 4. Corpus choice is not neutral for a serving benchmark

Two harness generations were discarded before the numbers above were trusted.
Raw `/v1/completions` continuation of arbitrary text hits corpus offsets where
the greedy next token is EOS (`completion_tokens=0`, no decode phase to
measure). The SFT calibration corpus is a multiple-choice set, so slices ending
in `"Answer: "` make the model emit a single letter and stop. Both are recorded
in `scratch/matrix/A-plain-256k-c1/bench-abandoned-*`.

### 5. The handoff's `clear_cache` precondition is wrong, and the reason matters

The previous handoff instructed the operator to run `clear_cache` and confirm
~119 GiB free before another large CUDA model boot. On this build that
instruction cannot be satisfied and does not need to be.

After `ds4-server` exits cleanly — no process in `pgrep`, none in `nvtop` —
`MemAvailable` sits at **14 GiB** and stays there. Sampled every 5 s for a
minute it did not move. The full `/proc/meminfo` accounts for only about
**17.7 GiB** of the 127.5 GiB total:

```text
MemFree 6.38  Cached 9.77  AnonPages 0.77  Slab 0.50  Buffers 0.30   (GiB)
largest process RSS: 0.6 GiB   nvtop processes: []
```

The missing ~110 GiB is held by the NVIDIA kernel module and appears in no
`/proc/meminfo` counter. **`clear_cache` is `drop_caches`, which frees page
cache — and there is only 9.77 GiB of page cache to free.** It cannot recover
this memory.

It also does not need to. The retained pool is handed straight back to the next
CUDA context: the phase-B server booted normally, in **230 s**, with
`MemAvailable` still reporting 14 GiB. The earlier handoff's advice was correct
for the *pre-aligned-artifact* build, where the weights were page-cache-resident
and `drop_caches` did move them.

**The correct precondition for a restart is that no other `ds4-server` process
is running.** A readiness gate that waits for `MemAvailable` to recover blocks
forever; `run_phase.sh` in `harness/` was changed to gate on the process and
treat free memory as advisory only.

### 6. `nvtop`'s per-process field name

The memory sampler must read `processes[].gpu_mem_bytes_alloc`. There is no
`gpu_memory_usage` key in nvtop 3.3.2's `-s` JSON, and `nvidia-smi` reports
`memory.used` as `N/A` on GB10, so a sampler that looks for either silently
records zero.

### 6. The pre-existing memory manifest is stale, and was left alone

`manifests/spark-memory-256k.json` records `cuda_allocation_gib: 12.70` and
concludes that "no GPU tool shows the model's footprint on this configuration".
That was true before the aligned-artifact tier landed. On this build the weights
are CUDA-owned and `nvtop` reports 103.62 GiB. The old file is left untracked
and unmodified as the record of that earlier configuration;
`manifests/spark-serving-256k.json` is the current one and says so explicitly.

## Artifacts

| What | Path |
|---|---|
| harness, orchestrator, samplers | `scratch/matrix/*.py`, `scratch/matrix/*.sh` |
| per-boot raw records | `scratch/matrix/<run-id>/bench/cells.jsonl` |
| per-boot server logs | `scratch/matrix/<run-id>/server.log` |
| per-boot memory samples | `scratch/matrix/<run-id>/mem.tsv` |
| API validation evidence | `scratch/matrix/A-plain-256k-c1/api/` |
