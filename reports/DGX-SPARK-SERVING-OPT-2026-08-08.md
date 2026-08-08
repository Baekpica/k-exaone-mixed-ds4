# K-EXAONE × ds4 DGX Spark — serving optimization, and a defect found on the way

Continues `DGX-SPARK-ITEMS-4-6-2026-08-07.md`, which closed checklist items 4–6
and left a prioritized fix list. This round took the top item and found that it
was gated on a correctness bug nobody had seen.

Host: `thinkstationpgx-8abc` — NVIDIA GB10 (`sm_121`), 121.6 GiB unified RAM,
driver 595.71.05, CUDA 13.3, Linux 6.17.

ds4: `feature/exaone-model-loader` @ `920427ac124078af021a0736792d2115b1d00bc2`,
built with `make cuda-spark`. Model: mixed-quant v1, 85.56 GiB, 3 shards.

## Disposition

| Item | State |
|---|---|
| Sliding KV ring sized for the prefill chunk | done — correctness fix, was not on the list |
| Partial prefix reuse (previous list, item 1) | done — 24× on multi-turn |
| Batched-session workspace sharing (item 3) | not started, now the top item |
| Decode row amortization (item 2) | **withdrawn** — see *Corrections* |
| MTP KV ring warm-up (item 4) | unchanged |

## 0. Lineage: this branch is not on the optimized fork

Worth stating first, because it frames what "reference `Entrpi/ds4`" can mean.

`Baekpica/ds4 feature/exaone-model-loader` sits on the `antirez/ds4` main
lineage. `Entrpi/ds4`'s release line is the `batched-serving` branch, currently
`v0.5.6`; the two diverged at `e16ead1` (2026-05-29) and it is **569 commits
ahead** of us on its side, 233 on ours.

So the serving machinery `ds4-on-spark` advertises — continuous batching,
prefix caching with disk-persisted KV banks, DSpark speculative decode — is not
in this branch and cannot be picked up by a merge. Rebasing 569 commits would
mean rewriting the exaone loader against a substantially different engine.

What upstream *is* good for is technique and negative results, and both paid off
this round: their SWA sizing rule was the fix for the defect below, and their
MTP write-up killed a large piece of work before it started.

## 1. The sliding KV ring was the width of the window

`exaone_graph_layer_kv_cap()` allocated `DS4_N_SWA` (128) positions per sliding
layer. Prefill runs 2048-token chunks, and `exaone_graph_layer()` stores every
row's KV *before* any row attends:

```c
ds4_gpu_exaone_kv_store_tensor(g->layer_kv[il], k, v, kv_dim, n_tok, pos0, kv_cap);
/* ... then ... */
ds4_gpu_exaone_attention_prefill_tensor(heads, q, g->layer_kv[il], n_tok, pos0, ...);
```

With `slot = position % kv_cap`, a 2048-row chunk into a 128-slot ring leaves
only the chunk's last 128 positions. Row `t` then reads `[t-127, t]`, and those
slots belong to positions from the end of the chunk.

**The rule already exists in this codebase.** `metal_graph_raw_cap_for_context()`
sizes the DeepSeek/GLM SWA cache as `raw_window + prefill_cap`, with the comment
*"During batched prefill the SWA cache must hold the current ubatch plus the
previous logical window."* The exaone port did not carry it over.

### Why the existing kernel test could not see it

`tests/test_exaone_kernels.c::test_attention_prefill` compares GPU attention
against a CPU mirror of **the same ring**. A ring that has dropped a position
agrees with itself, so both sides are wrong identically and the check passes. It
also skipped the leading rows outright:

```c
const uint32_t t0 = sliding ? (n_tok > kv_cap ? n_tok - kv_cap : 0u) : 0u;
```

The new check compares a chunked prefill against the token-at-a-time path, which
is the real contract — token-at-a-time is ground truth because each step only
reads the window it has just written.

```text
sliding ring == window (undersized)    bad_rows=199 expected=199  ok
sliding ring == window + chunk         bad_rows=0                 ok
```

199 of 200, not 72: the whole chunk is stored before any attention runs, so only
the final row's window is still resident. At chunk 2048 that is **all but the
last row of every chunk**, across 36 of the 48 layers.

### What it did to output

The `D-final-256k` continuation fixtures fed ~6 900 tokens of Manzoni's
*I promessi sposi* — Italian prose about Renzo's exile and don Rodrigo — and
asked for a summary. The pre-fix build answered:

> 이 문서는 **"로고스(Logos)"**라는 개념을 중심으로 전개됩니다. 로고스는 우주와
> 인간 존재의 근본적인 질서와 원리를 상징합니다.

and, on a second passage:

> 이 문서는 **"파스타 소스"**를 **"빵 반죽"**에 넣는 아이디어를 설명하고
> 있습니다. 이 방법은 "파스타 소스"를 "빵 반죽"에 넣는 것입니다.

The 12 full-attention layers were intact, which is enough to establish "prose,
Italian" and nothing more. Post-fix, same fixture:

> 이 문서는 이탈리아 문학 작품의 한 부분으로 보이며, 주로 **도니 로드리고**(Don
> Rodrigo)라는 인물의 심리적 갈등과 복잡한 인간관계를 다루고 있습니다 … 특히
> **몬차**에서의

Named-entity recall against the actual text. A planted-needle check confirms it
objectively: an exact string at 10 %, 50 % and 90 % depth of a 6 971-token
document is returned at **3/3** depths.

**Why nobody caught it.** Every API validation prompt is under 128 tokens
("대한민국의 수도는?"), where the ring never wraps. The exaone GPU-vs-CPU
harnesses take tokens on `argv`, so they were run with a handful. And the
published quality numbers were measured on llama.cpp, not on ds4.

### The MTP verify hit the same ring

The two-row target verify runs `exaone_graph_prefill_chunk(..., toks, 2u, pos, true)`.
After both rows are stored, the ring holds `[pos-126, pos+1]`; row 0 at `pos`
needs `[pos-127, pos]`. Slot `(pos-127) % 128 == (pos+1) % 128` — one key of 128
comes from a *future* position. So past depth 128 the accept decision was not
computed from the target's true argmax, and "speculation cannot change greedy
output" did not hold. The widened ring fixes it as a side effect.

### The fix

`kv_cap = DS4_N_SWA + prefill_cap`, clamped to `ctx_size`. The MTP block's
private ring passes `prefill_cap = 0` — it is stepped one accepted position at
a time and needs the window only.

| | before | after |
|---|---:|---:|
| sliding layer ring | 128 positions | 2 176 |
| KV at `-c 262144` | 12.02 GiB | 12.30 GiB |
| resident, `nvtop` | 103.62 GiB | 103.95 GiB |
| boot to `listening` | 226 s | 231–236 s |
| cold prefill, ~7K tokens | 136–166 s | 137–166 s |

+288 MiB of ring for no measurable time cost. Under `--batched-session N` it is
+288 MiB **per slot**, which makes the workspace-sharing item below more
valuable, not less.

## 2. Partial prefix reuse

The previous report's finding 2: `ds4_session_sync_internal` reused KV only when
the prompt contained the *entire* checkpoint, so a continuation sharing 6 984 of
7 086 tokens — 98.6 % — discarded all of it.

The blocker was never the policy, it was the ring. A prefill resuming at `start`
reads `[start-window+1, start]`, and the ring holds only the last `kv_cap`
positions written. With `kv_cap == window` the only safe rewind is zero. With
`kv_cap == window + chunk` it is about 2 049 tokens.

So the bound is a property of the allocation, and `ds4_session_exaone_rewind_span()`
reads it back from the graph rather than restating it:

```c
const int live = s->checkpoint_valid ? s->checkpoint.len : 0;
if ((uint32_t)live <= narrowest) return live;   /* nothing evicted yet */
if (narrowest <= DS4_N_SWA) return 0;
return (int)(narrowest - DS4_N_SWA + 1u);
```

and the sync path is now:

```c
const int live = s->checkpoint.len;
const int span = ds4_session_exaone_rewind_span(s);
start = ds4_session_common_prefix(s, prompt);
if (start > 0 && start == prompt->len) start = prompt->len - 1;
if (live - start > span) start = 0;
```

The server asks the engine for the bound instead of duplicating the rule, which
keeps `cached_tokens` honest — the number the client sees is the number that
actually happened.

### Measured

Same fixtures as `D-final-256k`, same 256K server configuration:

| Turn | Prompt tokens | TTFT before | TTFT after | Reused |
|---|---:|---:|---:|---:|
| 1 — cold, ~7K document + question | 6 978 | 166.2 s | 165.7 s | 0 |
| 2 — history + the reply + a follow-up | ~7 085 | 143.9 s | **5.9 s** | **6 992** |
| 3 — a different document, cold | 6 725 | 136.5 s | 137.0 s | 0 |

**24×** on turn 2. Turns 1 and 3 are unchanged, which is the half of the result
that matters for correctness: an unrelated prompt is not falsely matched onto a
live session. The server log shows turn 3 scoring `common=2` and taking the cold
path.

### Reuse does not change the answer

The thing worth proving about a reuse change is that the reused path answers what
a cold prefill answers. `benchmarks/warm_cold_equiv.py` sends a request warm,
evicts the session with an unrelated long prompt, and sends the byte-identical
request again cold:

| | prompt tokens | reused | elapsed |
|---|---:|---:|---:|
| warm | 3 508 | 3 455 | 5.5 s |
| cold | 3 508 | 0 | 67.8 s |

`identical=True`, `char_similarity=1.0000`, 12.3× faster.

This was not a foregone conclusion — a resumed prefill evaluates its tail as a
short chunk where a cold prefill evaluates it inside a 2048-row one, and the
GEMM tiling differs. It came out byte-identical on this fixture; the test
reports similarity rather than asserting equality so a future regression shows
its size instead of just failing.

## 3. Validation

| Check | Result |
|---|---|
| `tests/test_exaone_kernels` | all pass, including both new residency checks |
| `test_layer_pack`, `test_engine_mgpu_placement`, `test_gpu_args`, `test_gpu_args_cli.sh`, `test_sampling`, `test_exaone_mtp_policy` | pass |
| `make test` full target | blocked — needs `ds4flash.gguf`, absent on this host (pre-existing) |
| API validation, 16 checks | ALL PASS |
| needle recall, 3 depths | 3/3 |
| multi-turn continuation | reuse confirmed, cold turns unchanged |
| warm vs cold equivalence | byte-identical |
| resident at 256K | 103.95 GiB / 121.6 GiB |

## 4. Corrections to the previous report

**Finding 0 claimed that amortizing rows in the decode graph would turn MTP from
break-even into roughly a 25 % gain. Withdraw that.**
`Entrpi/ds4-on-spark/docs/MTP_PARITY_GAP.md` §4 reports building exactly that —
a weight-shared exact verifier, `metal_graph_verify_shared_exact` — and
measuring it bit-exact but **perf-neutral-to-negative** single-stream. GB10 A/B,
prose: no-MTP 19.64 > sequential 15.58 > shared 13.02 tok/s. The reason is the
one our own report gave for concurrency and did not apply here: the two draft
rows route to disjoint experts, so a shared sweep reads the union and saves no
bandwidth. Upstream's conclusion — the amortization axis is batched serving, not
a better single-stream verifier — should be ours too.

**Finding 1's "~30× headroom" needs a qualifier.** The 3 %-of-bandwidth figure is
the *depth-dependent* KV read. The depth-independent half is roughly consistent
with streaming the active weights, and `ds4-on-spark`'s roofline analysis puts
the donor's CUDA decode at ~95 % of the bandwidth roofline at steady state. The
headroom is real but it is in the attention path, not in decode as a whole.

**Finding 2's caveat about sliding layers was right and was the whole story.**
It flagged that "a rewind invalidates that assumption and needs checking". It
did — and checking it turned up the prefill defect.

## 5. Next

1. **Share the graph workspace across batched sessions.** `--batched-session N`
   costs `N × (KV + 1.60 GiB)`. `share_session_prefill_workspace` exists but is
   wired only into `metal_graph_alloc_raw_cap`; `exaone_graph_alloc` returns
   before that and never consults it, and exaone has its own batched decode path
   (`ds4_sessions_eval_batch_exaone`). The `b_*` batch tensors are the 1.60 GiB.
   Prefill is serialized under the server's `inference_mu`, which is the
   precondition the DeepSeek/GLM path already relies on. Worth ~11 GiB at 8
   slots, and now also recovers 8 × 288 MiB of the widened ring.
2. **Warm the MTP KV ring from the prompt**, to raise acceptance from the 34 %
   seen at depth toward the 69 % seen shallow. Unchanged in priority.
3. **Re-measure the throughput matrix.** Every number in
   `DGX-SPARK-ITEMS-4-6-2026-08-07.md` was taken against the defective prefill.
   The timings are still valid — the defect changed what was computed, not how
   much — but the depth models should be re-fit against the corrected engine
   before anything else is built on them.

## Artifacts

| What | Path |
|---|---|
| post-fix 256K run (continuation, recall, API) | `scratch/matrix/E-swa-fix-256k/` |
| warm-vs-cold equivalence | `scratch/matrix/F-warm-cold-256k/` |
| harnesses | `benchmarks/recall_test.py`, `benchmarks/warm_cold_equiv.py` |
| pre-fix run, for comparison | `scratch/matrix/D-final-256k/` |
