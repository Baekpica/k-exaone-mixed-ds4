# K-EXAONE × ds4 DGX Spark handoff — 2026-08-07 16:26 KST

Snapshot time: **2026-08-07 16:26:29 KST (+0900)**

Host: `thinkstationpgx-8abc` — NVIDIA GB10 (`sm_121`), 121.6 GiB unified RAM

Scope of this run: finish checklist items **1–3 only**. Items **4–6 were not
run** and are handed off below.

## Executive state

| Item | State at this snapshot |
|---|---|
| ds4 branch | `feature/exaone-model-loader` |
| ds4 HEAD | `8c45d39956f0edcc88834d9ec93dd026ff32f69d` |
| ds4 worktree | clean; 11 commits ahead of `origin/feature/exaone-model-loader` |
| Model | mixed-quant v1, 85.56 GiB, 3 shards |
| Server | stopped; nothing listening on `:8001` |
| MTP | implemented, exact target verification, automatic quench, default **off** |
| 64-token identity | pass: plain 64 = MTP 64, `mismatch=-1` |
| MTP result on GB10 | 12/12 drafts accepted, but slower; auto-quenched after cycle 12 |
| Memory after test exit | no CUDA process, but NVIDIA driver still retains about 90 GiB UMA |

The code commit for this slice is:

```text
8c45d39 exaone: add target-verified integrated MTP
```

## Checklist disposition

1. **Done — server stopped and UMA/worktree state checked.** The run began
   after the operator's `clear_cache`, with 115 GiB free / 118 GiB available
   and no `nvtop` process. The server remained stopped throughout this slice.
2. **Done — existing speculative paths and K-EXAONE `blk.48` audited.** The
   trained block is already embedded in the same GGUF; no support GGUF is
   needed. The official vLLM implementation confirmed input order
   `[enorm(embed(x[p+1])), hnorm(h[p])]`, followed by `nextn.eh_proj`, and the
   generic MTP scheduling contract confirmed that this pair runs at position
   `p+1`, not `p`.
3. **Done — target-verified MTP and auto-quench implemented and validated.**
   Exact greedy identity passed on the GB10. Runtime measurements showed a
   loss, so quenching fired as designed and MTP remains opt-in.
4. **Not run — plain/MTP context and concurrency matrix.** See the handoff
   section; native batched serving currently disables MTP.
5. **Not run — Model Card edit/upload/remote verification.** The current
   `reports/MODEL_CARD.md` is stale and still says ds4 cannot serve this model.
6. **Not run — final 256K server restart and API validation.** The exact command
   is recorded below.

## What item 3 added

### Integrated MTP graph

- Opt-in flags: `--exaone-mtp` and `--exaone-mtp-timing`.
- The same mixed GGUF supplies dense Q8_0 `blk.48`; a separate draft model was
  deliberately rejected because it would duplicate weights and startup work.
- Additional runtime state is small: one 128-row private f16 KV ring
  (**0.50 MiB**), one concat row, and one host logits row.
- MTP input is the trained ordering:
  `enorm(embed(x[p+1])) || hnorm(target_hidden[p])`.
- The decoder position is explicitly shifted to `p+1`. A unit test fixes both
  `0 -> 1` and `262142 -> 262143`, preventing the RoPE/KV off-by-one from
  returning.
- The MTP block runs its dense attention/MLP and then
  `nextn.shared_head_norm` plus the base model output head.

### Exact target verification

The first ordinary target decode seeds a proposal. Each subsequent speculative
cycle sends `[first_target_token, draft]` through a two-row target pass:

1. compute the target argmax from row 0;
2. accept the draft only when its token ID exactly equals that argmax;
3. on rejection, commit only row 0 and keep its logits;
4. on acceptance, commit both rows and retain row 1 logits;
5. update private MTP KV only along the target-verified prefix.

The target checkpoint never exposes an unverified draft. A rejected row may
temporarily occupy the next target KV slot, but the following ordinary decode
overwrites that exact slot before it can become visible.

### Automatic loss quench

- Warm-up: 12 verifier cycles.
- Comparison: accumulated MTP seed/speculative work against ordinary target
  decode time for the same number of committed tokens.
- Hysteresis: quench only when measured MTP work is more than 3% slower.
- Scope: session-local; once quenched, later tokens use plain target decode.
- Reset points: prompt sync, invalidate, and rewind.
- Diagnostic escape hatch: `DS4_EXAONE_MTP_NO_QUENCH=1` keeps MTP on. This is
  for measurement only, not the serving default.

Speculation is used only for greedy generation. Requests must explicitly set
`temperature: 0`; sampled requests use the normal path.

## Verification evidence

### Build and unit/kernel tests

```text
make cuda-spark                                      PASS, exit 0
make -j4 ds4 ds4-server tests/test_exaone_mtp_policy \
  tests/test_exaone_mtp_identity ds4_cpu.o CUDA_ARCH=sm_121
                                                     PASS
./tests/test_exaone_mtp_policy                       PASS
./tests/test_exaone_kernels                          PASS x3
git diff --check                                     PASS
```

The earlier stale-binary result `kv ring store (wrap) mismatched halfs=32` did
not reproduce after rebuilding with the current `sm_121a` objects. All three
runs reported `mismatched halfs=0`, and all other kernel checks passed,
including QK-norm + NeoX RoPE at position 262143.

Logs:

| Log | SHA-256 |
|---|---|
| `scratch/mtp-cuda-spark-build.log` | build exit recorded as 0 in adjacent `.exit` file |
| `scratch/mtp-kernels-run1.log` | `37fa74401f13a2e6f3e9cd0e6814a3349aad323bace9464f9f03fd044dac691d` |
| `scratch/mtp-kernels-run2.log` | same hash |
| `scratch/mtp-kernels-run3.log` | same hash |
| `scratch/mtp-identity-64.log` | `8f8940279d8dfbf716a77687b549a8bc004a0f2909738fbb1f053d4433cf5a66` |

### Real-model greedy identity and quench

Command:

```bash
cd /home/sunghoon/workspace/ds4-exaone/ds4
./tests/test_exaone_mtp_identity \
  ../models/K-EXAONE-236B-A23B-Mixed-Quant-GGUF/K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  64
```

Result:

```text
exaone MTP greedy identity: plain=64 MTP=64 mismatch=-1
MTP verifier: cycles=12 accepted=12 quenched=1
plain: 5.122 s, 12.494 tok/s
MTP:   5.726 s, 11.178 tok/s
```

The structured integer prompt produced 100% draft acceptance, so draft quality
was not the problem. The verifier/draft work itself was slower:

```text
measured MTP work: 2429.911 ms
target-only estimate: 1957.166 ms
MTP stats net_saved_ms: -444.293
```

The 64-token end-to-end rate was about **10.5% lower** with the MTP attempt
included. This matches the reference fork's warning that single-stream MTP can
lose on Spark. Therefore:

> **Decision: keep K-EXAONE MTP default OFF.** Use `--exaone-mtp-timing` only
> for explicit experiments until a different verifier/draft schedule shows a
> measured gain.

## GPU Mem versus Host Mem

The user-requested split was observed with `nvtop`, `free`, process RSS, and
`btop`.

### During startup

| Phase | `nvtop` process GPU Mem | Host/process observation |
|---|---:|---|
| clean start | 0 | 115 GiB system free |
| aligned artifact build | 34.26 GB | RSS rose 6.8 -> 32.7 GiB, almost all `RssFile` |
| expert promotion | 50.2 -> 96.49 GB | source/model pages rose, then were swept |
| ready, two ctx-127 sessions | **96.82 GB = 90.17 GiB** | RSS 4.56 GiB; system free 22 GiB |

Runtime logs account for the resident model as:

```text
39.09 GiB aligned artifacts + 45.39 GiB raw CUDA cache payload
```

The identity harness used two `ctx=127` sessions, each with only 0.02 GiB KV
and 0.10 GiB workspace. It therefore should not look like the previous 256K
server in `nvtop`:

- small-context identity peak: about **90.17 GiB GPU Mem**;
- prior 256K server: about **103.523 GiB GPU Mem**;
- difference: primarily the measured **12.02 GiB 256K target KV**.

This resolves the apparent "model did not fill 100G" discrepancy: 100+ GiB is
expected only after the 256K KV allocation, not for a 127-token test context.

### After process exit: driver-held UMA remains

At the snapshot, `nvtop` had no process and 0% GPU utilization, but:

```text
btop: Used 92.9 GiB, Cached 9.72 GiB, Free 19.7 GiB
/proc/meminfo: AnonPages 0.35 GiB, Cached 9.7 GiB
largest process RSS: codex app-server about 242 MiB
```

No user process owns the missing ~90 GiB. This is the NVIDIA 595.71.05 UMA
release regression already observed on this host, not ds4 page cache. Before
another large CUDA model boot, the operator must run the interactive-shell
`clear_cache` alias and verify free memory. Codex did **not** invoke that alias.

## Known limitations of this MTP slice

1. **Cold MTP KV.** Private MTP KV begins at the first generation decode; the
   prompt is not replayed through `blk.48`. Its attention window grows from 1
   to 128 rows. This keeps the slice small but may reduce acceptance on less
   regular prompts.
2. **No native-batch MTP.** `ds4-server --batched-session N` deliberately
   disables speculative decoding. Concurrency 2/4/8 MTP cells cannot be called
   MTP results until this is implemented; current rows would silently be
   target-only.
3. **Greedy only.** MTP does not run for nonzero-temperature sampling.
4. **Session-scoped policy.** Quench evidence resets on prompt sync/rewind and
   does not persist across sessions.
5. **Default off.** Even with 100% acceptance, this implementation lost
   throughput on the measured workload.
6. **Driver cleanup is external.** NVIDIA UMA can remain after a clean process
   exit. `nvtop` process disappearance is not sufficient proof of reclaimed
   system memory.
7. **No disk KV for this path.** Keep K-EXAONE serving on in-memory KV.

## Remaining item 4 — benchmark matrix, not executed

Required matrix:

```text
context:     2K, 4K, 8K, 16K, 32K, 64K, 128K, optional 256K
concurrency: 1, 2, 4, 8
mode:        plain, MTP
output:      one raw JSON record per run plus immutable run metadata
```

Before running it:

1. run `clear_cache` and confirm approximately 119 GiB free;
2. pin commit `8c45d39956f0edcc88834d9ec93dd026ff32f69d`;
3. use explicit `temperature: 0` for every request;
4. record `nvtop` GPU Mem, `free -b`, process RSS, prefill tok/s, decode tok/s,
   acceptance, quench state, prompt/decode token counts, and exact command;
5. preserve raw responses and timing JSON without post-processing them in
   place.

Important capability gate: current MTP is valid only at concurrency 1. For
concurrency 2/4/8, first add an MTP-aware batched scheduler or explicitly mark
those matrix cells **unsupported**. Do not publish target-only batched results
under an MTP label.

Single-stream comparison flags are:

```text
plain: no MTP flag
MTP:   --exaone-mtp-timing
```

## Remaining item 5 — Model Card, not executed

Update `reports/MODEL_CARD.md` before any upload. At minimum replace the stale
"ds4 cannot serve this yet" and "not validated on DGX Spark" statements with:

- GB10 resident model breakdown and 256K KV budget;
- `nvtop` GPU Mem versus Host/RSS measurement caveat;
- exact ds4 build and serving command;
- OpenAI API behavior already validated earlier;
- MTP identity, acceptance, throughput loss, auto-quench, and default-off
  decision;
- native-batch MTP and NVIDIA UMA cleanup limitations.

Upload only the README after review:

```bash
cd /home/sunghoon/workspace/ds4-exaone/k-exaone-mixed-ds4
hf upload Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF \
  reports/MODEL_CARD.md README.md \
  --repo-type model \
  --commit-message "docs: add DGX Spark ds4 serving results"
```

Then download the remote README to a fresh temporary directory and compare it
byte-for-byte with `reports/MODEL_CARD.md`. Do not touch or re-upload weights.

## Remaining item 6 — final 256K serving, not executed

Precondition:

```bash
clear_cache
free -h
nvtop -s
```

Expected clean state is no CUDA process and roughly 119 GiB free. Then:

```bash
cd /home/sunghoon/workspace/ds4-exaone/ds4
make cuda-spark

./ds4-server \
  -m ../models/K-EXAONE-236B-A23B-Mixed-Quant-GGUF/K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  --cuda \
  -c 262144 \
  --host 0.0.0.0 \
  --port 8001
```

This is intentionally plain/default-off serving. For a diagnostic MTP run,
add `--exaone-mtp-timing` and send greedy requests only.

This server does not expose llama.cpp-style `/health` or `/props`. Readiness
must be verified with both the model endpoint and a real completion:

```bash
curl -sS http://127.0.0.1:8001/v1/models

curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "K-EXAONE-236B-A23B",
    "messages": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }'
```

For streaming, repeat with `"stream": true` and `curl -N`; confirm SSE chunks,
`reasoning_content` separation, stop reason, and no state contamination across
four requests. If using `--batched-session 4`, remember that MTP is disabled.

## Repositories and artifacts

| Purpose | Path / revision |
|---|---|
| ds4 | `/home/sunghoon/workspace/ds4-exaone/ds4` @ `8c45d39956f0edcc88834d9ec93dd026ff32f69d` |
| mixed-quant project | `/home/sunghoon/workspace/ds4-exaone/k-exaone-mixed-ds4` |
| llama.cpp reference | `/home/sunghoon/workspace/ds4-exaone/llama.cpp` @ `6a32c29a746a2e44de463de647f9f6661eb5086b` |
| donor/reference fork | `/home/sunghoon/workspace/ds4-exaone/ds4-on-spark` @ `eed00a5648bbfc2dc25db7edd64a8d4f376974c4` |
| model shards | `/home/sunghoon/workspace/ds4-exaone/models/K-EXAONE-236B-A23B-Mixed-Quant-GGUF/` |
| build/test logs | `/home/sunghoon/workspace/ds4-exaone/scratch/` |

Current toolchain observed here: CUDA compiler 13.3, NVIDIA driver 595.71.05.
The untracked file `manifests/spark-memory-256k.json` pre-existed this handoff
and was deliberately left untouched.
