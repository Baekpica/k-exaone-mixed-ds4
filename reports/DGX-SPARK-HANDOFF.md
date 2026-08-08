# DGX Spark (`sm_121`) handoff

Current landing page for K-EXAONE-236B-A23B mixed-quant v1 serving on one
128 GB DGX Spark. Historical plans and dated reports remain in this repository,
but this file is the place to start.

## Current pins

| Component | Pin |
|---|---|
| Engine | [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader) `feature/exaone-model-loader` @ **`b2faf06`** |
| Converter / reports | [`Baekpica/k-exaone-mixed-ds4`](https://github.com/Baekpica/k-exaone-mixed-ds4) |
| Weights | [`Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF`](https://huggingface.co/Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF), v1, 85.56 GiB |
| Minimum correct long-prompt engine | `920427a`; use the current pin above |

The artifact preserves all 128 routed experts in all 47 MoE layers, the shared
expert, dense layer 0, and `blk.48` MTP: 781 tensors, 237.10B parameters.

## Build and serve

```bash
git clone https://github.com/Baekpica/ds4
cd ds4
git checkout b2faf06f1ce2702efa53ac17145b1f56d3fb23b8
make cuda-spark

./ds4-server \
  -m ../models/K-EXAONE-236B-A23B-Mixed-Quant-GGUF/K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1-00001-of-00003.gguf \
  --cuda -c 262144 --host 0.0.0.0 --port 8001
```

Run the server in detached tmux. On this host the CUDA driver retains unified
allocations after exit; gate restarts on `pgrep -x ds4-server`, not on the
`free` output. Readiness is `GET /v1/models` **plus a real completion**.

`GET /v1/models` exposes:

- `k-exaone-236b-a23b` — thinking on by default;
- `k-exaone-236b-a23b-chat` — direct-answer alias.

## Current performance

Cold, disjoint prompts; concurrency 1; 128 generated tokens:

| Frontier | Prefilled | Prefill | Decode | TTFT |
|---:|---:|---:|---:|---:|
| 2K | 1,387 | **269.6 t/s** | 10.75 t/s | 5.14 s |
| 8K | 7,923 | **276.5 t/s** | 10.46 t/s | 28.66 s |
| 32K | 31,300 | **245.0 t/s** | 9.00 t/s | 127.78 s |
| 64K | 64,663 | **207.3 t/s** | 7.27 t/s | 311.90 s |

The full 262,144-token context fits and was previously measured at 103.95 GiB
resident. Cold 256K prefill has not been measured end to end; do not turn the
64K result into a release claim by extrapolation. Multi-turn prefix reuse and
8-stream row batching remain available; the latter measured 18.5 t/s aggregate
on the previous serving round.

Details and evidence: `DGX-SPARK-PREFILL-OPT-2026-08-09.md`.

## Final profile and next work

The request-only Nsight slice, before the final 64-column-tail increment:

| Group | GPU time |
|---|---:|
| IQ2 gate/up D2R | 30.66% |
| Q3 routed down | 23.65% |
| dense/shared Q8 | 14.97% |
| Q4 routed path | 10.77% |
| embeddings | 4.90% |
| prefill attention | 2.62% |
| QK norm/RoPE | 0.60% |

Continue in that order: IQ2, Q3, dense Q8, Q4. QK/RoPE and attention are no
longer first-order targets. Each new path has an in-process rollback switch;
the dated prefill report lists them.

MTP executes and is target-verified, but remains off by default because the
measured verifier path is still a net decode loss. Acceptance alone is not a
performance result: compare total target+draft work and greedy output identity.

## Validation gates already passed

- full `sm_121` build of `ds4-server`, `ds4_test`, and
  `tests/test_exaone_kernels`;
- synthetic CUDA kernel suite and the same suite against the 85.56 GiB model;
- Q3/Q4 generic, bounded, worklist, tail64-off, and worklist-off output parity;
- IQ2 aligned wide-entry parity;
- fixed 4,325-token / 64-output-token response identity;
- bare/chat model-ID behavior and `./ds4_test --server`.

Benchmark prompts must use disjoint corpus slices. Because the engine reuses a
diverged prefix, growing every frontier from the same offset measures
incremental prefill instead of a cold frontier.
