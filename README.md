# k-exaone-mixed-ds4

**A 237 B-parameter model, structurally intact, served from a single 128 GB
DGX Spark — with its full 262 144-token context.**

Mixed-precision GGUF builds of **K-EXAONE-236B-A23B**, plus the ds4 engine work
to serve them on GB10.

Nothing is removed. All **128 routed experts** in all 47 MoE layers, the shared
expert, dense layer 0, and the original 1-layer MTP block are present — 781
tensors, identical to the BF16 source. No pruning, no expert dropping, no layer
truncation, no distillation. Only the storage precision changes, and it is
assigned by what each tensor *does* rather than by a global bit budget: router,
norms, attention, shared expert and dense layer 0 stay high-precision, and the
compression comes almost entirely from the routed experts, which hold ~64 % of
the parameters.

| | |
|---|---:|
| Parameters | 237.10 B (A23B active) |
| BF16 source | 441.63 GiB |
| **v1 artifact** | **85.56 GiB** (5.16×) |
| Context served on one GB10 | **262 144 tokens** |
| Resident at 256K, measured | **103.95 GiB / 121.6 GiB** |

**Artifacts:** [`Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF`](https://huggingface.co/Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF) ·
**Engine:** [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader)

| Variant | Size | Routed gate/up | Routed down | imatrix |
|---|---:|---|---|---|
| **v1** | 85.56 GiB | `IQ2_XXS` | `Q3_K` | yes |
| pilot | 87.84 GiB | `Q2_K` | `Q2_K` | no |

Both verified tensor-by-tensor against the recipe. v1 tracks the official
`Q8_0` build more closely than the pilot *while being smaller* — 0.183 vs 0.139
mean word-agreement over 32 fixtures, and 0.681 vs 0.250 on structured output.

## Layout

```
manifests/    pinned revisions, tensor inventory, size projections,
              per-tensor verification reports, imatrix coverage
converter/    recipe -> llama-quantize driver, verifier, calibration builder
benchmarks/   Phase C fixture harness, router-fixture dump
fixtures/     32 prompts, Q8_0 greedy reference, per-layer router decisions
reports/      model card, quality report, DGX Spark handoff
scripts/      download, build, imatrix, split+upload, handoff bundle
```

## Reproducing

```bash
./scripts/download_sources.sh gguf-bf16          # 441.7 GiB
./scripts/build_mixed_gguf.sh pilot              # no imatrix needed
./scripts/download_sources.sh gguf-q8            # for the imatrix run
python3 converter/tools/build_calibration.py
CHUNKS=775 ./scripts/gen_imatrix.sh
./scripts/build_mixed_gguf.sh v1 /workspace/artifacts/k-exaone-236b.imatrix
```

Every step verifies: `verify_gguf.py` checks each tensor's type against the
recipe, `check_imatrix.py` refuses an importance matrix with unactivated
experts, and `project_from_gguf.py` predicts the artifact size from the source
tensor table before anything is written.

## Status

- **Artifacts** — done, published, verified.
- **ds4** — serves the model end to end on CUDA. The `exaone-moe` family in
  [`Baekpica/ds4`](https://github.com/Baekpica/ds4/tree/feature/exaone-model-loader)
  carries architecture detection, the tensor binder, GQA attention with QK-norm,
  the LLLG sliding-window schedule, sigmoid-routed MoE, `sm_121` CUDA kernels,
  native session batching, and a target-verified MTP path through `blk.48`.
  The CPU reference forward still validates against llama.cpp on the same model:
  same greedy token, `attn_norm` exact to four decimals.
- **DGX Spark** — measured on GB10 / `sm_121`. The 256K server boots in ~4 min
  and sits at **103.95 GiB of 121.6 GiB** resident; the OpenAI-compatible API is
  validated for streaming, thinking-mode `reasoning_content` separation, and
  cross-request isolation. A multi-turn continuation resumes at the point it
  diverges from the live session, so turn 2 of a 7K-token chat costs **5.9 s**
  rather than 143.9 s. Flash-decode keeps deep decode fast (**8.7 t/s at a
  measured 33K**, ~4 t/s extrapolated at full context), and cross-session row
  batching lifts concurrent serving to **18.5 t/s aggregate at 8 streams**.
  MTP runs, is a mild loss (3–25 % by depth), and ships off by default with an
  automatic loss quench.
- **Pin ds4 at or after `d35f0dd`** (the serving-optimization round; `920427a`
  is the minimum for correct long prompts). Older than `920427a`, the
  `exaone-moe` sliding KV ring was sized to the attention window alone while prefill ran 2 048-token
  chunks, so all but the last row of each chunk attended over overwritten
  slots. 36 of 48 layers are sliding, and long-prompt comprehension was badly
  degraded as a result. Short prompts were unaffected, which is why the API
  checks passed throughout. See `reports/MODEL_CARD.md` and
  `reports/DGX-SPARK-ITEMS-4-6-2026-08-07.md`.

## Acknowledgements

The engine is other people's work:
[`antirez/ds4`](https://github.com/antirez/ds4) is the original runtime,
[`Entrpi/ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) is the DGX Spark
port that contributed the `sm_121` target and the aligned-artifact tier this
model depends on for speed, and
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) provides GGUF,
`llama-quantize` and the quant formats that produced these files. The base model
is [`LGAI-EXAONE/K-EXAONE-236B-A23B`](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B).

## License

Base model © LG Management Development Institute under the **K-EXAONE AI Model
License Agreement**. Derivative artifacts ship the agreement alongside and are
named beginning with "K-EXAONE", per §2.1. §2.2 reserves commercial
distribution to third parties to a separate agreement with the Licensor; these
artifacts are published as a research artifact only. Code in this repository is
MIT.
