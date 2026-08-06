# k-exaone-mixed-ds4

Mixed-precision GGUF builds of **K-EXAONE-236B-A23B** that fit a single
128 GB device, plus the ds4 engine work to serve them.

Quantization is assigned by what each tensor *does* rather than by a global bit
budget: the router, norms, attention, shared expert and dense layer 0 stay at
high precision, and the compression comes almost entirely from the routed
experts, which hold ~64 % of the parameters. All 128 routed experts, the shared
expert, and the original 1-layer MTP block are preserved — nothing pruned,
nothing merged.

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
- **ds4** — loads a K-EXAONE GGUF and runs a **CPU reference forward**:
  architecture detection, hparam validation, tensor binder, layout validation,
  Q3_K dequantization, GQA attention with QK-norm, the LLLG schedule, and the
  sigmoid-routed MoE. Validated against llama.cpp on the same model: same greedy
  token, `attn_norm` exact to four decimals, and the residual difference traced
  to 8-bit activation quantization rather than the architecture.
  **No CUDA kernels yet**, no batching, no MTP speculative decode — so run the
  artifacts with llama.cpp for now.
- **DGX Spark** — nothing has run on GB10. Sizing targets it; resident memory,
  `sm_121` correctness and serving throughput are unmeasured. See
  `reports/DGX-SPARK-HANDOFF.md`.

## License

Base model © LG Management Development Institute under the **K-EXAONE AI Model
License Agreement**. Derivative artifacts ship the agreement alongside and are
named beginning with "K-EXAONE", per §2.1. §2.2 reserves commercial
distribution to third parties to a separate agreement with the Licensor; these
artifacts are published as a research artifact only. Code in this repository is
MIT.
