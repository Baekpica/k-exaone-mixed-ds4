#!/usr/bin/env bash
# Capture the exact build environment and pinned revisions.
# Writes manifests/build-manifest.json, host-sm120.json and source-model.json.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p manifests

LLAMA_SHA=$(git -C llama.cpp rev-parse HEAD)
LLAMA_TAG=$(git -C llama.cpp describe --tags 2>/dev/null || echo unknown)
DS4_SHA=$(git -C ds4 rev-parse HEAD)
DS4_BASE=$(git -C ds4 rev-parse upstream/main 2>/dev/null || echo unknown)
SELF_SHA=$(git rev-parse HEAD 2>/dev/null || echo uncommitted)

jq -n \
  --arg llama_sha "$LLAMA_SHA" --arg llama_tag "$LLAMA_TAG" \
  --arg ds4_sha "$DS4_SHA" --arg ds4_base "$DS4_BASE" --arg self "$SELF_SHA" \
  --arg nvcc "$(nvcc --version | tail -1)" \
  --arg gcc "$(gcc --version | head -1)" \
  --arg cmake "$(cmake --version | head -1)" \
  --arg driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  --arg date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    generated_utc: $date,
    converter_repo: {commit: $self, url: "https://github.com/Baekpica/k-exaone-mixed-ds4"},
    llama_cpp: {commit: $llama_sha, describe: $llama_tag,
                url: "https://github.com/ggml-org/llama.cpp",
                build_dir: "build-sm120",
                cmake_flags: ["-DGGML_CUDA=ON","-DCMAKE_CUDA_ARCHITECTURES=120","-DLLAMA_CURL=OFF","-DCMAKE_BUILD_TYPE=Release"]},
    ds4: {fork: "https://github.com/Baekpica/ds4", commit: $ds4_sha,
          upstream: "https://github.com/Entrpi/ds4", upstream_base: $ds4_base,
          branch: "feature/exaone-model-loader"},
    toolchain: {nvcc: $nvcc, gcc: $gcc, cmake: $cmake, driver: $driver},
    build_targets: {"cuda-sm120": "RTX PRO 6000 development / microbenchmark (built here)",
                    "cuda-sm121": "DGX Spark release target (defined, not built on this host)"}
  }' > manifests/build-manifest.json

jq -n \
  --arg uname "$(uname -a)" \
  --arg cpu "$(lscpu | sed -n 's/^Model name: *//p' | head -1)" \
  --arg threads "$(nproc)" \
  --arg mem "$(free -b | awk '/^Mem:/{print $2}')" \
  --argjson gpus "$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | jq -R -s 'split("\n")|map(select(length>0))|map(split(", ")|{index:.[0],name:.[1],memory:.[2],compute_cap:.[3]})')" \
  --arg topo "$(nvidia-smi topo -m 2>/dev/null | head -6 | tr '\n' '|')" \
  '{role: "development host (sm_120) -- not a release serving target",
    uname: $uname, cpu: $cpu, threads: ($threads|tonumber),
    memory_bytes: ($mem|tonumber), gpus: $gpus, topology: $topo,
    storage: {"/workspace": "MooseFS FUSE, ~3TB usable, 1.1 GB/s write / 685 MB/s read",
              "/": "local overlay, ~1TB usable, 3.8 GB/s write"}}' \
  > manifests/host-sm120.json

jq -n --arg sha "$(cut -d' ' -f1 logs/sha-bf16.txt 2>/dev/null || echo pending)" \
  '{
    hf_model: {repo: "LGAI-EXAONE/K-EXAONE-236B-A23B",
               revision: "61e6d578eb102b578e5704e2916ac841df9eca0a",
               files: 109, bytes: 474281148160},
    hf_gguf: {repo: "LGAI-EXAONE/K-EXAONE-236B-A23B-GGUF",
              revision: "5bd0394e4f42c00df63e207b9c434387523a6b77",
              file: "K-EXAONE-236B-A23B-BF16.gguf", bytes: 474281148160,
              sha256_official: "73be2da8653976df036bf9b6466b011f86cb10f78bab30a47025638ec999d3f8",
              sha256_verified_local: $sha},
    license: {name: "k-exaone", file: "LICENSE",
              redistribution: "sec 2.1 requires the agreement to travel with any derivative and the derivative name to begin with K-EXAONE; sec 2.2 requires a separate agreement with the Licensor for commercial distribution to third parties"}
  }' > manifests/source-model.json

echo "wrote manifests/build-manifest.json manifests/host-sm120.json manifests/source-model.json"
