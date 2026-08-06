#!/usr/bin/env python3
"""Fetch safetensors headers via HTTP range requests -> exact shapes/dtypes
without downloading 442 GiB of weights."""
import json, sys, concurrent.futures as cf
from huggingface_hub import get_hf_file_metadata, hf_hub_url
import requests

REPO = "LGAI-EXAONE/K-EXAONE-236B-A23B"
REV = "61e6d578eb102b578e5704e2916ac841df9eca0a"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/k-exaone-mixed-ds4/manifests/st-headers.json"

shards = sorted({v for v in json.load(open(
    "/workspace/models/K-EXAONE-236B-A23B/model.safetensors.index.json"))["weight_map"].values()})

sess = requests.Session()
from huggingface_hub import get_token
tok = get_token()
HDRS = {"Authorization": f"Bearer {tok}"} if tok else {}


def one(shard):
    url = hf_hub_url(REPO, shard, revision=REV)
    # 1) first 8 bytes = little-endian u64 header length
    r = sess.get(url, headers={**HDRS, "Range": "bytes=0-7"}, timeout=60)
    r.raise_for_status()
    n = int.from_bytes(r.content[:8], "little")
    # 2) the JSON header itself
    r = sess.get(url, headers={**HDRS, "Range": f"bytes=8-{8 + n - 1}"}, timeout=120)
    r.raise_for_status()
    return shard, json.loads(r.content[:n].decode("utf-8"))


tensors = {}
with cf.ThreadPoolExecutor(16) as ex:
    for i, (shard, hdr) in enumerate(ex.map(one, shards), 1):
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            tensors[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "shard": shard}
        print(f"\r  {i}/{len(shards)} shards", end="", flush=True)
print()
json.dump(tensors, open(OUT, "w"), indent=1)
print(f"wrote {len(tensors)} tensor headers -> {OUT}")
