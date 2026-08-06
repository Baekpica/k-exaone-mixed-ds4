#!/usr/bin/env python3
"""Project mixed-quant artifact size from the real BF16 GGUF tensor table.

This supersedes the safetensors-derived estimate in inventory.py: it uses the
actual GGUF tensor names, shapes and source dtypes, and resolves types through
the same code path build_mixed.py emits, so what it reports is what
llama-quantize will produce.
"""
import argparse, json, sys, collections, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(Path(__file__).parent))
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: E402
from build_mixed import emit_patterns  # noqa: E402

GROUPS = [
    (r"^token_embd\.", "embedding"), (r"^output\.weight$", "output"),
    (r"^output_norm\.", "final_norm"),
    (r"\.nextn\.", "mtp"),
    (r"^blk\.\d+\.attn_(q|k)_norm\.", "attention_qk_norm"),
    (r"^blk\.\d+\.attn_norm\.", "attention_norm"),
    (r"^blk\.\d+\.ffn_norm\.", "ffn_norm"),
    (r"^blk\.\d+\.attn_(q|k|v|output)\.", "attention"),
    (r"^blk\.\d+\.(ffn_gate_inp|exp_probs_b)\.", "router"),
    (r"^blk\.\d+\.ffn_(gate|up)_exps\.", "routed_expert_gate_up"),
    (r"^blk\.\d+\.ffn_down_exps\.", "routed_expert_down"),
    (r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.", "shared_expert"),
    (r"^blk\.\d+\.ffn_(gate|up|down)\.", "dense_mlp"),
]
GROUPS = [(re.compile(p), g) for p, g in GROUPS]


def group_of(name, mtp_bid):
    if name.startswith(f"blk.{mtp_bid}."):
        return "mtp"
    for rx, g in GROUPS:
        if rx.search(name):
            return g
    return "other"


def resolve(name, src_type, pats, embd_t, out_t):
    """Mirror llama.cpp: never-quantize checks, then embd/output params, then
    the first matching regex."""
    if name.endswith("_norm.weight") or name.endswith("ffn_gate_inp.weight") \
            or not name.endswith("weight"):
        return src_type
    if name == "token_embd.weight":
        return embd_t.upper()
    if name == "output.weight":
        return out_t.upper()
    for rx, t, _ in pats:
        if re.search(rx, name):
            return t.upper()
    return src_type


def nbytes(ne, qtype):
    n = 1
    for d in ne:
        n *= d
    blck, size = GGML_QUANT_SIZES[getattr(GGMLQuantizationType, qtype)]
    return n // blck * size, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default=str(ROOT / "manifests/bf16-gguf-header.json"))
    ap.add_argument("--recipe", default=str(ROOT / "manifests/quant-recipe-v1.yaml"))
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--out")
    a = ap.parse_args()

    import yaml
    recipe = yaml.safe_load(open(a.recipe))
    pats = emit_patterns(recipe, a.variant)
    r = dict(recipe["rules"])
    vr = (recipe.get("variants") or {}).get(a.variant, {}) or {}
    embd_t, out_t = vr.get("embedding", r["embedding"]), vr.get("output", r["output"])

    d = json.load(open(a.header))
    mtp_bid = d["kv"]["exaone-moe.block_count"] - 1
    rows = []
    agg = collections.defaultdict(lambda: {"n": 0, "elem": 0, "src": 0, "dst": 0, "t": set()})
    for t in d["tensors"]:
        qt = resolve(t["name"], t["type"], pats, embd_t, out_t)
        db, n = nbytes(t["ne"], qt)
        sb, _ = nbytes(t["ne"], t["type"])
        g = group_of(t["name"], mtp_bid)
        rows.append({"name": t["name"], "group": g, "ne": t["ne"],
                     "src": t["type"], "dst": qt, "src_bytes": sb, "dst_bytes": db})
        A = agg[g]
        A["n"] += 1; A["elem"] += n; A["src"] += sb; A["dst"] += db; A["t"].add(qt)

    GI = 2 ** 30
    ts = sum(x["src_bytes"] for x in rows)
    td = sum(x["dst_bytes"] for x in rows)
    te = sum(x["ne"] and __import__("math").prod(x["ne"]) for x in rows)
    print(f"variant: {a.variant}   (source: real BF16 GGUF tensor table, {len(rows)} tensors)")
    print(f"{'group':24s} {'n':>4s} {'params':>15s} {'src GiB':>9s} {'dst GiB':>9s}  types")
    print("-" * 96)
    for g, A in sorted(agg.items(), key=lambda kv: -kv[1]["dst"]):
        print(f"{g:24s} {A['n']:4d} {A['elem']:15,d} {A['src']/GI:9.2f} {A['dst']/GI:9.2f}  "
              f"{','.join(sorted(A['t']))}")
    print("-" * 96)
    print(f"{'TOTAL':24s} {len(rows):4d} {te:15,d} {ts/GI:9.2f} {td/GI:9.2f}")
    lo, hi = recipe["recipe"]["target_size_gib"]
    hard = recipe["recipe"]["hard_limit_gib"]
    got = td / GI
    v = "IN TARGET" if lo <= got <= hi else ("under target" if got < lo else
        ("over target, under hard limit" if got <= hard else "OVER HARD LIMIT"))
    print(f"\ntarget {lo}-{hi} GiB, hard limit {hard} GiB  ->  {got:.2f} GiB  [{v}]")
    print("note: excludes GGUF metadata and per-tensor alignment padding (~0.02%)")
    if a.out:
        json.dump({"variant": a.variant, "total_src_bytes": ts, "total_dst_bytes": td,
                   "projected_gib": got, "n_tensors": len(rows),
                   "by_group": {k: {**{kk: vv for kk, vv in V.items() if kk != "t"},
                                    "types": sorted(V["t"])} for k, V in agg.items()},
                   "tensors": rows}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
