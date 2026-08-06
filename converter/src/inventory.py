#!/usr/bin/env python3
"""K-EXAONE tensor inventory + mixed-quant size projection.

Maps HF safetensors names -> GGUF tensor names for the exaone-moe architecture,
classifies every tensor into the logical module groups from the work order, and
projects the artifact size for a given recipe variant.

Ground truth for shapes comes from manifests/st-headers.json (fetched via HTTP
range requests). Once the real BF16 GGUF is on disk, verify_gguf.py cross-checks
these projections against the actual tensor table.
"""
import json, re, sys, argparse, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: E402

# ---------------------------------------------------------------- config
CFG = json.load(open(ROOT.parent / "models/K-EXAONE-236B-A23B/config.json"))
N_LAYERS = CFG["num_hidden_layers"]              # 48
N_EXPERTS = CFG["num_experts"]                   # 128
FIRST_DENSE = CFG["first_k_dense_replace"]       # 1  -> layer 0 dense
MTP_BID = N_LAYERS                               # MTP occupies blk.48

# ---------------------------------------------------------------- name mapping
# (regex on HF name) -> (gguf name template, logical group)
# {L} = layer index. Expert tensors collapse into one 3-D GGUF tensor.
MAP = [
    (r"^model\.embed_tokens\.weight$",            "token_embd.weight",              "embedding"),
    (r"^lm_head\.weight$",                        "output.weight",                  "output"),
    (r"^model\.norm\.weight$",                    "output_norm.weight",             "final_norm"),

    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$",          "blk.{L}.attn_norm.weight",      "attention_norm"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", "blk.{L}.ffn_norm.weight",       "ffn_norm"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$",        "blk.{L}.attn_q_norm.weight",    "attention_norm"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$",        "blk.{L}.attn_k_norm.weight",    "attention_norm"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",        "blk.{L}.attn_q.weight",         "attention_q"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",        "blk.{L}.attn_k.weight",         "attention_k"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",        "blk.{L}.attn_v.weight",         "attention_v"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",        "blk.{L}.attn_output.weight",    "attention_o"),

    # dense MLP (layer 0 only)
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",           "blk.{L}.ffn_gate.weight",       "dense_mlp_gate"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",             "blk.{L}.ffn_up.weight",         "dense_mlp_up"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",           "blk.{L}.ffn_down.weight",       "dense_mlp_down"),

    # router
    (r"^model\.layers\.(\d+)\.mlp\.gate\.weight$",                "blk.{L}.ffn_gate_inp.weight",   "router"),
    (r"^model\.layers\.(\d+)\.mlp\.e_score_correction_bias$",     "blk.{L}.exp_probs_b.bias",      "router"),

    # shared expert
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$", "blk.{L}.ffn_gate_shexp.weight", "shared_expert_gate"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$",   "blk.{L}.ffn_up_shexp.weight",   "shared_expert_up"),
    (r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$", "blk.{L}.ffn_down_shexp.weight", "shared_expert_down"),

    # routed experts -- 128 HF tensors merge into one 3-D GGUF tensor
    (r"^model\.layers\.(\d+)\.mlp\.experts\.\d+\.gate_proj\.weight$", "blk.{L}.ffn_gate_exps.weight", "routed_expert_gate"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.\d+\.up_proj\.weight$",   "blk.{L}.ffn_up_exps.weight",   "routed_expert_up"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.\d+\.down_proj\.weight$", "blk.{L}.ffn_down_exps.weight", "routed_expert_down"),

    # MTP / NextN block -> blk.48.*
    (r"^mtp\.fc\.weight$",                        f"blk.{MTP_BID}.nextn.eh_proj.weight",           "mtp"),
    (r"^mtp\.pre_fc_norm_embedding\.weight$",     f"blk.{MTP_BID}.nextn.enorm.weight",             "mtp"),
    (r"^mtp\.pre_fc_norm_hidden\.weight$",        f"blk.{MTP_BID}.nextn.hnorm.weight",             "mtp"),
    (r"^mtp\.norm\.weight$",                      f"blk.{MTP_BID}.nextn.shared_head_norm.weight",  "mtp"),
    (r"^mtp\.layers\.0\.input_layernorm\.weight$",          f"blk.{MTP_BID}.attn_norm.weight",     "mtp"),
    (r"^mtp\.layers\.0\.post_attention_layernorm\.weight$", f"blk.{MTP_BID}.ffn_norm.weight",      "mtp"),
    (r"^mtp\.layers\.0\.self_attn\.q_norm\.weight$",        f"blk.{MTP_BID}.attn_q_norm.weight",   "mtp"),
    (r"^mtp\.layers\.0\.self_attn\.k_norm\.weight$",        f"blk.{MTP_BID}.attn_k_norm.weight",   "mtp"),
    (r"^mtp\.layers\.0\.self_attn\.q_proj\.weight$",        f"blk.{MTP_BID}.attn_q.weight",        "mtp"),
    (r"^mtp\.layers\.0\.self_attn\.k_proj\.weight$",        f"blk.{MTP_BID}.attn_k.weight",        "mtp"),
    (r"^mtp\.layers\.0\.self_attn\.v_proj\.weight$",        f"blk.{MTP_BID}.attn_v.weight",        "mtp"),
    (r"^mtp\.layers\.0\.self_attn\.o_proj\.weight$",        f"blk.{MTP_BID}.attn_output.weight",   "mtp"),
    (r"^mtp\.layers\.0\.mlp\.gate_proj\.weight$",           f"blk.{MTP_BID}.ffn_gate.weight",      "mtp"),
    (r"^mtp\.layers\.0\.mlp\.up_proj\.weight$",             f"blk.{MTP_BID}.ffn_up.weight",        "mtp"),
    (r"^mtp\.layers\.0\.mlp\.down_proj\.weight$",           f"blk.{MTP_BID}.ffn_down.weight",      "mtp"),
]
MAP = [(re.compile(p), t, g) for p, t, g in MAP]

# groups llama.cpp refuses to quantize (kept at source dtype)
NEVER_QUANT = {"final_norm", "attention_norm", "ffn_norm", "router"}


def classify(hf_name):
    for rx, tmpl, group in MAP:
        m = rx.match(hf_name)
        if m:
            L = int(m.group(1)) if m.groups() else None
            return tmpl.replace("{L}", str(L)) if L is not None else tmpl, group, L
    return None, "other", None


def build(st_headers):
    """Collapse HF tensors into the GGUF tensor table."""
    gguf = {}
    for hf, meta in st_headers.items():
        name, group, L = classify(hf)
        if name is None:
            gguf.setdefault(f"UNMAPPED::{hf}", {"group": "other", "layer": None,
                                                "elements": 0, "src_dtype": meta["dtype"], "n_src": 0})
            continue
        e = gguf.setdefault(name, {"group": group, "layer": L, "elements": 0,
                                   "src_dtype": meta["dtype"], "n_src": 0})
        n = 1
        for d in meta["shape"]:
            n *= d
        e["elements"] += n            # expert tensors accumulate across 128 experts
        e["n_src"] += 1
    return gguf


# ---------------------------------------------------------------- recipe
def resolve_type(name, group, layer, recipe, variant):
    r = recipe["rules"]
    ov = {}
    for o in recipe.get("layer_overrides") or []:
        if layer is not None and layer in o["layers"]:
            ov = o
    vr = (recipe.get("variants") or {}).get(variant, {})

    if group in NEVER_QUANT:
        return "SOURCE"
    if group == "embedding":
        return vr.get("embedding", r["embedding"]).upper()
    if group == "output":
        return vr.get("output", r["output"]).upper()
    if group in ("attention_q", "attention_k", "attention_v", "attention_o"):
        return r["attention"].upper()
    if group in ("dense_mlp_gate", "dense_mlp_up", "dense_mlp_down"):
        return r["dense_layer_0"].upper()
    if group in ("shared_expert_gate", "shared_expert_up", "shared_expert_down"):
        return r["shared_expert"].upper()
    if group in ("routed_expert_gate", "routed_expert_up"):
        return (ov.get("routed_expert_gate_up")
                or vr.get("routed_expert_gate_up")
                or r["routed_expert_gate_up"]).upper()
    if group == "routed_expert_down":
        return (ov.get("routed_expert_down")
                or vr.get("routed_expert_down")
                or r["routed_expert_down"]).upper()
    if group == "mtp":
        # MTP norms are still norms -> never quantized
        if "norm" in name:
            return "SOURCE"
        return r["mtp"].upper()
    return "SOURCE"


def nbytes(elements, qtype, src_dtype="BF16"):
    t = src_dtype.upper().replace("BFLOAT16", "BF16").replace("FLOAT32", "F32").replace("FLOAT16", "F16")
    if qtype == "SOURCE":
        qtype = t
    blck, size = GGML_QUANT_SIZES[getattr(GGMLQuantizationType, qtype)]
    return elements // blck * size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--recipe", default=str(ROOT / "manifests/quant-recipe-v1.yaml"))
    ap.add_argument("--headers", default=str(ROOT / "manifests/st-headers.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import yaml
    recipe = yaml.safe_load(open(a.recipe))
    st = json.load(open(a.headers))
    gguf = build(st)

    rows, by_group = [], collections.defaultdict(lambda: {"n": 0, "elem": 0, "src": 0, "dst": 0, "types": set()})
    for name, e in sorted(gguf.items()):
        qt = resolve_type(name, e["group"], e["layer"], recipe, a.variant)
        src_b = nbytes(e["elements"], "SOURCE", e["src_dtype"])
        dst_b = nbytes(e["elements"], qt, e["src_dtype"])
        rows.append({"name": name, "group": e["group"], "layer": e["layer"],
                     "elements": e["elements"], "src_dtype": e["src_dtype"],
                     "quant": qt, "src_bytes": src_b, "dst_bytes": dst_b,
                     "n_source_tensors": e["n_src"]})
        g = by_group[e["group"]]
        g["n"] += 1; g["elem"] += e["elements"]; g["src"] += src_b; g["dst"] += dst_b
        g["types"].add(qt if qt != "SOURCE" else e["src_dtype"])

    tot_src = sum(r["src_bytes"] for r in rows)
    tot_dst = sum(r["dst_bytes"] for r in rows)
    tot_elem = sum(r["elements"] for r in rows)

    GI = 2 ** 30
    print(f"variant: {a.variant}")
    print(f"{'group':24s} {'tensors':>8s} {'params':>14s} {'src GiB':>9s} {'dst GiB':>9s}  types")
    print("-" * 96)
    for gname, g in sorted(by_group.items(), key=lambda kv: -kv[1]["dst"]):
        print(f"{gname:24s} {g['n']:8d} {g['elem']:14,d} {g['src']/GI:9.2f} {g['dst']/GI:9.2f}  "
              f"{','.join(sorted(g['types']))}")
    print("-" * 96)
    print(f"{'TOTAL':24s} {len(rows):8d} {tot_elem:14,d} {tot_src/GI:9.2f} {tot_dst/GI:9.2f}")
    lo, hi = recipe["recipe"]["target_size_gib"]
    hard = recipe["recipe"]["hard_limit_gib"]
    got = tot_dst / GI
    verdict = "IN TARGET" if lo <= got <= hi else ("UNDER TARGET" if got < lo else
              ("OVER TARGET, under hard limit" if got <= hard else "OVER HARD LIMIT"))
    print(f"\ntarget {lo}-{hi} GiB, hard limit {hard} GiB  ->  {got:.2f} GiB  [{verdict}]")

    if a.out:
        json.dump({"variant": a.variant, "total_params": tot_elem,
                   "total_src_bytes": tot_src, "total_dst_bytes": tot_dst,
                   "by_group": {k: {kk: (sorted(vv) if isinstance(vv, set) else vv)
                                    for kk, vv in v.items()} for k, v in by_group.items()},
                   "tensors": rows}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
