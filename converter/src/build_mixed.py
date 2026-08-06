#!/usr/bin/env python3
"""Recipe-driven mixed-quant GGUF builder for K-EXAONE.

Wraps llama.cpp's quantizer (work-order B3 priority 2). The recipe YAML is the
single source of truth; this emits an ordered --tensor-type-file and drives
llama-quantize.

Ordering matters: llama-quantize matches with std::regex_search and takes the
FIRST match, so patterns are emitted most-specific-first. A terminal catch-all
makes the recipe total -- every quantizable tensor gets an explicit type from
the recipe rather than falling through to llama.cpp's built-in mixture logic.
"""
import argparse, json, os, re, subprocess, sys, hashlib, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANTIZE = ROOT / "llama.cpp/build-sm120/bin/llama-quantize"

# Tensors llama.cpp never quantizes -- kept at source dtype.
# (tensor_allows_quantization: <2D, not ending in "weight", "_norm.weight",
#  "ffn_gate_inp.weight")
NEVER_QUANT_DOC = ["*_norm.weight", "blk.*.ffn_gate_inp.weight", "blk.*.exp_probs_b.bias"]


def emit_patterns(recipe, variant, n_layers=48):
    """Recipe -> ordered [(regex, ggml_type, comment)]."""
    r = dict(recipe["rules"])
    vr = (recipe.get("variants") or {}).get(variant, {}) or {}
    for k in ("routed_expert_gate_up", "routed_expert_down", "embedding", "output"):
        if k in vr:
            r[k] = vr[k]

    edge = []
    for o in recipe.get("layer_overrides") or []:
        edge.append(o)

    pats = []
    mtp_bid = n_layers  # MTP occupies blk.48

    # 1. MTP block first: its attn/ffn tensors would otherwise be caught by the
    #    generic attention/dense patterns below.
    pats.append((rf"^blk\.{mtp_bid}\.", r["mtp"], "MTP / NextN block"))

    # 2. edge-layer routed experts (most specific expert rule)
    for o in edge:
        # longest alternatives first so blk.44 is not shadowed by blk.4
        alts = "|".join(str(x) for x in sorted(o["layers"], key=lambda v: (-len(str(v)), v)))
        if "routed_expert_gate_up" in o:
            pats.append((rf"^blk\.({alts})\.ffn_(gate|up)_exps\.weight$",
                         o["routed_expert_gate_up"], f"edge layers {o['layers']} gate/up"))
        if "routed_expert_down" in o:
            pats.append((rf"^blk\.({alts})\.ffn_down_exps\.weight$",
                         o["routed_expert_down"], f"edge layers {o['layers']} down"))

    # 3. routed experts, all remaining layers
    pats.append((r"^blk\.\d+\.ffn_(gate|up)_exps\.weight$", r["routed_expert_gate_up"], "routed expert gate/up"))
    pats.append((r"^blk\.\d+\.ffn_down_exps\.weight$",      r["routed_expert_down"],    "routed expert down"))

    # 4. shared expert
    pats.append((r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$", r["shared_expert"], "shared expert"))

    # 5. dense layer 0 MLP
    pats.append((r"^blk\.0\.ffn_(gate|up|down)\.weight$", r["dense_layer_0"], "dense layer 0 MLP"))

    # 6. attention
    pats.append((r"^blk\.\d+\.attn_(q|k|v|output)\.weight$", r["attention"], "attention q/k/v/o"))

    # 7. terminal catch-all -- makes the recipe total
    pats.append((r".", r.get("other", "q8_0"), "catch-all (small tensors)"))
    return pats


def sha256(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bs):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-gguf")
    src.add_argument("--source-hf")
    ap.add_argument("--recipe", default=str(ROOT / "manifests/quant-recipe-v1.yaml"))
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--out")
    ap.add_argument("--imatrix")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--base-ftype", default=None,
                    help="llama-quantize positional ftype; defaults to the routed gate/up type")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-tensors", action="store_true")
    ap.add_argument("--compare-tensor")
    ap.add_argument("--resume", action="store_true",
                    help="skip if --out already exists and is non-empty")
    ap.add_argument("--print-patterns", action="store_true")
    a = ap.parse_args()

    import yaml
    recipe = yaml.safe_load(open(a.recipe))
    pats = emit_patterns(recipe, a.variant)

    if a.print_patterns or a.list_tensors:
        print(f"# recipe: {recipe['recipe']['name']}  variant: {a.variant}")
        print(f"# match order matters (regex_search, first match wins)")
        w = max(len(p) for p, _, _ in pats)
        for p, t, c in pats:
            print(f"{p:<{w}} = {t:<8s}  # {c}")
        print("\n# never quantized by llama.cpp (kept at source dtype):")
        for n in NEVER_QUANT_DOC:
            print(f"#   {n}")
        if a.print_patterns:
            return

    if a.source_hf:
        sys.exit("--source-hf not implemented: the official BF16 GGUF is the pinned "
                 "conversion reference (work order A2). Use --source-gguf.")

    if not a.out:
        sys.exit("--out required")
    out = Path(a.out)
    if a.resume and out.exists() and out.stat().st_size > 0:
        print(f"[resume] {out} exists ({out.stat().st_size/2**30:.2f} GiB) -- skipping")
        return

    ttf = out.with_suffix(".tensor-types.txt")
    ttf.parent.mkdir(parents=True, exist_ok=True)
    with open(ttf, "w") as f:
        for p, t, _ in pats:
            f.write(f"{p}={t}\n")

    r = dict(recipe["rules"])
    vr = (recipe.get("variants") or {}).get(a.variant, {}) or {}
    base_ftype = a.base_ftype or vr.get("routed_expert_gate_up") or r["routed_expert_gate_up"]

    cmd = [str(QUANTIZE)]
    if a.imatrix:
        cmd += ["--imatrix", a.imatrix]
    cmd += ["--tensor-type-file", str(ttf),
            "--token-embedding-type", vr.get("embedding", r["embedding"]),
            "--output-tensor-type", vr.get("output", r["output"])]
    if a.dry_run:
        cmd += ["--dry-run"]
    cmd += [a.source_gguf, str(out), base_ftype, str(a.threads)]

    print("[cmd]", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc != 0:
        sys.exit(f"llama-quantize failed rc={rc}")
    if a.dry_run:
        print(f"[dry-run] completed in {dt:.1f}s")
        return

    size = out.stat().st_size
    print(f"[done] {out}  {size/2**30:.2f} GiB in {dt/60:.1f} min")
    man = out.with_suffix(".manifest.json")
    json.dump({
        "artifact": str(out), "bytes": size, "gib": size / 2**30,
        "sha256": sha256(out),
        "recipe": recipe["recipe"], "variant": a.variant,
        "patterns": [{"regex": p, "type": t, "note": c} for p, t, c in pats],
        "source_gguf": a.source_gguf, "imatrix": a.imatrix,
        "base_ftype": base_ftype, "build_seconds": dt,
    }, open(man, "w"), indent=1)
    print(f"[manifest] {man}")


if __name__ == "__main__":
    main()
