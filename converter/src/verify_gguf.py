#!/usr/bin/env python3
"""Verify a GGUF against the recipe (work order B4).

Checks, for every tensor in the file:
  - the quant type matches what the recipe says it should be
  - shapes match the BF16 reference (when --reference is given)
  - the tensor set is complete: 48 layers, 128 experts, MTP block present
  - norms and router tensors were left unquantized

Exits non-zero on any mismatch. This is the gate between "the quantizer ran"
and "the artifact is what we asked for".
"""
import argparse, json, sys, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from build_mixed import emit_patterns  # noqa: E402


def expected_type(name, pats, embd_t, out_t):
    """Mirror llama.cpp resolution order: never-quantize, then the dedicated
    embedding/output params, then first-matching regex."""
    if name.endswith("_norm.weight") or name.endswith("ffn_gate_inp.weight") \
            or not name.endswith("weight"):
        return None                      # left at source dtype
    if name == "token_embd.weight":
        return embd_t.upper()
    if name == "output.weight":
        return out_t.upper()
    for rx, t, _ in pats:
        if re.search(rx, name):
            return t.upper()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--recipe", default=str(ROOT / "manifests/quant-recipe-v1.yaml"))
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--reference", help="BF16 GGUF to cross-check shapes against")
    ap.add_argument("--report", help="write JSON report here")
    ap.add_argument("--expect-layers", type=int, default=48)
    ap.add_argument("--expect-experts", type=int, default=128)
    a = ap.parse_args()

    import yaml
    recipe = yaml.safe_load(open(a.recipe))
    pats = emit_patterns(recipe, a.variant)
    r = dict(recipe["rules"])
    vr = (recipe.get("variants") or {}).get(a.variant, {}) or {}
    embd_t = vr.get("embedding", r["embedding"])
    out_t = vr.get("output", r["output"])

    rd = GGUFReader(a.gguf)
    got = {t.name: t for t in rd.tensors}
    print(f"tensors: {len(got)}")

    errors, warns = [], []
    by_type = collections.Counter()
    unquantized = []

    for name, t in sorted(got.items()):
        tt = t.tensor_type.name.upper()
        by_type[tt] += 1
        want = expected_type(name, pats, embd_t, out_t)
        if want is None:
            unquantized.append((name, tt))
            if tt not in ("F32", "F16", "BF16"):
                errors.append(f"{name}: expected source dtype (unquantized), got {tt}")
        elif tt != want:
            errors.append(f"{name}: recipe says {want}, file has {tt}")

    # --- completeness ---
    layers = sorted({int(m.group(1)) for n in got if (m := re.match(r"blk\.(\d+)\.", n))})
    n_blk = len(layers)
    mtp_bid = a.expect_layers
    if layers != list(range(a.expect_layers + 1)):
        errors.append(f"block indices {layers[:3]}..{layers[-3:]} (n={n_blk}); "
                      f"expected 0..{mtp_bid} (48 layers + MTP block)")
    for L in range(1, a.expect_layers):
        for suf in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            nm = f"blk.{L}.{suf}"
            if nm not in got:
                errors.append(f"missing {nm}")
            else:
                sh = list(got[nm].shape)
                if a.expect_experts not in sh:
                    errors.append(f"{nm}: shape {sh} has no dim == {a.expect_experts} experts")
    mtp_any = [n for n in got if n.startswith(f"blk.{mtp_bid}.")]
    if not mtp_any:
        errors.append(f"MTP block blk.{mtp_bid}.* absent -- artifact does not preserve MTP")
    else:
        print(f"MTP block blk.{mtp_bid}: {len(mtp_any)} tensors")

    # --- shapes vs reference ---
    if a.reference:
        ref = {t.name: list(t.shape) for t in GGUFReader(a.reference).tensors}
        if len(ref) != len(got):
            errors.append(f"tensor count {len(got)} != reference {len(ref)}")
        for n, t in got.items():
            if n not in ref:
                errors.append(f"{n}: not in reference")
            elif list(t.shape) != ref[n]:
                errors.append(f"{n}: shape {list(t.shape)} != reference {ref[n]}")

    print("\nquant type histogram:")
    for t, c in by_type.most_common():
        print(f"  {t:10s} {c:5d}")
    print(f"\nleft at source dtype: {len(unquantized)} tensors "
          f"({', '.join(sorted({t for _, t in unquantized}))})")

    size = Path(a.gguf).stat().st_size
    print(f"\nfile size: {size/2**30:.2f} GiB")
    lo, hi = recipe["recipe"]["target_size_gib"]
    hard = recipe["recipe"]["hard_limit_gib"]
    if size / 2**30 > hard:
        errors.append(f"size {size/2**30:.2f} GiB exceeds hard limit {hard} GiB")
    elif not (lo <= size / 2**30 <= hi):
        warns.append(f"size {size/2**30:.2f} GiB outside target {lo}-{hi} GiB (under hard limit)")

    for w in warns:
        print(f"WARN  {w}")
    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors[:40]:
            print(f"  ERR  {e}")
        if len(errors) > 40:
            print(f"  ... and {len(errors)-40} more")
    else:
        print("\nOK: file matches recipe")

    if a.report:
        json.dump({"gguf": a.gguf, "variant": a.variant, "bytes": size,
                   "gib": size / 2**30, "n_tensors": len(got),
                   "type_histogram": dict(by_type),
                   "unquantized": [{"name": n, "type": t} for n, t in unquantized],
                   "errors": errors, "warnings": warns},
                  open(a.report, "w"), indent=1)
        print(f"report -> {a.report}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
