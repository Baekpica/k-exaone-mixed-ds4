#!/usr/bin/env python3
"""Check importance-matrix coverage before it is used to build IQ2_XXS.

The risk this catches: with top-8 routing over 128 experts, a calibration run
that is too short or too narrow leaves some experts never activated. Their
imatrix rows stay zero, and IQ2_XXS quantization of those experts falls back to
something close to unweighted -- exactly the experts most likely to hold rare
capability. A silent zero here becomes a silently bad artifact.
"""
import argparse, sys, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imatrix")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--report")
    a = ap.parse_args()

    rd = GGUFReader(a.imatrix)
    kv = {k: f for k, f in rd.fields.items()}

    def get(k, default=None):
        f = kv.get(k)
        if f is None:
            return default
        try:
            v = f.contents()
            return v
        except Exception:
            return default

    chunks = get("imatrix.chunk_count")
    csize = get("imatrix.chunk_size")
    print(f"chunks: {chunks}  chunk_size: {csize}  "
          f"tokens: {(chunks or 0) * (csize or 0):,}")
    print(f"entries: {len(rd.tensors)}")

    import numpy as np
    zero_rows, expert_stats, problems = [], {}, []
    for t in rd.tensors:
        name = t.name
        arr = np.asarray(t.data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # per-row (per-expert for 3-D expert tensors) activation mass
        sums = arr.sum(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else arr
        n_zero = int((sums == 0).sum())
        if n_zero:
            zero_rows.append((name, n_zero, int(sums.size)))
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps", name)
        if m:
            expert_stats[name] = {"rows": int(sums.size),
                                  "zero_rows": n_zero,
                                  "min": float(sums.min()),
                                  "max": float(sums.max()),
                                  "mean": float(sums.mean())}
            if sums.size != a.experts:
                problems.append(f"{name}: {sums.size} rows, expected {a.experts} experts")

    print(f"\nrouted-expert tensors with imatrix data: {len(expert_stats)}")
    if expert_stats:
        worst = sorted(expert_stats.items(), key=lambda kv: kv[1]["min"])[:5]
        print("lowest per-expert activation mass:")
        for n, s in worst:
            print(f"  {n:34s} min={s['min']:.4g}  mean={s['mean']:.4g}  "
                  f"zero_rows={s['zero_rows']}/{s['rows']}")
        tot_zero = sum(s["zero_rows"] for s in expert_stats.values())
        tot_rows = sum(s["rows"] for s in expert_stats.values())
        print(f"\nexperts never activated: {tot_zero} / {tot_rows} expert-slots")
        if tot_zero:
            problems.append(f"{tot_zero} expert slots have zero activation mass")

    if zero_rows:
        print(f"\ntensors with any zero rows: {len(zero_rows)}")
        for n, z, tot in zero_rows[:10]:
            print(f"  {n:40s} {z}/{tot}")

    print("\n" + ("PROBLEMS:" if problems else "OK: every expert was activated"))
    for p in problems[:20]:
        print(f"  {p}")

    if a.report:
        json.dump({"imatrix": a.imatrix, "chunks": chunks, "chunk_size": csize,
                   "tokens": (chunks or 0) * (csize or 0),
                   "entries": len(rd.tensors),
                   "expert_tensors": len(expert_stats),
                   "expert_stats": expert_stats,
                   "problems": problems}, open(a.report, "w"), indent=1)
        print(f"report -> {a.report}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
