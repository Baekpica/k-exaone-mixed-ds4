#!/usr/bin/env python3
"""Turn llama-eval-callback's tensor dump into a comparable router fixture.

The dump is human-readable text with a header line per tensor followed by
bracketed values. We keep only what a port has to reproduce exactly: which
experts each layer selected, and with what weights.
"""
import argparse, json, re, sys
from pathlib import Path

HDR = re.compile(
    r"common_debug_cb_eval:\s+(?P<name>\S+)\s+=\s+\((?P<type>\w+)\).*?=\s+\{(?P<ne>[^}]*)\}")
NUM = re.compile(r"-?\d+\.?\d*(?:e[+-]?\d+)?")


def parse(path):
    """-> {tensor_name: [values...]} in dump order."""
    out, cur, vals = {}, None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = HDR.search(line)
        if m:
            if cur:
                out[cur] = vals
            cur, vals = m.group("name"), []
            continue
        if cur and ("[" in line or "]" in line):
            # value rows look like:  [ 12.0000, 3.0000, ... ]
            body = line.strip().strip("[],")
            if body and not body.startswith("..."):
                vals += [float(x) for x in NUM.findall(body)]
    if cur:
        out[cur] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--experts-used", type=int, default=8)
    a = ap.parse_args()

    raw = parse(a.raw)
    layers = {}
    for name, vals in raw.items():
        m = re.match(r"(ffn_moe_\w+)-(\d+)$", name)
        if not m:
            continue
        kind, il = m.group(1), int(m.group(2))
        L = layers.setdefault(il, {})
        if kind == "ffn_moe_topk":
            L["selected_experts"] = [int(v) for v in vals[:a.experts_used]]
        elif kind == "ffn_moe_weights_norm":
            L["weights"] = [round(v, 6) for v in vals[:a.experts_used]]
        elif kind == "ffn_moe_probs":
            L["probs_head"] = [round(v, 6) for v in vals[:16]]

    if not layers:
        print("no router tensors parsed -- check the dump and the --tensor-filter",
              file=sys.stderr)
        print(f"tensors seen: {list(raw)[:10]}", file=sys.stderr)
        sys.exit(1)

    # top-k selection is without replacement, so a repeated expert index means
    # the dump was elided (LLAMA_DEBUG_PRINT_N too small) and rows got stitched
    # together -- a silently wrong fixture is worse than none.
    bad = [il for il, L in layers.items()
           if L.get("selected_experts")
           and len(set(L["selected_experts"])) != len(L["selected_experts"])]
    if bad:
        print(f"ERROR: duplicate expert indices in layers {sorted(bad)[:8]}"
              f"{'...' if len(bad) > 8 else ''} -- rerun with LLAMA_DEBUG_PRINT_N "
              f">= {a.experts_used}", file=sys.stderr)
        sys.exit(2)

    doc = {"prompt": a.prompt, "n_expert_used": a.experts_used,
           "note": "first token position of the last prompt-eval batch",
           "validated": "expert indices distinct within each layer",
           "layers": {str(k): v for k, v in sorted(layers.items())}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"parsed {len(layers)} MoE layers -> {a.out}")
    for il in sorted(layers)[:3]:
        print(f"  layer {il}: experts {layers[il].get('selected_experts')}")


if __name__ == "__main__":
    main()
