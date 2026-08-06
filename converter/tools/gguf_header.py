#!/usr/bin/env python3
"""Minimal GGUF header reader: KV metadata + tensor table only.

gguf-py's GGUFReader mmaps every tensor's data, which fails on a partially
downloaded file. The header sits at the front, so parsing it by hand lets us
read metadata and the full tensor table while the download is still running.
"""
import struct, sys, json

# GGUF value type ids
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
       F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}
SZ = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1,
      U64: 8, I64: 8, F64: 8}

GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4",
}


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated header")
        return b

    def prim(self, t):
        return struct.unpack(FMT[t], self.raw(SZ[t]))[0]

    def string(self):
        n = self.prim(U64)
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.prim(U32)
            n = self.prim(U64)
            if et == STRING:
                return [self.string() for _ in range(n)]
            if et == ARRAY:
                return [self.value(ARRAY) for _ in range(n)]
            sz, fmt = SZ[et], FMT[et][1]
            buf = self.raw(sz * n)
            return list(struct.unpack(f"<{n}{fmt}", buf))
        return self.prim(t)


def read(path):
    f = open(path, "rb")
    r = R(f)
    magic = r.raw(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file: {magic!r}")
    version = r.prim(U32)
    n_tensors = r.prim(U64)
    n_kv = r.prim(U64)
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.prim(U32)
        kv[k] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.prim(U32)
        ne = [r.prim(U64) for _ in range(nd)]
        tt = r.prim(U32)
        off = r.prim(U64)
        tensors.append({"name": name, "ne": ne,
                        "type": GGML_TYPE.get(tt, f"?{tt}"), "offset": off})
    f.close()
    return {"version": version, "n_tensors": n_tensors, "kv": kv, "tensors": tensors}


if __name__ == "__main__":
    d = read(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"GGUF v{d['version']}  tensors={d['n_tensors']}  kv={len(d['kv'])}")
    print("\n=== metadata ===")
    for k, v in d["kv"].items():
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        if isinstance(v, list) and len(v) > 12:
            s = f"[{len(v)} items] {json.dumps(v[:8], ensure_ascii=False)}..."
        print(f"  {k:56s} = {s[:110]}")
    if out:
        json.dump(d, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"\nwrote {out}")
