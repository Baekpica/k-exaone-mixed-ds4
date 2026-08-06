#!/usr/bin/env python3
"""Build the imatrix calibration corpus for K-EXAONE.

Composition is our own, but the bucket structure follows the reference mix
Baekpica/Solar-Open2-120B-A15B-REAM-148E-Healing-Mix (instruction / reasoning /
Korean / multilingual / code). Two things had to change for K-EXAONE:

  * Language set. K-EXAONE serves en, ko, es, de, ja, vi. Nemotron-Multilingual
    -v2 covers ko/ja (plus hi/pt we skip); -v1 covers es/de (plus it/fr/zh we
    skip). Neither has Vietnamese, so vi comes from Wikipedia -- the only source
    available. Five of six languages therefore come from instruction-formatted
    data and one from encyclopedic prose; that asymmetry is recorded, not hidden.
  * Korean is weighted heaviest. The routed expert gate/up tensors go to
    IQ2_XXS, the most aggressive quant in the recipe, and Korean capacity is
    what this artifact exists to protect.

Sampling reads windows at spread-out byte offsets via HTTP range requests --
these JSONL shards are 0.07-7.7 GB each and we need ~16 MB total, so
downloading them is not worth it. Records are rendered with K-EXAONE's own chat
template markers so the activations the imatrix sees match serving time.
"""
import argparse, json, random, sys
from pathlib import Path
import requests
from huggingface_hub import HfApi, hf_hub_url, get_token

TOK = get_token()
HDRS = {"Authorization": f"Bearer {TOK}"} if TOK else {}
SESS = requests.Session()

# K-EXAONE chat markers (from chat_template.jinja)
ROLE = {"user": "<|user|>\n", "assistant": "<|assistant|>\n",
        "system": "<|system|>\n", "tool": "<|tool|>\n"}
EOT = "<|endofturn|>\n"

V2 = "nvidia/Nemotron-SFT-Multilingual-v2"
V1 = "nvidia/Nemotron-SFT-Multilingual-v1"

# bucket -> (repo, [files], share)
BUCKETS = [
    ("ko",            V2, ["ultra-v3_stem_ko_translated_postedit_final.jsonl",
                           "ultra-v3_math_ko_translated_final.jsonl",
                           "ultra-v3_code_ko_translated_final.jsonl"], 0.26),
    ("en_reasoning",  "nvidia/Nemotron-Cascade-SFT-Stage-1", None, 0.16),
    ("en_instruct",   "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", None, 0.12),
    ("ja",            V2, ["ultra-v3_stem_ja_translated_postedit_final.jsonl",
                           "ultra-v3_math_ja_translated_final.jsonl"], 0.09),
    ("es",            V1, ["data/super-v3_stem_es_translated_postedit_final.jsonl",
                           "data/super-v3_math_es_translated_final.jsonl"], 0.09),
    ("de",            V1, ["data/super-v3_stem_de_translated_postedit_final.jsonl",
                           "data/super-v3_math_de_translated_final.jsonl"], 0.09),
    ("code_algo",     "nvidia/Nemotron-SFT-Competitive-Programming-v2", None, 0.07),
]
WIKI_BUCKETS = [("vi", "vi", 0.07), ("en_prose", "en", 0.05)]

api = HfApi()
_sizes = {}


def file_size(repo, fn):
    key = (repo, fn)
    if key not in _sizes:
        info = api.dataset_info(repo, files_metadata=True)
        for s in info.siblings:
            _sizes[(repo, s.rfilename)] = s.size or 0
    return _sizes.get(key, 0)


def list_jsonl(repo):
    info = api.dataset_info(repo, files_metadata=True)
    out = []
    for s in info.siblings:
        if s.rfilename.endswith((".jsonl", ".json")) and (s.size or 0) > 1e6:
            _sizes[(repo, s.rfilename)] = s.size
            out.append(s.rfilename)
    return sorted(out)


def render(rec):
    """messages -> chat-template-rendered text."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        for k in ("text", "content", "problem", "question"):
            if isinstance(rec.get(k), str) and len(rec[k]) > 100:
                return rec[k]
        return None
    parts = []
    for m in msgs:
        r, c = m.get("role"), m.get("content")
        if not isinstance(c, str) or not c.strip() or r not in ROLE:
            continue
        parts.append(ROLE[r] + c.strip() + EOT)
    t = "".join(parts)
    return t if len(t) > 150 else None


def sample_jsonl(repo, fn, want_chars, windows=6, seed=0):
    """Fetch `windows` byte ranges spread across the shard, keep whole lines."""
    sz = file_size(repo, fn)
    if not sz:
        return []
    url = hf_hub_url(repo, fn, repo_type="dataset")
    win = max(1 << 20, min(6 << 20, want_chars * 3 // windows))
    rng = random.Random(seed)
    out, got = [], 0
    starts = [int(sz * (i + rng.random() * 0.6) / windows) for i in range(windows)]
    for st in starts:
        if got >= want_chars:
            break
        end = min(st + win, sz - 1)
        if end <= st:
            continue
        try:
            r = SESS.get(url, headers={**HDRS, "Range": f"bytes={st}-{end}"}, timeout=300)
            r.raise_for_status()
        except Exception as e:
            print(f"    range fail {fn}: {type(e).__name__}", file=sys.stderr)
            continue
        lines = r.content.decode("utf-8", "ignore").split("\n")[1:-1]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            t = render(rec)
            if t:
                t = t[:12000]
                out.append(t)
                got += len(t)
                if got >= want_chars:
                    break
    return out


def sample_wiki(lang, want_chars, seed=0):
    from datasets import load_dataset
    try:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                          split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
    except Exception as e:
        print(f"    wiki {lang} unavailable: {type(e).__name__}", file=sys.stderr)
        return []
    out, got = [], 0
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) < 400:
            continue
        t = t[:6000]
        out.append(t)
        got += len(t)
        if got >= want_chars:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/k-exaone-mixed-ds4/fixtures/calibration.txt")
    ap.add_argument("--total-mb", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    total = int(a.total_mb * 1024 * 1024)
    chunks, report = [], []

    for name, repo, files, share in BUCKETS:
        want = int(total * share)
        print(f"[{name}] target {want/1e6:.1f} MB from {repo}", flush=True)
        try:
            fl = files if files else list_jsonl(repo)[:4]
        except Exception as e:
            print(f"    repo unavailable: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            report.append({"bucket": name, "repo": repo, "chars": 0, "docs": 0, "status": "unavailable"})
            continue
        got = []
        per = want // max(1, len(fl))
        for i, fn in enumerate(fl):
            got += sample_jsonl(repo, fn, per, seed=a.seed + i)
        n = sum(map(len, got))
        print(f"    {len(got)} docs, {n/1e6:.2f} MB", flush=True)
        chunks += got
        report.append({"bucket": name, "repo": repo, "files": fl, "chars": n,
                       "docs": len(got), "status": "ok" if got else "empty"})

    for name, lang, share in WIKI_BUCKETS:
        want = int(total * share)
        print(f"[{name}] target {want/1e6:.1f} MB from wikipedia/{lang}", flush=True)
        got = sample_wiki(lang, want, a.seed)
        n = sum(map(len, got))
        print(f"    {len(got)} docs, {n/1e6:.2f} MB", flush=True)
        chunks += got
        report.append({"bucket": name, "repo": f"wikimedia/wikipedia:{lang}",
                       "chars": n, "docs": len(got), "status": "ok" if got else "empty"})

    random.Random(a.seed).shuffle(chunks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.replace("\r", "") + "\n\n")
    sz = Path(a.out).stat().st_size
    rp = Path(a.out).with_suffix(".composition.json")
    json.dump({"output": a.out, "bytes": sz, "documents": len(chunks),
               "seed": a.seed, "buckets": report}, open(rp, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {a.out}: {len(chunks)} docs, {sz/1e6:.1f} MB")
    print(f"composition -> {rp}")
    for r in report:
        print(f"  {r['bucket']:14s} {r['chars']/1e6:6.2f} MB  {r['status']}")


if __name__ == "__main__":
    main()
