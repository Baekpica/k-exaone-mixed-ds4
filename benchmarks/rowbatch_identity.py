#!/usr/bin/env python3
"""Does row-batched decode change the answer?

Runs one prompt solo -- every decode step takes the count==1 sequential path --
then again while 7 filler streams are mid-generation, so its steps ride the
row-batched pass.  Greedy, so the two texts must match token for token.

The fillers are started first and given a long budget; the probe is sent only
after all of them have produced their first content chunk, which is the moment
the batch counter is at its widest.  The server is expected to be started with
DS4_SERVER_BATCH_LOG=1 so the decode batch widths are on record.

  python3 rowbatch_identity.py --base http://127.0.0.1:8001 \
      --corpus .../promessi_sposi.txt --out-dir OUT
"""
import argparse, json, os, threading, time, urllib.request

PROBE = ("다음 문장을 이어서 자연스럽게 세 문장으로 완성하세요: "
         "바람이 부는 언덕 위에서 그는")


def chat_stream(base, content, max_tokens, timeout, first_chunk_evt=None):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "K-EXAONE-236B-A23B",
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "disabled"},
            "temperature": 0, "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"})
    text, usage = [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            raw = raw.strip()
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    text.append(delta)
                    if first_chunk_evt and not first_chunk_evt.is_set():
                        first_chunk_evt.set()
    return "".join(text), usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fillers", type=int, default=7)
    ap.add_argument("--probe-tokens", type=int, default=96)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    corpus = open(args.corpus, encoding="utf-8", errors="replace").read()

    print("identity: solo run (sequential decode path)")
    t0 = time.perf_counter()
    solo, u_solo = chat_stream(args.base, PROBE, args.probe_tokens, 600)
    t_solo = time.perf_counter() - t0
    print(f"identity: solo {t_solo:.1f}s tokens={u_solo and u_solo.get('completion_tokens')}")

    events = [threading.Event() for _ in range(args.fillers)]
    results = [None] * args.fillers

    def filler(i):
        text = corpus[400000 + i * 8000:400000 + i * 8000 + 700]
        try:
            results[i] = chat_stream(
                args.base, text + "\n\n---\n\n이 대목을 아주 상세히 분석해 주세요.",
                640, 1800, events[i])
        except Exception as exc:            # noqa: BLE001 - recorded, not fatal
            results[i] = ("", {"error": str(exc)})
            events[i].set()

    threads = [threading.Thread(target=filler, args=(i,)) for i in range(args.fillers)]
    for th in threads:
        th.start()
    for i, evt in enumerate(events):
        if not evt.wait(600):
            print(f"identity: filler {i} produced nothing in 600s"); return 1
    print(f"identity: all {args.fillers} fillers decoding; sending probe")

    t0 = time.perf_counter()
    batched, u_b = chat_stream(args.base, PROBE, args.probe_tokens, 600)
    t_b = time.perf_counter() - t0
    print(f"identity: batched {t_b:.1f}s tokens={u_b and u_b.get('completion_tokens')}")
    for th in threads:
        th.join()

    same = solo == batched
    print(f"\nidentity: solo == batched -> {same}")
    if not same:
        print(f"solo:    {solo!r}\nbatched: {batched!r}")
    with open(os.path.join(args.out_dir, "rowbatch-identity.json"), "w") as fh:
        json.dump({"identical": same,
                   "solo": {"text": solo, "usage": u_solo, "elapsed_s": t_solo},
                   "batched": {"text": batched, "usage": u_b, "elapsed_s": t_b},
                   "fillers": [
                       {"chars": len(r[0]) if r else 0,
                        "usage": (r[1] if r else None)}
                       for r in results]},
                  fh, ensure_ascii=False, indent=2)
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
