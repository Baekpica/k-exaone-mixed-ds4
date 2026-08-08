#!/usr/bin/env python3
"""Does the model actually read a long prompt?

A summary can look plausible while the model saw nothing, so this plants an
exact string at a known depth and asks for it back.  Pass/fail is a substring
test, not a judgement.

The needle is placed at several depths in separate requests.  Depth matters
here specifically: the exaone graph's 36 sliding layers hold a 128-position
window, so a prefill defect shows up as "recalls the needle only when it lands
in the final chunk" rather than as a flat failure.

  python3 recall_test.py --base http://127.0.0.1:8001 \
      --corpus .../promessi_sposi.txt --out-dir OUT
"""
import argparse, json, os, time, urllib.request

NEEDLE = "이 문서의 비밀 코드는 SPARK-7731입니다."
ANSWER = "SPARK-7731"


def post(base, payload, timeout=1800):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--doc-chars", type=int, default=24000)
    ap.add_argument("--depths", default="0.10,0.50,0.90")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    text = open(args.corpus, encoding="utf-8", errors="replace").read()
    doc = text[600000:600000 + args.doc_chars]

    results = []
    for d in [float(x) for x in args.depths.split(",")]:
        cut = int(len(doc) * d)
        # Land on a line boundary so the needle reads as its own sentence.
        nl = doc.rfind("\n", 0, cut)
        cut = nl + 1 if nl > 0 else cut
        planted = doc[:cut] + "\n" + NEEDLE + "\n" + doc[cut:]
        msg = [{"role": "user",
                "content": planted +
                "\n\n---\n\n위 문서에 적힌 비밀 코드를 그대로 답하세요."}]
        body, elapsed = post(args.base, {
            "model": "K-EXAONE-236B-A23B", "messages": msg,
            "thinking": {"type": "disabled"},
            "temperature": 0, "max_tokens": 48})
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        ok = ANSWER in content
        results.append({"depth": d, "found": ok, "content": content,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "elapsed_s": elapsed})
        print(f"depth={d:.2f} prompt_tokens={usage.get('prompt_tokens')} "
              f"found={'YES' if ok else 'NO '} {elapsed:7.1f}s  {content[:70]!r}")

    found = sum(1 for r in results if r["found"])
    print(f"\nneedle recalled at {found}/{len(results)} depths")
    with open(os.path.join(args.out_dir, "recall.json"), "w") as fh:
        json.dump({"needle": NEEDLE, "results": results,
                   "found": found, "total": len(results)}, fh,
                  ensure_ascii=False, indent=2)
    return 0 if found == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
