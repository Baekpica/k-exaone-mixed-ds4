#!/usr/bin/env python3
"""Does reusing a prefix change the answer?

Partial-prefix reuse is only worth having if the reused path answers what a
cold prefill answers.  This sends one request warm, evicts the session with an
unrelated long prompt, and sends the byte-identical request again cold.

The two paths are not expected to be bit-identical arithmetic -- a resumed
prefill evaluates its tail as a short chunk where a cold prefill evaluates it
inside a 2048-row one, and the GEMM tiling differs -- so this reports the
token-level agreement rather than asserting equality, and prints both answers
when they part.

  python3 warm_cold_equiv.py --base http://127.0.0.1:8001 \
      --corpus .../promessi_sposi.txt --out-dir OUT
"""
import argparse, difflib, json, os, time, urllib.request


def chat(base, messages, max_tokens=96, timeout=1800):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "K-EXAONE-236B-A23B", "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": 0, "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return (body["choices"][0]["message"]["content"],
            body.get("usage", {}),
            time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--doc-chars", type=int, default=12000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    text = open(args.corpus, encoding="utf-8", errors="replace").read()
    doc = text[600000:600000 + args.doc_chars]
    other = text[300000:300000 + args.doc_chars]

    q1 = [{"role": "user", "content": doc + "\n\n---\n\n이 대목의 인물 관계를 설명해 주세요."}]
    a1, u1, t1 = chat(args.base, q1)
    print(f"seed      prompt={u1.get('prompt_tokens')} cached={u1.get('prompt_tokens_details',{}).get('cached_tokens')} {t1:7.1f}s")

    # The follow-up shares everything up to the assistant turn, which is
    # exactly the shape a chat client produces and the shape that used to miss.
    q2 = q1 + [{"role": "assistant", "content": a1},
               {"role": "user", "content": "방금 답변을 두 문장으로 줄여 주세요."}]

    warm, uw, tw = chat(args.base, q2)
    print(f"warm      prompt={uw.get('prompt_tokens')} cached={uw.get('prompt_tokens_details',{}).get('cached_tokens')} {tw:7.1f}s")

    evict, ue, te = chat(args.base,
                         [{"role": "user", "content": other + "\n\n---\n\n요약해 주세요."}],
                         max_tokens=16)
    print(f"evict     prompt={ue.get('prompt_tokens')} cached={ue.get('prompt_tokens_details',{}).get('cached_tokens')} {te:7.1f}s")

    cold, uc, tc = chat(args.base, q2)
    print(f"cold      prompt={uc.get('prompt_tokens')} cached={uc.get('prompt_tokens_details',{}).get('cached_tokens')} {tc:7.1f}s")

    same = warm == cold
    ratio = difflib.SequenceMatcher(None, warm, cold).ratio()
    print(f"\nidentical={same}  char_similarity={ratio:.4f}  speedup={tc/tw:.1f}x")
    if not same:
        print(f"\nwarm: {warm!r}\ncold: {cold!r}")

    out = {"identical": same, "char_similarity": ratio,
           "warm": {"content": warm, "usage": uw, "elapsed_s": tw},
           "cold": {"content": cold, "usage": uc, "elapsed_s": tc},
           "seed": {"content": a1, "usage": u1, "elapsed_s": t1}}
    with open(os.path.join(args.out_dir, "warm-cold.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
