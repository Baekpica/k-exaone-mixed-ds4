#!/usr/bin/env python3
"""Phase C: run the fixture set against one artifact and record what the work
order asks for.

Starts llama-server once per artifact (a 87 GiB model must not be reloaded 32
times) and drives it over the OpenAI-compatible endpoint with greedy decoding.

Records per prompt: output text and token ids, output length, invalid UTF-8 /
broken Hangul jamo, repetition rate, JSON parse success, and timings. When
--reference is given, also first-divergence token index and token agreement
ratio against that run.
"""
import argparse, json, re, subprocess, sys, time, unicodedata, signal, os
from pathlib import Path
import urllib.request, urllib.error

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "llama.cpp/build-sm120/bin/llama-server"

# Hangul: composed syllables U+AC00-D7A3; isolated jamo U+1100-11FF / U+3130-318F.
# A healthy Korean generation is composed syllables. A collapse into isolated
# jamo is the failure mode the work order calls out for low-bit Korean models.
JAMO = re.compile(r"[ᄀ-ᇿ㄰-㆏]")
HANGUL = re.compile(r"[가-힣]")


def haystack(needle, target_chars=24000, seed=0):
    """Filler + needle, for the long_context fixtures."""
    import random
    rnd = random.Random(seed)
    filler = [
        "본 문서는 내부 검토용으로 작성되었으며 외부 배포를 금합니다.",
        "각 부서는 분기별 실적을 취합하여 다음 달 첫째 주까지 제출한다.",
        "The committee reviewed the quarterly figures and noted no material variance.",
        "Operational metrics remained within the agreed tolerance band throughout.",
        "회의록은 작성 후 7일 이내에 참석자 전원에게 회람한다.",
        "All findings are provisional until validated by the audit team.",
    ]
    parts, n = [], 0
    while n < target_chars:
        s = rnd.choice(filler)
        parts.append(s)
        n += len(s) + 1
    pos = rnd.randrange(len(parts) // 4, 3 * len(parts) // 4)
    parts.insert(pos, needle)
    return "\n".join(parts)


def repetition_rate(text, n=3):
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    from collections import Counter
    c = Counter(grams)
    return max(c.values()) / len(grams)


def jamo_ratio(text):
    h, j = len(HANGUL.findall(text)), len(JAMO.findall(text))
    return j / (h + j) if (h + j) else 0.0


def wait_ready(port, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"server not ready after {timeout}s")


def post(port, path, payload, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fixtures", default=str(ROOT / "fixtures/prompts.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="a previous run's JSON, to diff against")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--split-mode", default="layer")
    ap.add_argument("--thinking", action="store_true",
                    help="leave K-EXAONE's reasoning mode on. Off by default: with "
                         "reasoning on the model routinely spends >1500 tokens in "
                         "reasoning_content before content starts, so a bounded "
                         "max_tokens truncates mid-thought and every format check "
                         "scores zero for reasons that have nothing to do with quant "
                         "quality. Reported separately when enabled.")
    a = ap.parse_args()

    fixtures = [json.loads(l) for l in open(a.fixtures, encoding="utf-8") if l.strip()]
    for f in fixtures:
        if f.get("needs_haystack"):
            f["prompt"] = f["prompt"] + "\n\n---\n" + haystack(f["needle"])

    cmd = [str(SERVER), "-m", a.model, "--port", str(a.port), "-ngl", str(a.ngl),
           "-c", str(a.ctx), "-sm", a.split_mode, "--host", "127.0.0.1",
           "-np", "1", "--no-warmup"]
    print("[server]", " ".join(cmd), flush=True)
    log = open(Path(a.out).with_suffix(".server.log"), "w")
    t_load0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    results = {"label": a.label, "model": a.model,
               "model_bytes": Path(a.model).stat().st_size,
               "ctx": a.ctx, "max_tokens": a.max_tokens,
               "thinking": a.thinking, "runs": []}
    try:
        load_s = wait_ready(a.port)
        results["load_seconds"] = load_s
        print(f"[server] ready in {load_s:.1f}s", flush=True)

        for i, f in enumerate(fixtures, 1):
            t0 = time.time()
            try:
                payload = {
                    "messages": [{"role": "user", "content": f["prompt"]}],
                    "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 0,
                    "max_tokens": a.max_tokens, "stream": False,
                }
                if not a.thinking:
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                r = post(a.port, "/v1/chat/completions", payload)
                dt = time.time() - t0
                msg = r["choices"][0]["message"]
                txt = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                usage = r.get("usage", {})
                tim = r.get("timings", {})
                rec = {
                    "id": f["id"], "category": f["category"],
                    "output": txt,
                    "output_chars": len(txt),
                    "reasoning_chars": len(reasoning),
                    "thinking": a.thinking,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "wall_seconds": dt,
                    "prefill_tps": tim.get("prompt_per_second"),
                    "decode_tps": tim.get("predicted_per_second"),
                    "ttft_ms": tim.get("prompt_ms"),
                    "jamo_ratio": jamo_ratio(txt),
                    "repetition_3gram": repetition_rate(txt),
                    "valid_utf8": True,
                    "finish_reason": r["choices"][0].get("finish_reason"),
                }
                try:
                    txt.encode("utf-8").decode("utf-8")
                except Exception:
                    rec["valid_utf8"] = False
                if f["category"] == "json_tool":
                    m = re.search(r"[\[{].*[\]}]", txt, re.S)
                    try:
                        json.loads(m.group(0) if m else txt)
                        rec["json_parse_ok"] = True
                    except Exception:
                        rec["json_parse_ok"] = False
                if f.get("needle"):
                    key = re.sub(r"\s+", "", f["needle"])[:40]
                    rec["needle_found"] = re.sub(r"\s+", "", txt).find(key) >= 0
                results["runs"].append(rec)
                print(f"  [{i:2d}/{len(fixtures)}] {f['id']:12s} "
                      f"{rec['completion_tokens'] or 0:4d} tok  "
                      f"{rec['decode_tps'] or 0:6.2f} tok/s  jamo={rec['jamo_ratio']:.3f}",
                      flush=True)
            except Exception as e:
                print(f"  [{i:2d}] {f['id']} FAILED: {type(e).__name__}: {str(e)[:160]}",
                      flush=True)
                results["runs"].append({"id": f["id"], "category": f["category"],
                                        "error": f"{type(e).__name__}: {e}"})
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log.close()

    # --- diff against a reference run ---
    if a.reference and Path(a.reference).exists():
        ref = {r["id"]: r for r in json.load(open(a.reference))["runs"]}
        for r in results["runs"]:
            b = ref.get(r["id"])
            if not b or "output" not in r or "output" not in b:
                continue
            x, y = r["output"], b["output"]
            fd = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                      min(len(x), len(y)) if len(x) != len(y) else None)
            r["first_divergence_char"] = fd
            xs, ys = x.split(), y.split()
            agree = sum(1 for p, q in zip(xs, ys) if p == q)
            r["word_agreement_ratio"] = agree / max(1, max(len(xs), len(ys)))
            r["identical_to_reference"] = (x == y)

    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    ok = [r for r in results["runs"] if "error" not in r]
    print(f"\n[{a.label}] {len(ok)}/{len(fixtures)} completed -> {a.out}")
    if ok:
        import statistics as st
        print(f"  mean decode  {st.mean([r['decode_tps'] or 0 for r in ok]):.2f} tok/s")
        print(f"  mean jamo    {st.mean([r['jamo_ratio'] for r in ok]):.4f}")
        print(f"  mean rep3    {st.mean([r['repetition_3gram'] for r in ok]):.4f}")
        trunc = [r for r in ok if r.get("finish_reason") == "length"]
        print(f"  truncated    {len(trunc)}/{len(ok)} (finish_reason=length)")
        js = [r for r in ok if "json_parse_ok" in r]
        if js:
            print(f"  json ok      {sum(r['json_parse_ok'] for r in js)}/{len(js)}")
        nd = [r for r in ok if "needle_found" in r]
        if nd:
            print(f"  needle found {sum(r['needle_found'] for r in nd)}/{len(nd)}")
        if a.reference:
            idn = [r for r in ok if r.get("identical_to_reference") is not None]
            if idn:
                print(f"  identical    {sum(bool(r['identical_to_reference']) for r in idn)}/{len(idn)}")


if __name__ == "__main__":
    main()
