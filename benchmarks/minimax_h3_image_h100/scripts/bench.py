#!/usr/bin/env python3
"""Concurrency-1 latency benchmark for MiniMax-H3-Image servers (vLLM-Omni / SGLang).

python3 bench.py --framework vllm --url http://127.0.0.1:8000 --out results/vllm.json
"""

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).absolute().parent
PROMPT = json.loads((HERE / "request.t2i.json").read_text())["prompt"]

# (label, size, native_canvas)
# native_canvas=True -> disable the 2048^2 minimum canvas so 1024x1024 is really generated at 1024x1024
CASES = [
    ("1024_native", "1024x1024", True),
    ("1024_default", "1024x1024", False),  # model default: generated on 2048 canvas then LANCZOS down
    ("2048", "2048x2048", False),
    ("4096", "4096x4096", False),
]


def build_body(framework, size, native, seed):
    body = {
        "model": "MiniMaxAI/MiniMax-H3-Image",
        "prompt": PROMPT,
        "size": size,
        "seed": seed,
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
    }
    if native:
        if framework == "vllm":
            body["extra_params"] = {"min_generation_area": 1024}  # image endpoint reads extra_params (extra_args is ignored)
        else:
            body["min_generation_area"] = 1024
    return body


def one(url, body, save=None):
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/images/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        raw = r.read()
    dt = time.perf_counter() - t0
    payload = json.loads(raw)
    img = base64.b64decode(payload["data"][0]["b64_json"])
    extra = {k: v for k, v in payload.items() if k != "data"}
    extra.update({k: v for k, v in payload["data"][0].items() if k != "b64_json"})
    if save:
        Path(save).write_bytes(img)
    try:
        from PIL import Image
        wh = Image.open(io.BytesIO(img)).size
    except Exception:
        wh = None
    return dt, len(img), wh, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgdir = out.parent / f"images_{args.framework}"
    imgdir.mkdir(exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    wanted = args.cases.split(",")
    for label, size, native in CASES:
        if label not in wanted:
            continue
        print(f"== {args.framework} {label} ({size}, native_canvas={native})", flush=True)
        for i in range(args.warmup):
            dt, nbytes, wh, _ = one(args.url, build_body(args.framework, size, native, 42))
            print(f"  warmup {i}: {dt:.3f}s  {wh}  {nbytes/1e6:.1f}MB", flush=True)
        lat, extras = [], []
        for i in range(args.iters):
            save = imgdir / f"{label}.png" if i == 0 else None
            dt, nbytes, wh, extra = one(args.url, build_body(args.framework, size, native, 42), save)
            lat.append(dt)
            extras.append(extra)
            print(f"  iter {i}: {dt:.3f}s  {wh}  {nbytes/1e6:.1f}MB  {json.dumps(extra)[:300]}", flush=True)
        results[label] = {
            "size": size,
            "native_canvas": native,
            "latencies_s": lat,
            "median_s": statistics.median(lat),
            "mean_s": statistics.mean(lat),
            "min_s": min(lat),
            "max_s": max(lat),
            "png_mb": nbytes / 1e6,
            "image_size": wh,
            "server_extra": extras,
        }
        print(f"  -> median {statistics.median(lat):.3f}s  min {min(lat):.3f}s", flush=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
