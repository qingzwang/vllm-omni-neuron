#!/usr/bin/env python3
"""Per-stage latency breakdown of a 4096x4096 t2i request (concurrency 1).

Server must run with H3_PROF_FILE=<prof file> (plus SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1 for SGLang).
All timestamps are wall clock (time.time()) so client / API process / GPU worker share one timeline.

python3 profile4096.py --framework vllm --url http://127.0.0.1:8000 --prof-file /tmp/h3prof_vllm.jsonl --out x.json
"""

import argparse
import base64
import http.client
import json
import statistics
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).absolute().parent
PROMPT = json.loads((HERE / "request.t2i.json").read_text())["prompt"]


def new_lines(path, offset):
    p = Path(path)
    if not p.exists():
        return [], offset
    with p.open() as f:
        f.seek(offset)
        data = f.read()
        offset = f.tell()
    return [json.loads(l) for l in data.splitlines() if l.strip()], offset


def request(url, body):
    u = urllib.parse.urlparse(url)
    payload = json.dumps(body).encode()
    t = {}
    t["c_start"] = time.time()
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    conn.request("POST", "/v1/images/generations", body=payload, headers={"Content-Type": "application/json"})
    t["c_sent"] = time.time()
    resp = conn.getresponse()
    t["c_headers"] = time.time()
    raw = resp.read()
    t["c_body"] = time.time()
    conn.close()
    obj = json.loads(raw)
    t["c_json"] = time.time()
    img = base64.b64decode(obj["data"][0]["b64_json"])
    t["c_b64"] = time.time()
    return t, obj, len(raw), len(img)


def breakdown_vllm(t, recs):
    api = next(r for r in recs if r["kind"] == "api")
    pipe = next(r for r in recs if r["kind"] == "pipeline")
    m = pipe["marks"]
    rows = []
    rows.append(("client -> API handler (HTTP + request parsing)", api["t_handler_start"] - t["c_start"]))
    rows.append(("API -> engine -> GPU worker dispatch", m[0][1] - api["t_handler_start"]))
    prev = m[0][1]
    labels = {
        "prepare": "request parsing + noise generation (CPU)",
        "text_encode": "text encoder (Qwen3-VL, TP2)",
        "denoise_setup": "DiT packing / RoPE table prep",
        "unpatchify": "latent unpatchify",
        "vae_decode": "VAE decode (tiled, 2 GPUs)",
        "quantize_resize": "quantize to uint8",
        "d2h": "GPU -> CPU copy (uint8 image)",
    }
    for name, ts in m[1:]:
        lab = f"DiT denoise {name}" if name.startswith("step_") else labels.get(name, name)
        rows.append((lab, ts - prev))
        prev = ts
    rows.append(("worker -> API process (IPC transfer + PIL conversion)", api["t_engine_done"] - prev))
    rows.append(("extract images", api["t_extract"] - api["t_engine_done"]))
    rows.append(("PNG encode + base64 (API process)", api["t_encoded"] - api["t_extract"]))
    rows.append(("JSON serialize (orjson)", api["t_json"] - api["t_encoded"]))
    rows.append(("response send -> client headers", t["c_headers"] - api["t_json"]))
    rows.append(("client body download", t["c_body"] - t["c_headers"]))
    rows.append(("client JSON parse", t["c_json"] - t["c_body"]))
    rows.append(("client base64 decode", t["c_b64"] - t["c_json"]))
    return rows


def breakdown_sglang(t, recs, dump):
    api = next(r for r in recs if r["kind"] == "api")
    fwd = next(r for r in recs if r["kind"] == "worker_forward")
    save = next(r for r in recs if r["kind"] == "worker_save_outputs")
    rows = []
    rows.append(("client -> API handler (HTTP + request parsing)", api["t_handler_start"] - t["c_start"]))
    rows.append(("API: build request", api["t_submit"] - api["t_handler_start"]))
    rows.append(("API -> scheduler -> GPU worker dispatch", fwd["t0"] - api["t_submit"]))
    stage_sum = 0.0
    steps = {s["step"]: s["duration_ms"] / 1000 for s in dump["denoise_steps_ms"]}
    labels = {
        "MiniMaxH3TextEncodingStage": "text encoder (Qwen3-VL, fold)",
        "MiniMaxH3LatentPreparationStage": "latent / noise preparation",
        "MiniMaxH3DecodingStage": "VAE decode (+ uint8, D2H)",
    }
    for st in dump["steps"]:
        name, d = st["name"], st["duration_ms"] / 1000
        stage_sum += d
        if name == "MiniMaxH3DenoisingStage":
            ssum = sum(steps.values())
            rows.append(("DiT denoise setup (packing / RoPE / AdaLN plan)", d - ssum))
            for k in sorted(steps):
                rows.append((f"DiT denoise step_{k}", steps[k]))
        elif d >= 0.0005 or name in labels:
            rows.append((labels.get(name, name), d))
    rows.append(("pipeline overhead outside stages", (fwd["t1"] - fwd["t0"]) - stage_sum))
    rows.append(("worker: forward -> output encoding", save["t0"] - fwd["t1"]))
    rows.append(("PNG encode (GPU worker)", save["t1"] - save["t0"]))
    rows.append(("worker -> API process (IPC transfer)", api["t_result"] - save["t1"]))
    rows.append(("base64 encode (API)", api["t_b64"] - api["t_result"]))
    rows.append(("build response object", api["t_return"] - api["t_b64"]))
    rows.append(("JSON serialize + response send -> client headers", t["c_headers"] - api["t_return"]))
    rows.append(("client body download", t["c_body"] - t["c_headers"]))
    rows.append(("client JSON parse", t["c_json"] - t["c_body"]))
    rows.append(("client base64 decode", t["c_b64"] - t["c_json"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--prof-file", required=True)
    ap.add_argument("--size", default="4096x4096")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--same-prompt", action="store_true", help="reuse one prompt (lets SGLang hit its conditioning cache)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    offset = Path(args.prof_file).stat().st_size if Path(args.prof_file).exists() else 0
    runs = []
    for i in range(args.warmup + args.iters):
        prompt = PROMPT if args.same_prompt else f"{PROMPT} (Variation {i}, {time.time_ns()}.)"
        body = {"model": "MiniMaxAI/MiniMax-H3-Image", "prompt": prompt, "size": args.size, "seed": 42,
                "generator_device": "cpu", "response_format": "b64_json", "output_format": "png"}
        dump_path = None
        if args.framework == "sglang":
            dump_path = str(out.parent / f"sgl_perf_{i}.json")
            body["perf_dump_path"] = dump_path
        t, obj, nraw, nimg = request(args.url, body)
        time.sleep(0.5)  # let the server flush its records
        recs, offset = new_lines(args.prof_file, offset)
        e2e = t["c_b64"] - t["c_start"]
        if i < args.warmup:
            print(f"warmup {i}: e2e {e2e:.3f}s", flush=True)
            continue
        if args.framework == "vllm":
            rows = breakdown_vllm(t, recs)
        else:
            rows = breakdown_sglang(t, recs, json.loads(Path(dump_path).read_text()))
        total = sum(d for _, d in rows)
        runs.append({"e2e_s": e2e, "rows": rows, "response_mb": nraw / 1e6, "png_mb": nimg / 1e6})
        print(f"iter {i}: e2e {e2e:.3f}s (sum of rows {total:.3f}s)", flush=True)

    # median per row
    labels = [r for r, _ in runs[0]["rows"]]
    med = {lab: statistics.median(run["rows"][k][1] for run in runs) for k, lab in enumerate(labels)}
    e2e = statistics.median(r["e2e_s"] for r in runs)
    result = {"framework": args.framework, "size": args.size, "iters": len(runs), "e2e_median_s": e2e,
              "row_medians_s": med, "runs": runs}
    out.write_text(json.dumps(result, indent=1))
    print(f"\n{args.framework} {args.size}: median e2e {e2e:.3f}s  (response {runs[0]['response_mb']:.1f}MB JSON, "
          f"PNG {runs[0]['png_mb']:.1f}MB)")
    for lab in labels:
        print(f"  {med[lab]*1000:9.1f} ms  {lab}")


if __name__ == "__main__":
    main()
