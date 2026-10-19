# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 text-to-image on Neuron via the Omni entrypoint.

Usage:
    python examples/hunyuan_image3/run.py                          # 1024x1024, 50 steps, TP=16
    python examples/hunyuan_image3/run.py --height 768 --width 1280
    python examples/hunyuan_image3/run.py --tensor-parallel-size 32
    python examples/hunyuan_image3/run.py --cfg-parallel-size 2       # TP=16 x CFG=2
    python examples/hunyuan_image3/run.py --profile                # warm runs after the first
"""

import argparse
import os
import statistics
import tempfile
import time

import torch
import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

# torch_neuronx patches F.gelu with a wrapper around the C builtin;
# Dynamo cannot trace through the C function in fullgraph mode.
torch.nn.functional.gelu = torch.ops.aten.gelu.default

parser = argparse.ArgumentParser(description="HunyuanImage-3.0 text-to-image on Neuron")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--guidance-scale", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model-path", type=str, default="tencent/HunyuanImage-3.0")
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=None,
    help="Override the stage config's TP degree (and use cores 0..TP-1).",
)
parser.add_argument(
    "--cfg-parallel-size",
    type=int,
    default=None,
    help="Override the stage config's CFG parallelism (2: the conditional and the "
    "unconditional rows on separate TP groups).",
)
parser.add_argument(
    "--stage-config",
    type=str,
    default=None,
    help="Stage config YAML. Defaults to hunyuan_image3_stage.yaml next to this script.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default='A neon shop sign that reads "HUNYUAN IMAGE 3", rainy night, reflections on wet pavement',
)
parser.add_argument("--output", type=str, default="hunyuan_image3.png")
parser.add_argument(
    "--profile",
    action="store_true",
    help="After the first (compiling) request, time --profile-runs warm requests.",
)
parser.add_argument("--profile-runs", type=int, default=3)
args = parser.parse_args()


def _stage_config() -> str:
    path = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hunyuan_image3_stage.yaml"
    )
    if args.tensor_parallel_size is None and args.cfg_parallel_size is None:
        return path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    stage = cfg["stage_args"][0]
    parallel = stage["engine_args"]["parallel_config"]
    if args.tensor_parallel_size is not None:
        parallel["tensor_parallel_size"] = args.tensor_parallel_size
    if args.cfg_parallel_size is not None:
        parallel["cfg_parallel_size"] = args.cfg_parallel_size
    num_cores = parallel["tensor_parallel_size"] * parallel.get("cfg_parallel_size", 1)
    stage["runtime"]["devices"] = f"0-{num_cores - 1}"
    out = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, out)
    out.close()
    return out.name


stage_cfg = _stage_config()
with open(stage_cfg) as f:
    _parallel = yaml.safe_load(f)["stage_args"][0]["engine_args"]["parallel_config"]
_tp = _parallel["tensor_parallel_size"]
_cfg = _parallel.get("cfg_parallel_size", 1)
# Set env vars before Omni spawns workers so they are inherited.
env_profiles.apply(env_profiles.HUNYUAN_IMAGE3_T2I, env_profiles.thread_limits(_tp * _cfg))


def main():
    # Loading 160 GB of weights and a cold compile exceed vllm-omni's default 600 s
    # worker handshake; see examples/wan22/run.py.
    cold_timeout = int(os.environ.get("HUNYUAN_IMAGE3_HANDSHAKE_TIMEOUT_S", "7200"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as _sdp

        if getattr(_sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0) < cold_timeout:
            _sdp._HANDSHAKE_POLL_TIMEOUT_S = cold_timeout
    except Exception as e:
        print(f"[init] could not patch handshake timeout ({e!r}); using default 600s")

    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=cold_timeout,
        init_timeout=cold_timeout,
    )
    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    size = f"{args.height}x{args.width}"
    print(f"Generating: {size}, {args.steps} steps, guidance {args.guidance_scale}")
    t0 = time.perf_counter()
    result = omni.generate({"prompt": args.prompt}, params)
    print(f"[run] first request (includes compilation): {time.perf_counter() - t0:.2f}s")

    if args.profile:
        times = []
        for i in range(args.profile_runs):
            t0 = time.perf_counter()
            result = omni.generate({"prompt": args.prompt}, params)
            times.append(time.perf_counter() - t0)
            print(f"[profile] run {i + 1}/{args.profile_runs}: {times[-1]:.3f}s")
        print("=" * 60)
        print(
            f"[profile] {size}, {args.steps} steps, TP={_tp} CFG={_cfg}: mean {statistics.mean(times):.3f}s, "
            f"min {min(times):.3f}s, max {max(times):.3f}s over {len(times)} runs"
        )
        print("=" * 60)

    image = result[0].request_output.images[0]
    if hasattr(image, "save"):
        image.save(args.output)
    else:
        from PIL import Image

        Image.fromarray(image).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
