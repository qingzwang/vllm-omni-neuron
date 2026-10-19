# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 text-to-image and image editing on Neuron via the Omni entrypoint.

Usage:
    python examples/qwen_image_2_1/run.py                        # 1024x1024, 40 steps
    python examples/qwen_image_2_1/run.py --dev                  # 512x512, 8 steps
    python examples/qwen_image_2_1/run.py --tensor-parallel-size 4
    python examples/qwen_image_2_1/run.py --tensor-parallel-size 4 --context-parallel-size 2
    python examples/qwen_image_2_1/run.py --profile              # warm runs after the first
    python examples/qwen_image_2_1/run.py --image in.png --prompt "Make it snow"   # edit
    python examples/qwen_image_2_1/run.py --image a.png b.png \
        --prompt "Put the person from image 1 into the scene of image 2"         # multi-image
    python examples/qwen_image_2_1/run.py --image in.png --mask-box 100 80 400 360 \
        --prompt "Replace the object circled in red with a vase of flowers"     # region mark
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

parser = argparse.ArgumentParser(description="Qwen-Image 2.1 text-to-image and editing on Neuron")
parser.add_argument("--dev", action="store_true", help="Dev mode: 512x512, 8 steps")
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model-path", type=str, default="Qwen/Qwen-Image-2.1")
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=None,
    help="Override the stage config's TP degree (and use cores 0..TP-1).",
)
parser.add_argument(
    "--context-parallel-size",
    type=int,
    default=None,
    help="Override the stage config's context parallelism (parallel_config.ring_degree).",
)
parser.add_argument(
    "--stage-config",
    type=str,
    default=None,
    help="Stage config YAML. Defaults to qwen_image_2_1_stage.yaml next to this script.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
)
parser.add_argument(
    "--image",
    type=str,
    nargs="+",
    default=None,
    help="Condition image(s) (up to 10) to edit with --prompt as the instruction; refer "
    "to them as image 1, image 2, ... The output size defaults to the last image's "
    "aspect ratio at --edit-resolution.",
)
parser.add_argument(
    "--edit-resolution",
    type=int,
    default=None,
    help="Side of the square area edit inputs (and default outputs) are resized to "
    "(model_config.edit_resolution, default 1024).",
)
parser.add_argument(
    "--negative-prompt",
    type=str,
    default=None,
    help="Negative prompt; with --true-cfg-scale > 1 enables true classifier-free "
    "guidance (off by default: the model is meant to be sampled without it).",
)
parser.add_argument("--true-cfg-scale", type=float, default=None)
parser.add_argument(
    "--mask",
    type=str,
    nargs="+",
    default=None,
    help="Mask image(s) of regions to edit on the last --image (white, or transparent "
    "for images with alpha, marks a region); see --mask-mode.",
)
parser.add_argument(
    "--mask-box",
    type=float,
    nargs=4,
    action="append",
    metavar=("X0", "Y0", "X1", "Y1"),
    help="A box region on the last --image, in its pixels (repeatable; after --mask).",
)
parser.add_argument(
    "--mask-mode",
    choices=("outline", "image"),
    default=None,
    help="outline (default): draw each region's outline on the image in its color "
    "(red, green, blue, ...); image: pass the regions as an extra mask image after the inputs.",
)
parser.add_argument("--mask-colors", type=str, nargs="+", default=None)
parser.add_argument(
    "--mask-composite",
    action="store_true",
    help="Keep the original pixels outside the regions in the output.",
)
parser.add_argument("--output", type=str, default="qwen_image_2_1.png")
parser.add_argument(
    "--profile",
    action="store_true",
    help="After the first (compiling) request, time --profile-runs warm requests.",
)
parser.add_argument("--profile-runs", type=int, default=3)
args = parser.parse_args()


def _stage_config() -> str:
    path = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qwen_image_2_1_stage.yaml"
    )
    if (
        args.tensor_parallel_size is None
        and args.context_parallel_size is None
        and args.edit_resolution is None
    ):
        return path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    stage = cfg["stage_args"][0]
    parallel = stage["engine_args"]["parallel_config"]
    if args.tensor_parallel_size is not None:
        parallel["tensor_parallel_size"] = args.tensor_parallel_size
    if args.context_parallel_size is not None:
        # CP is sized with ring_degree; vllm-omni derives sequence_parallel_size from it.
        parallel["ring_degree"] = args.context_parallel_size
    if args.edit_resolution is not None:
        model_config = stage["engine_args"].setdefault("model_config", {})
        model_config["edit_resolution"] = args.edit_resolution
    num_cores = parallel["tensor_parallel_size"] * parallel.get("ring_degree", 1)
    stage["runtime"]["devices"] = f"0-{num_cores - 1}"
    out = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, out)
    out.close()
    return out.name


stage_cfg = _stage_config()
with open(stage_cfg) as f:
    _parallel = yaml.safe_load(f)["stage_args"][0]["engine_args"]["parallel_config"]
_tp = _parallel["tensor_parallel_size"]
_cp = _parallel.get("ring_degree", 1)
# Set env vars before Omni spawns workers so they are inherited.
env_profiles.apply(env_profiles.QWEN_IMAGE21_T2I, env_profiles.thread_limits(_tp * _cp))


def main():
    # A cold compile exceeds vllm-omni's default 600 s worker handshake; see examples/wan22/run.py.
    cold_timeout = int(os.environ.get("QWEN_IMAGE21_HANDSHAKE_TIMEOUT_S", "3600"))
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

    height, width, steps = (512, 512, 8) if args.dev else (1024, 1024, 40)
    request = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        request["negative_prompt"] = args.negative_prompt
    if args.image:
        from PIL import Image

        images = [Image.open(path) for path in args.image]
        request["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
        regions = [Image.open(path) for path in args.mask or []] + list(args.mask_box or [])
        if regions:
            request["multi_modal_data"]["mask_image"] = regions
        # The pipeline derives the output size from the condition image.
        height = width = None
    mask_options = {
        "mask_mode": args.mask_mode,
        "mask_colors": args.mask_colors,
        "mask_composite": args.mask_composite or None,
    }
    params = OmniDiffusionSamplingParams(
        extra_args={k: v for k, v in mask_options.items() if v is not None},
        height=args.height or height,
        width=args.width or width,
        num_inference_steps=args.steps or steps,
        seed=args.seed,
        true_cfg_scale=args.true_cfg_scale,
    )
    size = f"{params.height}x{params.width}" if params.height else "edit (image aspect)"
    print(f"Generating: {size}, {params.num_inference_steps} steps")
    t0 = time.perf_counter()
    result = omni.generate(request, params)
    print(f"[run] first request (includes compilation): {time.perf_counter() - t0:.2f}s")

    if args.profile:
        times = []
        for i in range(args.profile_runs):
            t0 = time.perf_counter()
            result = omni.generate(request, params)
            times.append(time.perf_counter() - t0)
            print(f"[profile] run {i + 1}/{args.profile_runs}: {times[-1]:.3f}s")
        print("=" * 60)
        print(
            f"[profile] {size}, {params.num_inference_steps} steps, "
            f"TP={_tp} CP={_cp}: mean {statistics.mean(times):.3f}s, min {min(times):.3f}s, "
            f"max {max(times):.3f}s over {len(times)} runs"
        )
        print("=" * 60)

    images = result[0].request_output.images
    image = images[0]
    if hasattr(image, "save"):
        image.save(args.output)
    else:
        from PIL import Image

        Image.fromarray(image).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
