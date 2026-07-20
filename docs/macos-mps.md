# ARDY on Apple Silicon with MPS

This repository includes a Python 3.11 dependency lock for the interactive
demo on Apple Silicon. TensorRT is CUDA-only and must not be installed on
macOS. The default BF16 text encoder on MPS requires macOS 14 or newer; on an
older supported macOS release, run the text encoder service with `--fp32`.

## Reproducible environment

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements-macos.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

The lock is generated from the `demo` extra only and constrains the validated
MPS build to PyTorch 2.11.0:

```bash
uv pip compile \
  --python .venv/bin/python \
  --extra demo \
  --constraints requirements-macos.constraints.txt \
  --output-file requirements-macos.lock.txt \
  pyproject.toml
```

## Verification and use

```bash
.venv/bin/python -m unittest discover -s tests -v

# Eager MPS is the recommended mode on this machine.
LOCAL_CACHE=true .venv/bin/python scripts/run_demo.py --no-compile --device mps

# Command-line text-to-motion generation.
LOCAL_CACHE=true .venv/bin/python scripts/generate.py \
  "A person walks forward." \
  --model core8 \
  --device mps
```

Automatic device selection follows `cuda -> mps -> cpu`. In the interactive
demo, TensorRT choices appear only when the motion model is on CUDA. On MPS,
the acceleration choices are `None` and `torch.compile`, and the text-encoder
choices include `mps / bfloat16`.

The first local text-encoder load downloads the gated Llama 3 8B safetensor
shards plus the two LLM2Vec adapters. It does not require the duplicate
`original/consolidated.00.pth` file. Hugging Face access to
`meta-llama/Meta-Llama-3-8B-Instruct` must already be granted.

In the interactive demo, MPS accelerator entry points are serialized across
callback threads and playback/visualization state is kept on CPU. Model
inference remains on MPS; CPU and CUDA inference placement and precision are
unchanged.

## Reference measurements

Observed on an M3 Max (30 GPU cores, 36 GB unified memory) with PyTorch 2.11:

- Core Horizon8 eager, 10 denoising steps: 0.206 s for an 8-frame horizon
  (0.4 s of motion at 20 FPS), with about 866 MiB allocated by MPS.
- LLM2Vec/Llama 3 8B: BF16 weights on MPS, about 14.5 GiB allocated; a first
  prompt embedding took 2.47 s.
- Full cached CLI startup plus one text-to-motion sample peaked at about
  22.6 GB and completed in 83 s; most of this time was model loading.
- Core Horizon40 in the eager Viser demo generated its first 40-frame horizon
  in 2.01 s and regenerated after a prompt update in 2.28 s.
- `torch.compile` works, but its warmed Core Horizon8 time was 0.322 s versus
  0.206 s eager on this setup, so `--no-compile` remains the default
  recommendation.

These figures are indicative and vary with Metal shader caches and current
system memory pressure.
