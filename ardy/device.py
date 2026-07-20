# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime device selection shared by ARDY command-line and demo entry points."""

from __future__ import annotations

from typing import Union

import torch

DeviceLike = Union[str, torch.device]


def mps_is_available() -> bool:
    """Return whether this PyTorch build can currently use the MPS backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def available_device_types() -> list[str]:
    """Available user-facing device types, ordered by ARDY preference."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if mps_is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def select_device(device: DeviceLike | None = None) -> str:
    """Resolve a requested device, defaulting to CUDA, then MPS, then CPU.

    Explicit CUDA/MPS requests fail early when the backend is unavailable,
    instead of surfacing a less useful tensor-placement error later.
    """
    if device is None or str(device).strip().lower() in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if mps_is_available():
            return "mps"
        return "cpu"

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device!r}")
    if resolved.type == "mps" and not mps_is_available():
        raise RuntimeError(f"MPS device requested but MPS is unavailable: {device!r}")
    if resolved.type not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported ARDY device {device!r}; choose auto, cuda, mps, or cpu.")
    return str(resolved)


def device_type(device: DeviceLike) -> str:
    """Return the backend type for a resolved device string/device."""
    return torch.device(device).type


def supports_tensorrt(device: DeviceLike) -> bool:
    """TensorRT acceleration is meaningful only for CUDA model placement."""
    return device_type(device) == "cuda"


def clear_device_cache(device: DeviceLike) -> None:
    """Release allocator caches for the selected accelerator when supported."""
    backend = device_type(device)
    if backend == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps" and mps_is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
