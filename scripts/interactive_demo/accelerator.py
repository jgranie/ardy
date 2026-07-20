# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPS concurrency helpers for the interactive demo.

Metal-backed PyTorch work must not overlap across the demo's callback, prewarm,
generation, and playback threads. Motion generation remains concurrent on CUDA
and CPU; MPS generation is serialized and its playback state is offloaded. The
shared text encoder is serialized separately because the GUI can move it to MPS
even while the motion model remains on CPU.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import wraps

import torch


def uses_serialized_accelerator(device) -> bool:
    """Whether demo accelerator work must use the shared serialization lock."""
    return torch.device(device).type == "mps"


def accelerator_guard(owner):
    """Return the demo's shared lock on MPS and a no-op context elsewhere."""
    if uses_serialized_accelerator(owner.device):
        return owner._accelerator_lock
    return nullcontext()


def serialized_accelerator(method):
    """Serialize a demo method on MPS without changing CPU/CUDA concurrency."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with accelerator_guard(self):
            return method(self, *args, **kwargs)

    # Makes coverage of accelerator entry points testable without executing
    # heavyweight model/demo code.
    wrapped.__accelerator_serialized__ = True
    return wrapped


def serialized_text_encoder(method):
    """Serialize access to the shared, runtime-movable text encoder.

    The model may run on CPU while the GUI moves the encoder to MPS, so this
    lock cannot be conditional on the model device. It does not serialize
    CPU/CUDA motion generation.
    """

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._accelerator_lock:
            return method(self, *args, **kwargs)

    wrapped.__accelerator_serialized__ = True
    return wrapped


def playback_device(model_device) -> str:
    """Device for state read by playback and GUI threads."""
    resolved = torch.device(model_device)
    return "cpu" if resolved.type == "mps" else str(resolved)


def move_to_playback_device(tensor, model_device):
    """Detach and offload a tensor only when MPS playback isolation needs it."""
    if tensor is None or not uses_serialized_accelerator(model_device):
        return tensor
    return tensor.detach().to(device="cpu")


def move_playback_tensors(model_device, **tensors):
    """Move a named group of session tensors across the playback boundary."""
    return {name: move_to_playback_device(tensor, model_device) for name, tensor in tensors.items()}


def create_playback_skeleton(skeleton, model_device):
    """Create an independent CPU skeleton for MPS visualization.

    Reloading from the skeleton's asset folder avoids copying any live MPS
    buffers. Other backends retain the exact skeleton object they already use.
    """
    if not uses_serialized_accelerator(model_device):
        return skeleton
    return type(skeleton)(folder=skeleton.folder, name=skeleton.name, load=True)
