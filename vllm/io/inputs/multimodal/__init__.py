# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .base import MultiModalPlaceholderMap
from .hasher import MultiModalHashDict
from .hasher import MultiModalHasher
from .inputs import BatchedTensorInputs
from .inputs import ModalityData
from .inputs import MultiModalDataBuiltins
from .inputs import MultiModalDataDict
from .inputs import MultiModalKwargs
from .inputs import MultiModalPlaceholderDict
from .inputs import NestedTensors
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global [`MultiModalRegistry`][vllm.io.inputs.multimodal.registry.MultiModalRegistry]
is used by model runners to dispatch data processing according to the target
model.

Info:
    [mm_processing](../../../design/mm_processing.html)
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHashDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
