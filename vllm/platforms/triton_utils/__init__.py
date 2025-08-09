# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.triton_tuils.importing import HAS_TRITON
from vllm.platforms.triton_tuils.importing import TritonLanguagePlaceholder
from vllm.platforms.triton_tuils.importing import TritonPlaceholder

if HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()

__all__ = ["HAS_TRITON", "triton", "tl"]
