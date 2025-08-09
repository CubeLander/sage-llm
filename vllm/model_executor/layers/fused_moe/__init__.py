# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any
from typing import Optional

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize)
from vllm.platforms.triton_tuils import HAS_TRITON

_config: Optional[dict[str, Any]] = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[dict[str, Any]]:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoEConfig",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
    "FusedMoEPermuteExpertsUnpermute",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalize",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts)
    from vllm.model_executor.layers.fused_moe.batched_triton_or_deep_gemm_moe import (  # noqa: E501
        BatchedTritonOrDeepGemmExperts)
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        CutlassExpertsFp8)
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        cutlass_moe_fp4)
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        cutlass_moe_fp8)
    from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
        DeepGemmExperts)
    from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
        BatchedTritonExperts)
    import vllm.model_executor.layers.fused_moe.fused_marlin_moe  # noqa
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_config_file_name)
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
    from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts)

    __all__ += [
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
        "cutlass_moe_fp8",
        "cutlass_moe_fp4",
        "CutlassExpertsFp8",
        "TritonExperts",
        "BatchedTritonExperts",
        "DeepGemmExperts",
        "BatchedDeepGemmExperts",
        "TritonOrDeepGemmExperts",
        "BatchedTritonOrDeepGemmExperts",
    ]
