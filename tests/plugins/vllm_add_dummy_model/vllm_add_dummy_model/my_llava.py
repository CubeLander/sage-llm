# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.io.inputs.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.llava import LlavaDummyInputsBuilder
from vllm.model_executor.models.llava import LlavaForConditionalGeneration
from vllm.model_executor.models.llava import LlavaMultiModalProcessor
from vllm.model_executor.models.llava import LlavaProcessingInfo
from vllm.model_executor.sampling_metadata import SamplingMetadata


@MULTIMODAL_REGISTRY.register_processor(LlavaMultiModalProcessor,
                                        info=LlavaProcessingInfo,
                                        dummy_inputs=LlavaDummyInputsBuilder)
class MyLlava(LlavaForConditionalGeneration):

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states, sampling_metadata)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits
