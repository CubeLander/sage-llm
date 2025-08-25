# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import (DeviceConfig, KVTransferConfig, ModelConfig,
                         VllmConfig, set_current_vllm_config)
from vllm.logger import init_logger

logger = init_logger("test_expert_parallel")







