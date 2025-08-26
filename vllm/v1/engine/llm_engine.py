# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from copy import copy
from typing import Any, Callable, Optional, Union

from typing_extensions import TypeVar

from vllm.v1.engine.core import EngineCore
from vllm.config import  VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.transformers_utils.tokenizer_group import TokenizerGroup, init_tokenizer_from_configs
from vllm.utils import Device
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.metrics.loggers import PrometheusStatLogger, StatLoggerBase, StatLoggerFactory
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:

    def __init__(
        self,
        vllm_config: VllmConfig,
        log_stats: bool,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.log_stats = log_stats
        self.stat_logger: Optional[StatLoggerBase] = None
        if self.log_stats:
            self.stat_logger = PrometheusStatLogger(vllm_config)

        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.

        if self.model_config.skip_tokenizer_init:
            self.tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config,
                scheduler_config=vllm_config.scheduler_config,
                lora_config=vllm_config.lora_config,
            )

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(vllm_config=vllm_config, tokenizer=self.tokenizer, mm_registry=mm_registry)

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer, log_stats=self.log_stats)

        self.engine_core = EngineCore(
            vllm_config=vllm_config,
            log_stats=self.log_stats,
        )

        # Don't keep the dummy data in memory
        self.reset_mm_cache()
        self.engine_core.execute_dummy_batch()

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        return has_unfinished or self.engine_core.dp_engines_running()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def abort_request(self, request_ids: list[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> None:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request, tokenization_kwargs, trace_headers, priority
        )

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_str, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            request_id, params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            # Make a new RequestState and queue.
            self.output_processor.add_request(child_request, prompt_str, parent_req, idx)
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

    def step(self) -> Union[list[RequestOutput], list[PoolingRequestOutput]]:

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        iteration_stats = IterationStats() if self.log_stats else None
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs, engine_core_timestamp=outputs.timestamp, iteration_stats=iteration_stats
        )

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        if self.stat_logger is not None:
            assert outputs.scheduler_stats is not None
            self.stat_logger.record(scheduler_stats=outputs.scheduler_stats, iteration_stats=iteration_stats)

        return processed_outputs.request_outputs

    def get_vllm_config(self):
        return self.vllm_config

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.processor.mm_registry.reset_processor_cache(self.model_config)
        self.processor.mm_input_cache_client.reset()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(self, device: Optional[Device] = None):
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    def get_tokenizer_group(self) -> TokenizerGroup:
        if self.tokenizer is None:
            raise ValueError("Unable to get tokenizer because " "skip_tokenizer_init is True")

        return self.tokenizer

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def __del__(self):
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)
