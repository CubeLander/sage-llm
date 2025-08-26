# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import time
from concurrent.futures import Future
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.utils import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
    unify_kv_cache_configs,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.mm_input_cache import MultiModalInputCacheServer
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(self, vllm_config: VllmConfig, log_stats: bool, executor_fail_callback: Optional[Callable] = None):

        # plugins need to be loaded at the engine/scheduler level too

        self.vllm_config = vllm_config
        logger.info("Initializing a V1 LLM engine (v%s) with config: %s", VLLM_VERSION, vllm_config)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = MultiprocExecutor(vllm_config)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(vllm_config)

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.chunked_prefill_enabled = False

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
        )

        self.mm_input_cache_server = MultiModalInputCacheServer(vllm_config.model_config, MULTIMODAL_REGISTRY)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput], SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)

    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(kv_cache_specs)
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker, available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs])
        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, " "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request.request_id)}")

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [task for task in self.get_supported_tasks() if task in POOLING_TASKS]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} " f"Supported tasks: {supported_pooling_tasks}"
                )

        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    def execute_model_with_error_logging(
        self,
        model_fn: Callable[[SchedulerOutput], ModelRunnerOutput],
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Execute the model and log detailed info on failure."""
        try:
            return model_fn(scheduler_output)
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(self.vllm_config, scheduler_output, self.scheduler.make_stats())
            raise err

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        model_output = self.execute_model_with_error_logging(
            self.model_executor.execute_model, scheduler_output  # type: ignore
        )
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)  # type: ignore

        return (engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0)

    def step_with_batch_queue(self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        assert self.batch_queue is not None

        engine_core_outputs = None
        scheduler_output = None
        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        if not self.batch_queue.full():
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens > 0:
                future = self.model_executor.execute_model(scheduler_output)
                self.batch_queue.put_nowait((future, scheduler_output))  # type: ignore

        scheduled_batch = scheduler_output is not None and scheduler_output.total_num_scheduled_tokens > 0

        # If no more requests can be scheduled and the job queue is not empty,
        # block until the first batch in the job queue is finished.
        # TODO(comaniac): Ideally we should peek the first batch in the
        # job queue to check if it's finished before scheduling a new batch,
        # but peeking the first element in a queue is not thread-safe,
        # so we need more work.
        if not scheduled_batch and not self.batch_queue.empty():
            future, scheduler_output = self.batch_queue.get_nowait()

            # Blocking until the first result is available.
            model_output = self.execute_model_with_error_logging(lambda _: future.result(), scheduler_output)

            self.batch_queue.task_done()
            engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        return engine_core_outputs, scheduled_batch

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 processor, P0 mirror, P1 mirror)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are " "in progress may lead to desynced internal caches."
            )

        self.mm_input_cache_server.reset()

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_executor.save_sharded_state(path=path, pattern=pattern, max_size=max_size)

    def collective_rpc(
        self,
        method: Union[str, Callable[..., _R]],
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def save_tensorized_model(
        self,
        tensorizer_config,
    ) -> None:
        self.model_executor.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        if request.mm_hashes is not None:
            assert request.mm_kwargs is not None

            # Note on thread safety: no race condition.
            # `mm_input_cache_server` is reset at the end of LLMEngine init,
            # and will only accessed in the input processing thread afterwards.
            request.mm_kwargs = self.mm_input_cache_server.get_and_update(request.mm_kwargs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave
