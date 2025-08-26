# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import signal
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any, Callable, Optional, Union, cast

import cloudpickle

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.proc.worker_proc import UnreadyWorkerProcHandle, WorkerProc, WorkerProcHandle
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method,
                        get_loopback_ip, get_open_port,)
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class MultiprocExecutor:

    supports_pp: bool = True

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        # HOTLLM_OPTIMIZATION: This connector check can be simplified to self.has_connector = False
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)

    def start_worker_monitor(self):
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers
                             if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, "
                "shutting down executor.", proc_name)
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        Thread(target=monitor_workers,
               daemon=True,
               name="MultiprocWorkerMonitor").start()

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config",
                            args=(kv_cache_configs, ))
        self.collective_rpc("compile_or_warm_up_model")


    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue(
                (send_method, args, kwargs, unique_reply_rank))

            workers = (self.workers[unique_reply_rank],
                       ) if unique_reply_rank is not None else self.workers
            responses = []

            def get_response(w: WorkerProcHandle,
                             dequeue_timeout: Optional[float] = None,
                             cancel_event: Optional[threading.Event] = None):
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause")
                return result

            for w in workers:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())

                if non_block:
                    result = self.io_thread_pool.submit(  # type: ignore
                        get_response, w, dequeue_timeout, self.shutdown_event)
                else:
                    result = get_response(w, dequeue_timeout)

                responses.append(result)

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()

            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None

            if workers := getattr(self, 'workers', None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return self.world_size - self.parallel_config.tensor_parallel_size

    def determine_available_memory(self) -> list[int]:  # in bytes
        output = self.collective_rpc("determine_available_memory")
        return output

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        output = self.collective_rpc("get_kv_cache_spec")
        return output


    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))