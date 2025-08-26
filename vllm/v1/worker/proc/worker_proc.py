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
from vllm.utils import (decorate_logs, get_distributed_init_method,
                        get_loopback_ip, get_mp_context, get_open_port,
                        set_process_title)
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)




@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Optional[Connection] = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue  # The worker process writes to this MQ
    death_writer: Optional[Connection] = None

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        pp_str = f"PP{rank // tp_size}" if pp_size > 1 else ""
        tp_str = f"TP{rank % tp_size}" if tp_size > 1 else ""
        suffix = f"{pp_str}{'_' if pp_str and tp_str else ''}{tp_str}"
        process_name = "VllmWorker"
        if suffix:
            set_process_title(suffix, append=True)
            process_name = f"{process_name} {suffix}"
        decorate_logs(process_name)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)

        # Initialize device and loads weights
        self.worker.init_device()
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle]
    ) -> list[WorkerProcHandle]:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * len(unready_proc_handles))
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    # Extract the message queue handle.
                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[unready_proc_handle.rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    os.kill(os.getpid(), signal.SIGTERM)
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(target=monitor_parent_death,
                                   daemon=True,
                                   name="WorkerDeathMonitor")
            death_monitor.start()

        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                WorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue

            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, output))
