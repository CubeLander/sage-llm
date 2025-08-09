# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence and its related classes."""
from abc import ABC
from abc import abstractmethod
from array import array
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence as GenericSequence
import copy
from dataclasses import dataclass
from dataclasses import field
import enum
from functools import reduce
from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import msgspec
import torch

from vllm.core.types.logprob import Logprob
from vllm.core.types.sequence import VllmSequence
from vllm.core.types.sequence_data import SequenceData
from vllm.core.types.sequence_status import SequenceStatus
from vllm.io.inputs import SingletonInputs
from vllm.io.inputs.multimodal import MultiModalKwargs
from vllm.io.inputs.multimodal import MultiModalPlaceholderDict
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorOutput)

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1


def array_full(token_id: int, count: int):
    """[`array`][] equivalent of [numpy.full][]."""
    return array(VLLM_TOKEN_ID_ARRAY_TYPE, [token_id]) * count


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = list[Optional[dict[int, Logprob]]]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = list[dict[int, Logprob]]





class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
    scheduler_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None


class SequenceDataDelta(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """Delta SequenceData to send to workers per step."""
    # A new token to be appended to existing SequenceData.
    new_output_token_ids: list[int]
    # Overwriting existing `cumulative_logprob`
    new_cumulative_logprob: float
    # Overwriting existing `num_computed_tokens`.
    new_num_computed_tokens: int
    # Overwriting existing `stage`.
    new_stage: SequenceStage


class SequenceGroupState(msgspec.Struct,
                         omit_defaults=True):  # type: ignore[call-arg]
    """Mutable state tied to a specific sequence group"""

    # for multi-step decoding
    num_steps: int = 1
    current_step: int = 0

    @property
    def remaining_steps(self) -> int:
        return self.num_steps - self.current_step


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
        lora_request: LoRA request.
        pooling_params: The parameters used to generate the pooler
            for a pooling model.
        pooled_data: The extracted hidden states from a pooling model.
        encoder_seq: Optional, the single encoder sequence. Should be None
                     unless you are working with an encoder/decoder model.
        trace_headers: OpenTelemetry trace headers.
        priority: User-defined priority of the request.
        draft_size: The number of speculative tokens plus one from the target
                    model; equal to max number of tokens a step can generate
                    for single-draft speculative decoding but larger than
                    that for multi-draft SD (currently not supported).
    """

    def __init__(self,
                 request_id: str,
                 seqs: list[VllmSequence],
                 arrival_time: float,
                 sampling_params: Optional[SamplingParams] = None,
                 lora_request: Optional[LoRARequest] = None,
                 pooling_params: Optional[PoolingParams] = None,
                 pooled_data: Optional[torch.Tensor] = None,
                 encoder_seq: Optional[VllmSequence] = None,
                 trace_headers: Optional[Mapping[str, str]] = None,
                 priority: int = 0,
                 draft_size: int = 1) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.first_seq = seqs[0]
        self.arrival_time = arrival_time
        self.is_single_seq = len(seqs) == 1
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}

        self.sampling_params = sampling_params
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.last_token_latency = 0.0
        self.lora_request = lora_request
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.state = SequenceGroupState()
        self.pooling_params = pooling_params
        self.pooled_data = pooled_data
        self.encoder_seq = encoder_seq
        self.trace_headers = trace_headers
        self.priority = priority

        self.cached_request_output = None

    @property
    def prompt(self) -> Optional[str]:
        return self.first_seq.prompt

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.first_seq.prompt_token_ids

    @property
    def encoder_prompt(self) -> Optional[str]:
        # There are either 0 or 1 encoder sequences
        # If one is present, its prompt is distinct
        # from the decoder's.
        return (self.encoder_seq.prompt
                if self.encoder_seq is not None else None)

    @property
    def encoder_prompt_token_ids(self) -> Optional[list[int]]:
        # There are either 0 or 1 encoder sequences
        # If one is present, its prompt token ids are
        # distinct from the decoder's.
        return (self.encoder_seq.prompt_token_ids
                if self.encoder_seq is not None else None)

    @property
    def token_type_ids(self) -> Optional[list[int]]:
        return self.first_seq.token_type_ids

    @property
    def multi_modal_data(self) -> MultiModalKwargs:
        if self.first_seq.multi_modal_data:
            return self.first_seq.multi_modal_data
        elif self.encoder_seq is not None:
            return self.encoder_seq.multi_modal_data
        return MultiModalKwargs({})

    @property
    def multi_modal_placeholders(self) -> MultiModalPlaceholderDict:
        if self.first_seq.multi_modal_data:
            return self.first_seq.multi_modal_placeholders
        elif self.encoder_seq is not None:
            return self.encoder_seq.multi_modal_placeholders
        return {}

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def init_multi_step(self, num_steps: int) -> None:
        self.state.num_steps = num_steps
        self.state.current_step = 0

    def init_multi_step_from_lookahead_slots(self, num_lookahead_slots: int,
                                             num_scheduler_steps: int,
                                             is_multi_step: bool,
                                             enable_chunking: bool) -> None:

        if not is_multi_step:
            self.init_multi_step(num_steps=num_scheduler_steps)
            return

        # Multi-Step case
        is_prefill = self.is_prefill()

        # The asserts below reflect the expectations of the current system.
        if is_prefill and enable_chunking:
            assert num_lookahead_slots == num_scheduler_steps
            self.init_multi_step(num_steps=num_lookahead_slots)
        else:
            is_decode: bool = not is_prefill
            # If it is a prefill, num_lookahead_slots must be 0
            assert num_lookahead_slots == 0 or is_decode
            # If it is a decode, num_lookahead_slots + 1 must match
            # the scheduler steps.
            assert num_lookahead_slots + 1 == num_scheduler_steps or is_prefill
            self.init_multi_step(num_steps=num_lookahead_slots + 1)

    def set_last_token_time(self, now: float) -> None:
        """Sets the last token time for Request level timings."""
        # If still in prefill phase, assertion fails.
        assert not self.is_prefill(), (
            "seq_group.set_last_token_time() should not be called "
            "if the seq_group is in prefill phase.")
        self.last_token_latency = now - self.metrics.last_token_time
        self.metrics.last_token_time = now

    def get_last_token_latency(self) -> float:
        """Returns the latency of the last token."""
        assert not self.is_prefill(), (
            "seq_group.get_last_token_latency() should not be called "
            "if the seq_group is in prefill phase.")
        return self.last_token_latency

    def maybe_set_first_token_time(self, time: float) -> None:
        """Sets the first token time for Request level timings."""
        # Note: in a case where a sequence_group is swapped and
        #   recomputed, the time between iterations is counted
        #   in TPOT, rather than recalculating TTFT (since from the )
        #   POV of the user, there is simply a long generation delay.
        if (self.metrics.first_token_time is None
                and self.first_seq.get_output_len() == 1):
            self.metrics.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        """Sets the first scheduled time and time in queue for Request
        level timings."""
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = time - self.metrics.arrival_time

    def set_finished_time(self, time: Optional[float]) -> None:
        """Sets the finished time for Request level timings."""
        self.metrics.finished_time = time

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.is_single_seq:
            return 0 if self.first_seq.is_finished() else 1
        return self.num_seqs() - self.num_finished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> list[VllmSequence]:
        if status is None:
            return self.seqs

        if self.is_single_seq:
            return self.seqs if self.first_seq.status == status else []

        return [seq for seq in self.seqs if seq.status == status]

    def is_encoder_decoder(self) -> bool:
        return self.encoder_seq is not None

    def get_encoder_seq(self) -> Optional[VllmSequence]:
        return self.encoder_seq

    def get_finished_seqs(self) -> list[VllmSequence]:
        if self.is_single_seq:
            return self.seqs if self.first_seq.is_finished() else []

        return [seq for seq in self.seqs if seq.is_finished()]

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        for seq in self.seqs:
            if not seq.is_finished():
                seq.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        num_uncomputed_tokens = 0
        for seq in self.seqs:
            if not seq.is_finished():
                num_uncomputed_tokens += seq.data.get_num_uncomputed_tokens()
        return num_uncomputed_tokens

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        # Optimization. We don't need to call get_seqs if we don't need to
        # filter by states.
        if status is None:
            return len(self.seqs)

        if self.is_single_seq:
            return 1 if self.seqs[0].status == status else 0

        return len(self.get_seqs(status))

    def num_finished_seqs(self) -> int:
        if self.is_single_seq:
            return 1 if self.seqs[0].is_finished() else 0
        return len(self.get_finished_seqs())

    def is_finished(self) -> bool:
        if self.is_single_seq:
            return self.first_seq.is_finished()
        return all(seq.is_finished() for seq in self.seqs)

    def is_prefill(self) -> bool:
        return self.first_seq.is_prefill()

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs)})")

    def uses_prompt_embeds(self) -> bool:
        """Returns True if the sequence group uses input embeds."""
        return any(seq.data.prompt_embeds is not None for seq in self.seqs)


class SequenceGroupMetadataDelta(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """Delta of SequenceGroupMetadata.

    After sending the first SequenceGroupMetadata, vLLM scheduler
    only sends delta to reduce the data payload size.
    """
    seq_data_delta: dict[int, SequenceDataDelta]
    request_id: str
    block_tables: dict[int, list[int]]
    is_prompt: bool
    do_sample: bool = True
    token_chunk_size: Optional[int] = None
    computed_block_nums: Optional[list[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(
        default_factory=lambda: SequenceGroupState())


class SequenceGroupMetadata(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """Metadata for a sequence group. Used to create `AttentionMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
        do_sample: True if sampling is required. Sampling is not required when
            e.g., prefill is chunked, and the current iteration only computes
            query tokens for prefill, we don't need sampling.
        token_chunk_size: The number of tokens to be processed (per sequence).
            None if chunking is not required.
        lora_request: LoRA request.
        computed_block_nums: The block numbers that are already computed,
            used in prefix caching.
        state: Internal state tied to this sequence group.
        multi_modal_data: Multi modal data.
        mm_processor_kwargs: Multimodal input processor / mapper overrides.
        encoder_seq_data: Optional sequence data for encoder prompt
                          (SequenceGroup.encoder_seq). Should be None
                          unless you are working with an encoder/decoder
                          model.
        cross_block_table: Optional cross-attention block table associated
                           with the encoder prompt
                           (SequenceGroup.encoder_seq). Should be None
                           unless you are working with an encoder/decoder
                           model.
    """

    request_id: str
    is_prompt: bool
    seq_data: dict[int, SequenceData]
    sampling_params: Optional[SamplingParams]
    block_tables: dict[int, list[int]]
    do_sample: bool = True
    pooling_params: Optional[PoolingParams] = None
    lora_request: Optional[LoRARequest] = None
    computed_block_nums: Optional[list[int]] = None
    state: Optional[SequenceGroupState] = msgspec.field(
        default_factory=lambda: SequenceGroupState())
    token_type_ids: Optional[list[int]] = None
    multi_modal_data: Optional[MultiModalKwargs] = None
    multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
    encoder_seq_data: Optional[SequenceData] = None
    cross_block_table: Optional[list[int]] = None
    token_chunk_size: Optional[int] = None

    ### Stateful fields that are lazily defined. ###
    # The number of speculative tokens adopted in this request.
    # None means specuative decoding is not used.
    # Zero means speculative decoding is disabled for some reasons.
    # TODO: We should maintain this states out of the sequence group.
    num_speculative_tokens: Optional[int] = None

    def __post_init__(self):
        if self.seq_data is not None and self.token_chunk_size is None:
            if self.is_prompt:
                self.token_chunk_size = next(iter(
                    self.seq_data.values())).get_len()
            else:
                self.token_chunk_size = 1

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    # Multi-Step Chunked-Prefill property
    @property
    def is_single_step_prompt(self) -> bool:
        # do_sample is true, only when the token_chunk_size matches the
        # num_uncomputed_tokens of the sequence. This indicates that
        # the prompt will finish processing in a single `execute_model`
        # step.
        return self.is_prompt and self.do_sample

    def get_first_seq_id(self) -> int:
        # This is an efficient way of fetching the seq_id when
        # we know this SequenceGroup has only one sequence.
        return next(iter(self.seq_data))

    def apply_delta(self,
                    sequence_group_metadata_delta: SequenceGroupMetadataDelta):
        for id, delta in sequence_group_metadata_delta.seq_data_delta.items():
            self.seq_data[id].apply_delta(delta)
        assert self.request_id == sequence_group_metadata_delta.request_id
        self.block_tables = sequence_group_metadata_delta.block_tables
        self.token_chunk_size = sequence_group_metadata_delta.token_chunk_size
        self.do_sample = sequence_group_metadata_delta.do_sample
        self.is_prompt = sequence_group_metadata_delta.is_prompt

    def finish_step(self) -> None:
        assert self.state is not None
        assert self.state.current_step < self.state.num_steps, \
            f"current step {self.state.current_step}, num_steps {self.state.num_steps}" # noqa
        self.state.current_step += 1


class SequenceOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """
    parent_seq_id: int
    output_token: int
    logprobs: dict[int, Logprob]
    output_embed: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        output_embed_shape = \
            self.output_embed.shape if self.output_embed is not None else None
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"output_embed.shape={output_embed_shape}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        equal = (self.parent_seq_id == other.parent_seq_id
                 and self.output_token == other.output_token)
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal


class SequenceGroupOutput(ABC):
    """The base class for model outputs associated with a sequence group."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class CompletionSequenceGroupOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """The model output associated with a completion sequence group."""
    __metaclass__ = SequenceGroupOutput
    samples: list[SequenceOutput]
    # Prompt logprob for each prompt query token.
    prompt_logprobs: Optional[PromptLogprobs]
    step_index: Optional[int] = 0

    def __repr__(self) -> str:
        return (f"CompletionSequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionSequenceGroupOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


class PoolingSequenceGroupOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
):
    """The model output associated with a pooling sequence group."""
    __metaclass__ = SequenceGroupOutput
    # Annotated as Any to be compatible with msgspec
    # The actual type is in SequenceGroup.pooled_data
    data: Any

    def get_data_nbytes(self) -> int:
        data: torch.Tensor = self.data
        return data.nbytes

    def __repr__(self) -> str:
        return f"PoolingSequenceGroupOutput(data={self.data}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoolingSequenceGroupOutput):
            raise NotImplementedError()
        return self.data == other.data


class PoolerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """The output from a pooling operation in the pooling model."""
    outputs: list[PoolingSequenceGroupOutput]

    def get_data_nbytes(self) -> int:
        return sum(o.get_data_nbytes() for o in self.outputs)

    def __getitem__(self, idx: int) -> PoolingSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value: PoolingSequenceGroupOutput):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other,
                          self.__class__) and self.outputs == other.outputs


def get_all_seq_ids(
        seq_group_metadata_list: list[SequenceGroupMetadata]) -> list[int]:
    """Given a list of SequenceGroupMetadata, create a list of all
    sequence ids.
    """
    return [seq_id for sg in seq_group_metadata_list for seq_id in sg.seq_data]


def get_all_seq_ids_and_request_ids(
    seq_group_metadata_list: list[SequenceGroupMetadata]
) -> tuple[list[int], dict[str, set[int]]]:
    """Given a list of SequenceGroupMetadata, create a list of all
    sequence ids.
    """
    seq_ids: list[int] = []
    request_id_seq_ids_mapping: defaultdict[str, set[int]] = defaultdict(set)
    for sg in seq_group_metadata_list:
        for seq_id in sg.seq_data:
            seq_ids.append(seq_id)
            request_id_seq_ids_mapping[sg.request_id].add(seq_id)
    return seq_ids, request_id_seq_ids_mapping


class HiddenStates(msgspec.Struct, array_like=True,
                   omit_defaults=True):  # type: ignore[call-arg]
    """Hidden states corresponding to in-progress sequences.
    Used in speculative decoding to pass hidden states from
    the target model to the proposer model.

    seq_ids are the sequence ids of each entry of the batch
    dimension of the hidden_states tensor"""
    # Scorer hidden states. For prefill step, it is used for hidden states of
    # all tokens, whereas for decode step, it use used for last accepted tokens.
    hidden_states: torch.Tensor
    # The sequence group metadata list. Only needed for decode step.
    seq_group_metadata_list: Optional[list[SequenceGroupMetadata]] = None
    # Scorer hidden states of the 2nd last token proposed by the proposer (
    # irrespective of whether it was accepted or not). Only used for cases when
    # last proposed token is accepted (i.e., in case of bonus tokens). For the
    # case of no bonus tokens, these are ignored.
    second_last_token_hidden_states: Optional[torch.Tensor] = None

    _seq_ids: list[int] = msgspec.field(default_factory=list)

    def __post_init__(self):
        if self.seq_group_metadata_list is not None:
            assert len(self.seq_group_metadata_list) == len(self.hidden_states)
            self._seq_ids = get_all_seq_ids(self.seq_group_metadata_list)

    @property
    def seq_ids(self) -> list[int]:
        return self._seq_ids

    def update(self,
               hidden_states: torch.Tensor,
               seq_group_metadata_list: list[SequenceGroupMetadata],
               second_last_token_hidden_states: Optional[torch.Tensor] = None):
        """Update hidden states from target model invocation. Only used for
        decode steps"""
        assert len(seq_group_metadata_list) == len(hidden_states)
        self._seq_ids.extend(get_all_seq_ids(seq_group_metadata_list))
        self.hidden_states = torch.cat([self.hidden_states, hidden_states])

        if self.second_last_token_hidden_states is not None:
            # Adding dummy hidden_states to this to maintain same shape
            self.second_last_token_hidden_states = torch.cat([
                self.second_last_token_hidden_states,
                torch.zeros_like(hidden_states)
                if second_last_token_hidden_states is None else
                second_last_token_hidden_states
            ])

    def prune(self,
              seq_group_metadata_list: list[SequenceGroupMetadata]) -> None:
        """Prune to provided list of sequence ids. Only used for decode steps.
        """
        # Currently this prunes all seq_ids not present in
        # seq_group_metadata_list which might cause problems where a sequence
        # may be "paused" then "resumed" later. This should only prune sequences
        # which are confirmed to be aborted.
        seq_ids = get_all_seq_ids(seq_group_metadata_list)
        # Only keep sequence IDs that exist in self._seq_ids
        seq_ids = [seq_id for seq_id in seq_ids if seq_id in self._seq_ids]
        if seq_ids != self._seq_ids:
            # Batch contents changed - prune removed sequences.
            index = [self._seq_ids.index(seq_id) for seq_id in seq_ids]
            self.hidden_states = self.hidden_states[index]
            if self.second_last_token_hidden_states is not None:
                self.second_last_token_hidden_states = self\
                    .second_last_token_hidden_states[index]
            self._seq_ids = seq_ids

    def expand_with_bonus_tokens(
            self, seq_with_bonus_token_in_last_step: set) -> None:
        """Expand hidden states for sequences with bonus tokens. This is in
        alignment with `MultiStepWorker._expand_execute_model_request`."""
        if self.second_last_token_hidden_states is None \
            or not seq_with_bonus_token_in_last_step:
            return

        index = []
        for seq_id in self._seq_ids:
            i = self._seq_ids.index(seq_id)
            if seq_id in seq_with_bonus_token_in_last_step:
                index.append(i + len(self._seq_ids))
            index.append(i)

        self.hidden_states = torch.cat(
            [self.hidden_states, self.second_last_token_hidden_states])[index]


class ExecuteModelRequest(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    """The model execution request, containing CPU metadata only. The LLM
    engine should create an instance of this class for each request batch."""
    # The sequence group metadata list.
    seq_group_metadata_list: list[Union[SequenceGroupMetadata,
                                        SequenceGroupMetadataDelta]]
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: list[tuple[int,
                                  int]] = msgspec.field(default_factory=list)
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: list[tuple[int,
                                   int]] = msgspec.field(default_factory=list)
    # Blocks to copy. Source to dest block.
    blocks_to_copy: list[tuple[int, int]] = msgspec.field(default_factory=list)
    # Virtual engine ID for pipeline parallel.
    virtual_engine: int = 0
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int = 0
    # The number of requests in the running queue.
    running_queue_size: int = 0
    # Optional hidden states from prior step.
    previous_hidden_states: Optional[HiddenStates] = None
    # The number of forward steps to run.
    num_steps: int = 1
    # Finished request ids since last step.
    finished_requests_ids: list[str] = msgspec.field(default_factory=list)
    # The last sampled token ids for multi step decoding.
    last_sampled_token_ids: Optional[torch.Tensor] = None
    # Async callback
    async_callback: Optional[Callable] = None

    @property
    def is_first_multi_step(self) -> bool:
        # TODO(will) make this be able to handle batches with variable number of
        # steps
        assert len(self.seq_group_metadata_list) > 0
        first_seq_group = self.seq_group_metadata_list[0]
        assert first_seq_group.state is not None
        return first_seq_group.state.current_step == 0

    @property
    def is_last_step(self) -> bool:
        # TODO(will) make this be able to handle batches with variable number of
        # steps
        assert len(self.seq_group_metadata_list) > 0
        first_seq_group = self.seq_group_metadata_list[0]
        assert first_seq_group.state is not None
        return first_seq_group.state.remaining_steps == 1

    @property
    def current_step(self) -> int:
        # TODO(will) make this be able to handle batches with variable number of
        # steps
        assert len(self.seq_group_metadata_list) > 0
        state = self.seq_group_metadata_list[0].state
        assert state is not None
        return state.current_step

    def clone(
        self, seq_group_metadata_list: list[Union[SequenceGroupMetadata,
                                                  SequenceGroupMetadataDelta]]
    ) -> "ExecuteModelRequest":
        """Clone the request with a new sequence group metadata list."""
        return ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy(),
            virtual_engine=self.virtual_engine,
            num_lookahead_slots=self.num_lookahead_slots,
            running_queue_size=self.running_queue_size,
            previous_hidden_states=self.previous_hidden_states,
            num_steps=self.num_steps,
            finished_requests_ids=self.finished_requests_ids,
            last_sampled_token_ids=self.last_sampled_token_ids.clone()
            if self.last_sampled_token_ids is not None else None,
            async_callback=self.async_callback)


@dataclass
class SequenceGroupBase:
    group_id: str  # the original request id before splitting

    assembled_seq_group: Optional[SequenceGroup] = None

    # seq id to a unique index inside this group
    seq_id_to_index: dict[str, int] = field(default_factory=dict)

    # seq ids to be finished
    to_be_finished: dict[str, SequenceGroup] = field(default_factory=dict)

    # seq id to finished sequences
    finished_reqs: dict[str, SequenceGroup] = field(default_factory=dict)

    streaming: bool = False

    output_produced: bool = False

    @staticmethod
    def add_request(request_id: str, engine, params, *args, **kwargs):
        """When we are ready to add a request with request_id and params
        into the engine, we can split the request into multiple requests.
        """
        raise NotImplementedError

    def finish_seq(self, seq: SequenceGroup):
        """The sequence `seq` finishes, we should record the information.
        """
        del self.to_be_finished[seq.request_id]
        self.finished_reqs[seq.request_id] = seq

    def maybe_assemble_group(
            self, seq_group: SequenceGroup) -> Optional[SequenceGroup]:
        """Assemble the sequence group, for producing the final
        output, or adding request in the engine again.
        """
        raise NotImplementedError
