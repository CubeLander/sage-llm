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

from vllm.io.inputs import SingletonInputs
from vllm.io.inputs.multimodal import MultiModalKwargs
from vllm.io.inputs.multimodal import MultiModalPlaceholderDict
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceDataDelta, SequenceStage, array_full

if TYPE_CHECKING:
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorOutput)
    
VLLM_TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1




class SequenceData(msgspec.Struct,
                   omit_defaults=True):  # type: ignore[call-arg]
    """Data associated with a sequence.

    Args:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output. Set to an empty list if
            None.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """
    # NOTE: we cannot use Union[list, array] because msgspec cannot support
    # union of 2 list types.
    _prompt_token_ids: array
    _output_token_ids: array = msgspec.field(
        default_factory=lambda: array(VLLM_TOKEN_ID_ARRAY_TYPE, []))

    _prompt_embeds: Optional[torch.Tensor] = None
    _output_embeds: Optional[torch.Tensor] = None

    ### The below fields should not be passed as an argument ###
    _cumulative_logprob: float = 0.0
    _prompt_token_ids_tuple: tuple[int,
                                   ...] = msgspec.field(default_factory=tuple)
    # The number of tokens that are computed (that run against the model).
    _num_computed_tokens: int = 0
    # The number of tokens with prefix cache hit.
    _num_cached_tokens: int = 0
    _stage: SequenceStage = SequenceStage.PREFILL
    _cached_all_token_ids: list[int] = msgspec.field(default_factory=list)
    _cached_all_token_embeds: Optional[torch.Tensor] = None

    # It is used to get delta input. It is reset when `get_delta_and_reset`
    # is called.
    _new_appended_tokens: list[int] = msgspec.field(default_factory=list)

    # It is used to compute mrope_position_ids.
    _mrope_position_delta: Optional[int] = None

    @staticmethod
    def from_prompt_token_counts(
            *token_counts: tuple[int, int]) -> "SequenceData":
        """
        Construct a [`SequenceData`][vllm.sequence.SequenceData] instance
        by concatenating prompt token sequences.

        Each tuple represents one token sequence, expressed in the form
        `(token_id, count)`.
        """
        if len(token_counts) == 0:
            return SequenceData.from_seqs([])

        prompt_token_ids_arr = reduce(
            array.__iadd__,
            (array_full(token_id, count) for token_id, count in token_counts),
        )

        return SequenceData(prompt_token_ids_arr)

    @staticmethod
    def from_seqs(
        prompt_token_ids: GenericSequence[int],
        output_token_ids: Optional[GenericSequence[int]] = None,
        *,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> "SequenceData":
        """
        Construct a [`SequenceData`][vllm.sequence.SequenceData] instance
        from prompt and output token sequences.
        """
        prompt_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                     prompt_token_ids)

        if output_token_ids is None:
            return SequenceData(prompt_token_ids_arr,
                                _prompt_embeds=prompt_embeds)

        output_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                     output_token_ids)

        return SequenceData(prompt_token_ids_arr,
                            _output_token_ids=output_token_ids_arr,
                            _prompt_embeds=prompt_embeds)

    def __post_init__(self) -> None:
        assert self._prompt_token_ids.typecode == "l"
        assert self._output_token_ids.typecode == "l"
        self._prompt_token_ids_tuple: tuple[int, ...] = tuple(
            self._prompt_token_ids)
        self._update_cached_all_tokens()
        if self._prompt_embeds is not None:
            self._update_cached_all_token_embeds()

    def _update_cached_all_tokens(self):
        assert isinstance(self._prompt_token_ids, array)
        assert isinstance(self._output_token_ids, array)
        self._cached_all_token_ids: list[int] = list(self._prompt_token_ids +
                                                     self._output_token_ids)

    def _update_cached_all_token_embeds(self):
        assert isinstance(self._prompt_embeds, torch.Tensor)
        self._cached_all_token_embeds: torch.Tensor = self._prompt_embeds
        if self._output_embeds is not None:
            self._cached_all_token_embeds = torch.cat(
                (self._cached_all_token_embeds, self._output_embeds), dim=0)

    @property
    def cumulative_logprob(self) -> float:
        return self._cumulative_logprob

    @property
    def prompt_token_ids(self) -> tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        raise NotImplementedError

    @property
    def prompt_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        return self._prompt_token_ids

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self,
                         new_output_token_ids: GenericSequence[int]) -> None:
        self._output_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                       new_output_token_ids)
        self._update_cached_all_tokens()

    @property
    def output_embeds(self) -> Optional[torch.Tensor]:
        return self._output_embeds

    @output_embeds.setter
    def output_embeds(self, new_output_token_embeds: torch.Tensor) -> None:
        self._output_token_embeds = new_output_token_embeds
        self._update_cached_all_token_embeds()

    @property
    def output_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        assert isinstance(self._output_token_ids, array)
        return self._output_token_ids

    @property
    def prompt_embeds(self) -> Optional[torch.Tensor]:
        return self._prompt_embeds

    @prompt_embeds.setter
    def prompt_embeds(self, prompt_embeds: torch.Tensor) -> None:
        self._prompt_embeds = prompt_embeds
        self._update_cached_all_token_embeds()

    @property
    def mrope_position_delta(self) -> Optional[int]:
        return self._mrope_position_delta

    @mrope_position_delta.setter
    def mrope_position_delta(self, new_mrope_position_delta):
        self._mrope_position_delta = new_mrope_position_delta

    def append_token_id(self,
                        token_id: int,
                        logprob: float,
                        token_embed: Optional[torch.Tensor] = None) -> None:
        self._output_token_ids.append(token_id)
        self._new_appended_tokens.append(token_id)
        self._cached_all_token_ids.append(token_id)
        self._cumulative_logprob += logprob
        if token_embed is not None:
            # Do not pass in with batch or sequence dimensions
            assert token_embed.ndim == 1
            token_embed = token_embed.detach().cpu().unsqueeze(0)
            if self._output_embeds is None:
                self._output_embeds = token_embed
            else:
                self._output_embeds = torch.cat(
                    (self._output_embeds, token_embed), dim=0)
            assert self._cached_all_token_embeds is not None
            self._cached_all_token_embeds = torch.cat(
                (self._cached_all_token_embeds,
                 token_embed.to(device=self._cached_all_token_embeds.device)),
                dim=0)

    def get_len(self) -> int:
        return len(self._output_token_ids) + len(self._prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self._prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self._output_token_ids)

    def get_token_ids(self) -> list[int]:
        return self._cached_all_token_ids

    def get_token_embeddings(self) -> Optional[torch.Tensor]:
        return self._cached_all_token_embeds

    def get_prefix_token_ids(
            self, num_tokens: int
    ) -> tuple[tuple[int, ...], Optional[tuple[int, ...]]]:
        """Get prefix tokens, and make the return value hashable"""
        prompt_length = self.get_prompt_len()
        if num_tokens > prompt_length:
            return (self._prompt_token_ids_tuple,
                    tuple(self._output_token_ids[:num_tokens - prompt_length]))
        else:
            return (self._prompt_token_ids_tuple[:num_tokens], None)

    def get_num_computed_tokens(self) -> int:
        """Return the number of prefill tokens that are already computed."""
        return self._num_computed_tokens

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len(), (
            self._num_computed_tokens, self.get_len())
        # If all tokens are computed, it means it is in decoding phase.
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = SequenceStage.DECODE

    def get_num_cached_tokens(self) -> int:
        """Return the number of tokens with prefix cache hit."""
        return self._num_cached_tokens

    def update_num_cached_tokens(self, num_cached_tokens: int):
        """Update the number of tokens with prefix cache hit."""
        self._num_cached_tokens = num_cached_tokens

    def reset_state_for_recompute(self) -> None:
        """Reset the number of computed tokens from this sequence. It is
        supposed to be called when a sequence needs to be started from
        the beginning again (e.g., sequence is preempted).
        """
        self._num_computed_tokens = 0
        self._stage = SequenceStage.PREFILL
        self._new_appended_tokens = []

    def get_num_uncomputed_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self._output_token_ids:
            return self._prompt_token_ids[-1]
        return self._output_token_ids[-1]

    def get_prompt_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> tuple[int, ...]:
        return self.output_token_ids

    def get_delta_and_reset(self) -> SequenceDataDelta:
        delta = SequenceDataDelta(self._new_appended_tokens,
                                  self._cumulative_logprob,
                                  self.get_num_computed_tokens(), self.stage)
        # Reset delta state.
        self._new_appended_tokens = []
        return delta

    def apply_delta(self, delta: SequenceDataDelta):
        self._num_computed_tokens = delta.new_num_computed_tokens
        self._cumulative_logprob = delta.new_cumulative_logprob
        self._stage = delta.new_stage
        self._output_token_ids.extend(delta.new_output_token_ids)
        self._cached_all_token_ids.extend(delta.new_output_token_ids)

    @property
    def stage(self) -> SequenceStage:
        return self._stage

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self._prompt_token_ids}, "
                f"prompt_embeds.shape="
                f"{getattr(self._prompt_embeds, 'shape', None)}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"get_num_computed_tokens={self.get_num_computed_tokens()})")

