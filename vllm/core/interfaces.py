# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC
from abc import abstractmethod
import enum
from typing import List
from typing import Optional
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.core.types.sequence import VllmSequence
from vllm.sequence import SequenceGroup
from vllm.utils import Device


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager(ABC):

    @staticmethod
    def get_block_space_manager_class(version: str):
        version = version.lower()

        if version == "selfattn":
            from vllm.core.block_manager import SelfAttnBlockSpaceManager
            return SelfAttnBlockSpaceManager

        if version == "placeholder":
            from vllm.core.placeholder_block_space_manager import (
                PlaceholderBlockSpaceManager)
            return PlaceholderBlockSpaceManager

        raise ValueError(f"Unknown version {version=}")

    @abstractmethod
    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        pass

    @abstractmethod
    def allocate(self, seq_group: SequenceGroup) -> None:
        pass

    @abstractmethod
    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        pass

    @abstractmethod
    def append_slots(
        self,
        seq: VllmSequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def fork(self, parent_seq: VllmSequence, child_seq: VllmSequence) -> None:
        pass

    @abstractmethod
    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        pass

    @abstractmethod
    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        pass

    @abstractmethod
    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def free(self, seq: VllmSequence) -> None:
        pass

    @abstractmethod
    def get_block_table(self, seq: VllmSequence) -> List[int]:
        pass

    @abstractmethod
    def get_num_free_gpu_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_cpu_blocks(self) -> int:
        pass

    @abstractmethod
    def access_all_blocks_in_seq(
        self,
        seq: VllmSequence,
        access_time: float,
    ) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, seqs: List[VllmSequence]) -> GenericSequence[int]:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        pass

    @abstractmethod
    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        pass

    @abstractmethod
    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        """Reset prefix cache for specified or all devices."""
        pass

    @abstractmethod
    def get_num_cached_tokens(self, seq: VllmSequence) -> int:
        pass

    @abstractmethod
    def remove_seq_from_computed_blocks_tracker(self, seq: VllmSequence) -> None:
        pass