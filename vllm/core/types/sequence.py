from typing import Optional, Union
from vllm.core.types.logprob import Logprob
from vllm.core.types.sequence_data import SequenceData
from vllm.io.inputs import SingletonInputs
from vllm.io.inputs.multimodal import MultiModalKwargs, MultiModalPlaceholderDict
from vllm.lora.request import LoRARequest
from vllm.sequence import SampleLogprobs, SequenceStatus, SequenceStage


class VllmSequence:
    """Stores the data, status, and block information of a sequence.

    The sequence is constructed from the
    [`DecoderOnlyInputs`][vllm.io.inputs.data.DecoderOnlyInputs] (for decoder-only)
    or [`EncoderDecoderInputs`][vllm.io.inputs.data.EncoderDecoderInputs]
    (for encoder-decoder) instance passed in through the `inputs`
    constructor argument.

    Args:
        seq_id: The ID of the sequence.
        inputs: The inputs of the sequence.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
        eos_token_id: The end-of-sequence (EOS) token id recognized by this LLM.
        lora_request: LoRA request.
    """

    def __init__(
        self,
        seq_id: int,
        inputs: SingletonInputs,
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = inputs
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request

        self.data = SequenceData.from_seqs(
            self.prompt_token_ids,
            prompt_embeds=self.inputs["prompt_embeds"]
            if self.inputs["type"] == "embeds" else None)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None

        # These are used to keep track of delta outputs
        self._last_output_token_ids_offset: int = 0
        self._last_output_text_offset: int = 0

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[list[str]] = None

    @property
    def n_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size

    @property
    def prompt(self) -> Optional[str]:
        if self.inputs["type"] == "embeds":
            return None
        return self.inputs.get("prompt")

    @property
    def prompt_token_ids(self) -> list[int]:
        if self.inputs["type"] == "embeds":
            return [0] * len(self.inputs["prompt_embeds"])
        return self.inputs["prompt_token_ids"]

    @property
    def token_type_ids(self) -> list[int]:
        if self.inputs["type"] == "embeds":
            return []
        return self.inputs.get("token_type_ids", [])

    @property
    def multi_modal_data(self) -> MultiModalKwargs:
        if self.inputs["type"] == "multimodal":
            return self.inputs["mm_kwargs"]

        return MultiModalKwargs({})

    @property
    def multi_modal_placeholders(self) -> MultiModalPlaceholderDict:
        if self.inputs["type"] == "multimodal":
            return self.inputs["mm_placeholders"]

        return {}

    @property
    def lora_int_id(self) -> int:
        return self.lora_request.lora_int_id if self.lora_request else 0

    def get_output_text_to_return(self, buffer_length: int,
                                  delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        truncate = buffer_length and not self.is_finished()
        if not delta:
            return self.output_text[:-buffer_length] if truncate else (
                self.output_text)
        length = len(self.output_text)
        if truncate:
            length -= buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""

    def get_output_token_ids_to_return(
            self, delta: bool) -> Union[GenericSequence[int], int]:
        """If delta is True, only new tokens since the last call to
        this method are returned"""
        if not delta:
            return self.get_output_token_ids()

        output_len = self.get_output_len()

        # Get the number of new tokens
        num_new_tokens = output_len - self._last_output_token_ids_offset
        self._last_output_token_ids_offset = output_len

        # Return new tokens
        if num_new_tokens == 1:
            # Optimization for single decode token case
            # (which is what we have most of the time)
            return self.data._cached_all_token_ids[-1]

        if num_new_tokens == 0:
            return []

        return self.data._cached_all_token_ids[-num_new_tokens:]

    def hash_of_block(self, logical_idx: int) -> int:
        # TODO This can produce incorrect hash when block size > prompt size

        # Compute the number of tokens in the sequence
        # TODO: The current hashing function is O(L^2). We should optimize
        # this in the future.
        num_tokens = self.num_hashed_tokens_of_block(logical_idx)
        hashed_tokens = self.data.get_prefix_token_ids(num_tokens)
        return hash((hashed_tokens, self.lora_int_id))

    def extra_hash(self) -> Optional[int]:
        """
        This function computes an extra hash for a sequence, specifically
        designed for prefix caching mode. The final sequence hash is determined
        by applying token_ids from the sequence's blocks.
        """
        if self.lora_int_id == 0:
            return None

        # NOTE: If there are additional factors influencing the block aside from
        # token_ids, include them as input parameters to the hash.
        return hash(self.lora_int_id)

    def num_hashed_tokens_of_block(self, logical_idx: int):
        return logical_idx * self.block_size + self.block_size

    def reset_state_for_recompute(self):
        """Reset the sequence states for recomputation."""
        self.data.reset_state_for_recompute()

    def append_token_id(self,
                        token_id: int,
                        logprobs: dict[int, Logprob],
                        token_embed: Optional[torch.Tensor] = None) -> None:
        assert token_id in logprobs
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id].logprob,
                                  token_embed)

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> list[int]:
        return self.data.get_token_ids()

    def get_prompt_token_ids(self) -> tuple[int, ...]:
        return self.data.get_prompt_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> tuple[int, ...]:
        return self.data.get_output_token_ids()

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "VllmSequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def get_num_new_tokens(self) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., 1 for decode, or
            the remaining prompt size for prefill.
        """
        if self.data.stage == SequenceStage.DECODE:
            return 1
        return self.data.get_num_uncomputed_tokens()

    def get_num_computed_tokens(self) -> int:
        return self.data.get_num_computed_tokens()

    def is_prefill(self) -> bool:
        return self.data.stage == SequenceStage.PREFILL

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={self.n_blocks})")
