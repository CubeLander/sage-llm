# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .data import DecoderOnlyInputs
from .data import EmbedsInputs
from .data import EncoderDecoderInputs
from .data import ExplicitEncoderDecoderPrompt
from .data import ProcessorInputs
from .data import PromptType
from .data import SingletonInputs
from .data import SingletonPrompt
from .data import TextPrompt
from .data import TokenInputs
from .data import TokensPrompt
from .data import build_explicit_enc_dec_prompt
from .data import embeds_inputs
from .data import to_enc_dec_tuple_list
from .data import token_inputs
from .data import zip_enc_dec_prompts
from .registry import DummyData
from .registry import InputContext
from .registry import InputProcessingContext
from .registry import InputRegistry

INPUT_REGISTRY = InputRegistry()
"""
The global [`InputRegistry`][vllm.io.inputs.registry.InputRegistry] which is used
by [`LLMEngine`][vllm.LLMEngine] to dispatch data processing according to the
target model.
"""

__all__ = [
    "TextPrompt",
    "TokensPrompt",
    "PromptType",
    "SingletonPrompt",
    "ExplicitEncoderDecoderPrompt",
    "TokenInputs",
    "EmbedsInputs",
    "token_inputs",
    "embeds_inputs",
    "DecoderOnlyInputs",
    "EncoderDecoderInputs",
    "ProcessorInputs",
    "SingletonInputs",
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "DummyData",
    "InputContext",
    "InputProcessingContext",
    "InputRegistry",
]
