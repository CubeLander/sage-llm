# Summary

[](){ #configuration }

## Configuration

API documentation for vLLM's configuration classes.

- [vllm.config.ModelConfig][]
- [vllm.config.CacheConfig][]
- [vllm.config.LoadConfig][]
- [vllm.config.ParallelConfig][]
- [vllm.config.SchedulerConfig][]
- [vllm.config.DeviceConfig][]
- [vllm.config.SpeculativeConfig][]
- [vllm.config.LoRAConfig][]
- [vllm.config.MultiModalConfig][]
- [vllm.config.PoolerConfig][]
- [vllm.config.DecodingConfig][]
- [vllm.config.ObservabilityConfig][]
- [vllm.config.KVTransferConfig][]
- [vllm.config.CompilationConfig][]
- [vllm.config.VllmConfig][]

[](){ #offline-inference-api }

## Offline Inference

LLM Class.

- [vllm.LLM][]

LLM Inputs.

- [vllm.io.inputs.PromptType][]
- [vllm.io.inputs.TextPrompt][]
- [vllm.io.inputs.TokensPrompt][]

## vLLM Engines

Engine classes for offline and online inference.

- [vllm.LLMEngine][]
- [vllm.AsyncLLMEngine][]

## Inference Parameters

Inference parameters for vLLM APIs.

[](){ #sampling-params }
[](){ #pooling-params }

- [vllm.SamplingParams][]
- [vllm.PoolingParams][]

[](){ #multi-modality }

## Multi-Modality

vLLM provides experimental support for multi-modal models through the [vllm.io.inputs.multimodal][] package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models][supported-mm-models]
via the `multi_modal_data` field in [vllm.io.inputs.PromptType][].

Looking to add your own multi-modal model? Please follow the instructions listed [here](../contributing/model/multimodal.md).

- [vllm.io.inputs.multimodal.MULTIMODAL_REGISTRY][]

### Inputs

User-facing inputs.

- [vllm.io.inputs.multimodal.inputs.MultiModalDataDict][]

Internal data structures.

- [vllm.io.inputs.multimodal.inputs.PlaceholderRange][]
- [vllm.io.inputs.multimodal.inputs.NestedTensors][]
- [vllm.io.inputs.multimodal.inputs.MultiModalFieldElem][]
- [vllm.io.inputs.multimodal.inputs.MultiModalFieldConfig][]
- [vllm.io.inputs.multimodal.inputs.MultiModalKwargsItem][]
- [vllm.io.inputs.multimodal.inputs.MultiModalKwargs][]
- [vllm.io.inputs.multimodal.inputs.MultiModalInputs][]

### Data Parsing

- [vllm.io.inputs.multimodal.parse][]

### Data Processing

- [vllm.io.inputs.multimodal.processing][]

### Memory Profiling

- [vllm.io.inputs.multimodal.profiling][]

### Registry

- [vllm.io.inputs.multimodal.registry][]

## Model Development

- [vllm.model_executor.models.interfaces_base][]
- [vllm.model_executor.models.interfaces][]
- [vllm.model_executor.models.adapters][]
