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

- [vllm.inputs.PromptType][]
- [vllm.inputs.TextPrompt][]
- [vllm.inputs.TokensPrompt][]

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

vLLM provides experimental support for multi-modal models through the [vllm.inputs.multimodal][] package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models][supported-mm-models]
via the `multi_modal_data` field in [vllm.inputs.PromptType][].

Looking to add your own multi-modal model? Please follow the instructions listed [here](../contributing/model/multimodal.md).

- [vllm.inputs.multimodal.MULTIMODAL_REGISTRY][]

### Inputs

User-facing inputs.

- [vllm.inputs.multimodal.inputs.MultiModalDataDict][]

Internal data structures.

- [vllm.inputs.multimodal.inputs.PlaceholderRange][]
- [vllm.inputs.multimodal.inputs.NestedTensors][]
- [vllm.inputs.multimodal.inputs.MultiModalFieldElem][]
- [vllm.inputs.multimodal.inputs.MultiModalFieldConfig][]
- [vllm.inputs.multimodal.inputs.MultiModalKwargsItem][]
- [vllm.inputs.multimodal.inputs.MultiModalKwargs][]
- [vllm.inputs.multimodal.inputs.MultiModalInputs][]

### Data Parsing

- [vllm.inputs.multimodal.parse][]

### Data Processing

- [vllm.inputs.multimodal.processing][]

### Memory Profiling

- [vllm.inputs.multimodal.profiling][]

### Registry

- [vllm.inputs.multimodal.registry][]

## Model Development

- [vllm.model_executor.models.interfaces_base][]
- [vllm.model_executor.models.interfaces][]
- [vllm.model_executor.models.adapters][]
