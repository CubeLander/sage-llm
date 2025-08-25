# Model Conversion with ConvertOption

## Overview

The `ConvertOption` configuration in vLLM enables automatic model conversion and adaptation for different pooling tasks. This feature allows users to transform text generation models into specialized models for embedding, classification, and reward tasks without manual model modifications.

## Configuration

### ConvertOption Values

The `ConvertOption` is defined as:

```python
ConvertOption = Literal["auto", "none", "embed", "classify", "reward"]
```

### Usage

```bash
# Auto-detect conversion type based on model architecture
vllm serve model_name --convert auto

# Explicitly convert to embedding model
vllm serve model_name --convert embed

# Convert to classification model
vllm serve model_name --convert classify

# Convert to reward model  
vllm serve model_name --convert reward

# No conversion (default for most generative models)
vllm serve model_name --convert none
```

## Conversion Types

### 1. Auto Conversion (`--convert auto`)

When set to `"auto"`, vLLM automatically determines the conversion type based on the model's architecture name:

| Architecture Pattern | Conversion Type | Supported Tasks |
|---------------------|-----------------|-----------------|
| `*ForCausalLM` | `none` | Text generation |
| `*ForConditionalGeneration` | `none` | Text generation |
| `*ForTextEncoding` | `embed` | `encode`, `embed` |
| `*EmbeddingModel` | `embed` | `encode`, `embed` |
| `*ForSequenceClassification` | `classify` | `encode`, `classify`, `score` |
| `*ForAudioClassification` | `classify` | `encode`, `classify`, `score` |
| `*ForImageClassification` | `classify` | `encode`, `classify`, `score` |
| `*ForVideoClassification` | `classify` | `encode`, `classify`, `score` |
| `*ClassificationModel` | `classify` | `encode`, `classify`, `score` |
| `*ForRewardModeling` | `reward` | `encode` |
| `*RewardModel` | `reward` | `encode` |
| `*Model` | `embed` | `encode`, `embed` |

### 2. Embedding Conversion (`--convert embed`)

Converts models to support embedding tasks:

- **Supported Tasks**: `encode`, `embed`
- **Use Cases**: 
  - Text similarity
  - Semantic search
  - Document retrieval
  - Vector databases
- **API Methods**: `LLM.embed()`, `LLM.encode()`

### 3. Classification Conversion (`--convert classify`)

Converts models to support classification tasks:

- **Supported Tasks**: `encode`, `classify`, `score`
- **Use Cases**:
  - Text classification
  - Sentiment analysis
  - Topic categorization
  - Cross-encoder reranking
- **API Methods**: `LLM.classify()`, `LLM.score()`

### 4. Reward Conversion (`--convert reward`)

Converts models to support reward modeling:

- **Supported Tasks**: `encode`
- **Use Cases**:
  - RLHF (Reinforcement Learning from Human Feedback)
  - Text quality assessment
  - Preference modeling
- **API Methods**: `LLM.reward()`

### 5. No Conversion (`--convert none`)

Keeps the original model functionality without any conversion:

- **Supported Tasks**: `generate`, `transcription` (depending on model)
- **Use Cases**: Standard text generation
- **API Methods**: `LLM.generate()`

## Implementation Details

### Model Adapters

The conversion is implemented through adapters in `vllm/model_executor/models/adapters.py`:

```python
# Key adapter functions
def as_embedding_model(cls) -> cls  # For embed conversion
def as_seq_cls_model(cls) -> cls    # For classify conversion  
def as_reward_model(cls) -> cls     # For reward conversion
```

### Runner Integration

The conversion works in conjunction with the `RunnerOption`:

```python
_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],                              # No conversions for generate runner
    "pooling": ["embed", "classify", "reward"], # All conversions for pooling runner
    "draft": [],                                 # No conversions for draft runner
}
```

### Automatic Detection Logic

1. vLLM first determines if the model is a generative or pooling model
2. Based on the architecture suffix, it selects appropriate runner and conversion types
3. If conversion is needed, it applies the corresponding adapter
4. The converted model implements the appropriate pooling interface

## Examples

### Converting a Generative Model to Embedding Model

```python
from vllm import LLM

# Original generative model converted to embedding model
llm = LLM(model="microsoft/DialoGPT-medium", runner="pooling", convert="embed")

# Now supports embedding tasks
embeddings = llm.embed(["Hello world", "How are you?"])
```

### Auto-Conversion Based on Architecture

```python
# Model with ForSequenceClassification architecture
llm = LLM(model="cardiffnlp/twitter-roberta-base-sentiment-latest", convert="auto")
# Automatically converts to classify type

# Use classification API
results = llm.classify(["I love this!", "This is terrible"])
```

### Explicit Reward Model Conversion

```python
# Convert any compatible model to reward model
llm = LLM(model="bert-base-uncased", convert="reward")

# Use for reward modeling
rewards = llm.reward(["Good response", "Bad response"])
```

## Compatibility

### Model Requirements

- Models must be compatible with vLLM's pooling infrastructure
- Original model architecture should support the target conversion type
- Model must have appropriate hidden states for pooling operations

### Runner Compatibility

| Runner Type | Compatible Conversions |
|-------------|----------------------|
| `generate` | `none` |
| `pooling` | `embed`, `classify`, `reward` |
| `draft` | `none` |

### Task API Mapping

| Conversion Type | Available APIs | Task Types |
|----------------|---------------|------------|
| `none` | `generate()` | `generate`, `transcription` |
| `embed` | `embed()`, `encode()` | `encode`, `embed` |
| `classify` | `classify()`, `score()` | `encode`, `classify`, `score` |
| `reward` | `reward()` | `encode` |

## Best Practices

### 1. Use Auto-Conversion When Possible

```bash
# Let vLLM decide the best conversion
vllm serve model_name --convert auto
```

### 2. Explicit Conversion for Non-Standard Architectures

```bash
# For models that don't follow standard naming conventions
vllm serve custom_model --convert embed --runner pooling
```

### 3. Verify Conversion Support

Before using conversion, ensure your model architecture is supported:

```python
from vllm import LLM

try:
    llm = LLM(model="your_model", convert="embed")
    print("Conversion successful!")
except ValueError as e:
    print(f"Conversion failed: {e}")
```

## Troubleshooting

### Common Issues

1. **"This model does not support --runner pooling"**
   - Solution: Use explicit conversion with `--convert <type>`

2. **Architecture not recognized**
   - Solution: Use explicit `--runner pooling --convert <type>`

3. **Conversion type mismatch**
   - Solution: Check model architecture and use appropriate conversion type

### Debug Information

Enable logging to see conversion decisions:

```python
import logging
logging.basicConfig(level=logging.INFO)

# vLLM will log conversion decisions
llm = LLM(model="your_model", convert="auto")
```

## Related Documentation

- [Pooling Models](../models/pooling_models.md)
- [Model Configuration](./model_resolution.md)
- [Engine Arguments](./engine_args.md)
- [Supported Models](../models/supported_models.md)

## Changelog

- **v0.x.x**: Initial implementation of ConvertOption
- **v0.x.x**: Added auto-conversion based on architecture patterns
- **v0.x.x**: Enhanced support for reward model conversion
