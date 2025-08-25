# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union, Any, Dict
from dataclasses import dataclass

from vllm.config import (
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    SchedulerConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    MultiModalConfig,
    PoolerConfig,
    SpeculativeConfig,
    DecodingConfig,
    ObservabilityConfig,
    VllmConfig
)
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class HotLLMModelConfig:
    """Simplified model configuration for HotLLM."""
    model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "auto"
    seed: Optional[int] = None
    max_model_len: Optional[int] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192


@dataclass
class HotLLMEngineConfig:
    """Simplified engine configuration for HotLLM."""
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    block_size: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    disable_log_stats: bool = True
    disable_async_output_proc: bool = False


@dataclass
class HotLLMParallelConfig:
    """Simplified parallel configuration for HotLLM."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    disable_custom_all_reduce: bool = False
    enable_expert_parallel: bool = False
    distributed_executor_backend: Optional[str] = None


class LLM:
    """Simplified LLM interface for HotLLM.
    
    This class provides a cleaner interface compared to the original vLLM LLM class,
    accepting only essential configuration parameters grouped into model_config,
    engine_config, and parallel_config.
    
    Args:
        model_config: Configuration for the model. If None, uses default values.
        engine_config: Configuration for the engine. If None, uses default values.
        parallel_config: Configuration for parallelism. If None, uses default values.
        **kwargs: Additional arguments passed to the underlying EngineArgs.
    """
    
    def __init__(
        self,
        model_config: Optional[HotLLMModelConfig] = None,
        engine_config: Optional[HotLLMEngineConfig] = None,
        parallel_config: Optional[HotLLMParallelConfig] = None,
        **kwargs
    ):
        # Use default configs if not provided
        if model_config is None:
            model_config = HotLLMModelConfig()
        if engine_config is None:
            engine_config = HotLLMEngineConfig()
        if parallel_config is None:
            parallel_config = HotLLMParallelConfig()
            
        # Convert simplified configs to EngineArgs
        engine_args = self._create_engine_args(
            model_config, engine_config, parallel_config, **kwargs
        )
        
        # Create the engine
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, 
            usage_context=UsageContext.LLM_CLASS
        )
        self.engine_class = type(self.llm_engine)
        
        logger.info("HotLLM initialized successfully")
        logger.info(f"Model: {model_config.model}")
        logger.info(f"Tensor parallel size: {parallel_config.tensor_parallel_size}")
        logger.info(f"GPU memory utilization: {engine_config.gpu_memory_utilization}")
    
    def _create_engine_args(
        self,
        model_config: HotLLMModelConfig,
        engine_config: HotLLMEngineConfig,
        parallel_config: HotLLMParallelConfig,
        **kwargs
    ) -> EngineArgs:
        """Convert simplified configs to EngineArgs."""
        
        # Merge kwargs with config values, giving priority to kwargs
        args_dict = {
            # Model configuration
            "model": model_config.model,
            "dtype": model_config.dtype,
            "seed": model_config.seed,
            "max_model_len": model_config.max_model_len,
            "trust_remote_code": model_config.trust_remote_code,
            "revision": model_config.revision,
            "quantization": model_config.quantization,
            "enforce_eager": model_config.enforce_eager,
            "max_seq_len_to_capture": model_config.max_seq_len_to_capture,
            
            # Engine configuration
            "gpu_memory_utilization": engine_config.gpu_memory_utilization,
            "swap_space": engine_config.swap_space,
            "cpu_offload_gb": engine_config.cpu_offload_gb,
            "max_num_batched_tokens": engine_config.max_num_batched_tokens,
            "max_num_seqs": engine_config.max_num_seqs,
            "block_size": engine_config.block_size,
            "enable_prefix_caching": engine_config.enable_prefix_caching,
            "disable_log_stats": engine_config.disable_log_stats,
            "disable_async_output_proc": engine_config.disable_async_output_proc,
            
            # Parallel configuration
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
            "enable_expert_parallel": parallel_config.enable_expert_parallel,
            "distributed_executor_backend": parallel_config.distributed_executor_backend,
        }
        
        # Override with any additional kwargs
        args_dict.update(kwargs)
        
        # Remove None values
        args_dict = {k: v for k, v in args_dict.items() if v is not None}
        
        return EngineArgs(**args_dict)
    
    # Delegate methods to the underlying engine
    def generate(self, *args, **kwargs):
        """Generate text from prompts."""
        # Import here to avoid circular imports
        from vllm.entrypoints.llm import LLM as OriginalLLM
        
        # Create a temporary instance with our engine to use the generate method
        original_llm = OriginalLLM.__new__(OriginalLLM)
        original_llm.llm_engine = self.llm_engine
        original_llm.engine_class = self.engine_class
        
        # Set up required attributes
        from vllm.utils import Counter
        original_llm.request_counter = Counter()
        original_llm.default_sampling_params = None
        
        # Get supported tasks
        try:
            from vllm import envs
            if hasattr(envs, 'VLLM_USE_V1') and envs.VLLM_USE_V1:
                original_llm.supported_tasks = self.llm_engine.get_supported_tasks()
            else:
                original_llm.supported_tasks = self.llm_engine.model_config.supported_tasks
        except:
            original_llm.supported_tasks = ["generate"]
        
        return original_llm.generate(*args, **kwargs)
    
    def chat(self, *args, **kwargs):
        """Generate chat responses."""
        from vllm.entrypoints.llm import LLM as OriginalLLM
        
        original_llm = OriginalLLM.__new__(OriginalLLM)
        original_llm.llm_engine = self.llm_engine
        original_llm.engine_class = self.engine_class
        
        from vllm.utils import Counter
        original_llm.request_counter = Counter()
        original_llm.default_sampling_params = None
        
        try:
            from vllm import envs
            if hasattr(envs, 'VLLM_USE_V1') and envs.VLLM_USE_V1:
                original_llm.supported_tasks = self.llm_engine.get_supported_tasks()
            else:
                original_llm.supported_tasks = self.llm_engine.model_config.supported_tasks
        except:
            original_llm.supported_tasks = ["generate"]
        
        return original_llm.chat(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        """Encode text to embeddings."""
        from vllm.entrypoints.llm import LLM as OriginalLLM
        
        original_llm = OriginalLLM.__new__(OriginalLLM)
        original_llm.llm_engine = self.llm_engine
        original_llm.engine_class = self.engine_class
        
        from vllm.utils import Counter
        original_llm.request_counter = Counter()
        original_llm.default_sampling_params = None
        
        try:
            from vllm import envs
            if hasattr(envs, 'VLLM_USE_V1') and envs.VLLM_USE_V1:
                original_llm.supported_tasks = self.llm_engine.get_supported_tasks()
            else:
                original_llm.supported_tasks = self.llm_engine.model_config.supported_tasks
        except:
            original_llm.supported_tasks = ["generate", "embed"]
        
        return original_llm.encode(*args, **kwargs)
    
    def get_tokenizer(self, *args, **kwargs):
        """Get the tokenizer."""
        return self.llm_engine.get_tokenizer_group().get_lora_tokenizer(*args, **kwargs)
    
    def set_tokenizer(self, tokenizer):
        """Set a custom tokenizer."""
        from vllm.entrypoints.llm import LLM as OriginalLLM
        
        original_llm = OriginalLLM.__new__(OriginalLLM)
        original_llm.llm_engine = self.llm_engine
        return original_llm.set_tokenizer(tokenizer)
    
    @property
    def model_config(self):
        """Access the model configuration."""
        return self.llm_engine.model_config
    
    @property
    def cache_config(self):
        """Access the cache configuration."""
        if hasattr(self.llm_engine, 'cache_config'):
            return self.llm_engine.cache_config
        return getattr(self.llm_engine, 'vllm_config', {}).get('cache_config')
    
    @property
    def parallel_config(self):
        """Access the parallel configuration."""
        if hasattr(self.llm_engine, 'parallel_config'):
            return self.llm_engine.parallel_config
        return getattr(self.llm_engine, 'vllm_config', {}).get('parallel_config')