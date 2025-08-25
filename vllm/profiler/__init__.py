# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM GPU性能监控模块

这个模块提供了全面的vLLM GPU利用率和性能监控功能，包括：
- GPU硬件层面的实时监控
- CUDA内核执行性能分析
- PyTorch操作级别的延迟和资源使用监控
- 模型层级的性能指标统计

主要组件：
- gpu_monitor: GPU硬件监控和torch操作分析
- vllm_monitor: vLLM专用的性能监控器
- integration: 无缝集成到现有vLLM代码的工具
- cli: 命令行监控工具
"""

from vllm.profiler.gpu_monitor import (
    GPUHardwareMonitor,
    TorchOpProfiler,
    LayerwiseProfiler,
    GPUMetrics,
    TorchOpMetrics,
    LayerMetrics,
    monitor_vllm_execution,
    profile_vllm_layer
)

from vllm.profiler.vllm_monitor import (
    VLLMPerformanceMonitor,
    InferenceStepMetrics,
    monitor_vllm_model_runner,
    profile_vllm_inference
)

from vllm.profiler.integration import (
    enable_vllm_profiling,
    disable_vllm_profiling,
    get_monitoring_summary,
    VLLMProfilingContext,
    profile_vllm_function
)

# 版本信息
__version__ = "1.0.0"

# 导出的主要接口
__all__ = [
    # GPU监控相关
    "GPUHardwareMonitor",
    "TorchOpProfiler", 
    "LayerwiseProfiler",
    "GPUMetrics",
    "TorchOpMetrics",
    "LayerMetrics",
    "monitor_vllm_execution",
    "profile_vllm_layer",
    
    # vLLM专用监控
    "VLLMPerformanceMonitor",
    "InferenceStepMetrics", 
    "monitor_vllm_model_runner",
    "profile_vllm_inference",
    
    # 集成工具
    "enable_vllm_profiling",
    "disable_vllm_profiling", 
    "get_monitoring_summary",
    "VLLMProfilingContext",
    "profile_vllm_function"
]
