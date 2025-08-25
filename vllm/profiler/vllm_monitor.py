# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM实时性能监控集成工具

这个工具可以在vLLM运行时实时监控：
1. GPU硬件利用率 (GPU使用率、显存、温度、功耗)
2. CUDA内核执行性能
3. 每个torch操作的延迟和资源使用
4. 模型每一层的性能指标

使用方法:
```python
from vllm.profiler.vllm_monitor import VLLMPerformanceMonitor

# 在模型初始化后添加监控
monitor = VLLMPerformanceMonitor(model_runner, enable_realtime_monitoring=True)
monitor.start_monitoring()

# 在推理过程中
with monitor.profile_inference_step() as metrics:
    # 正常的vLLM推理代码
    outputs = model_runner.execute_model(...)
    
# 导出监控报告
monitor.export_performance_report("vllm_performance_report.json")
```
"""

import json
import time
import threading
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from vllm.logger import init_logger
from vllm.profiler.gpu_monitor import (
    GPUHardwareMonitor, TorchOpProfiler, LayerMetrics, 
    GPUMetrics, TorchOpMetrics, KernelMetrics
)

logger = init_logger(__name__)


class VLLMModelHooks:
    """为vLLM模型添加性能监控钩子"""
    
    def __init__(self, profiler: 'VLLMPerformanceMonitor'):
        self.profiler = profiler
        self.hooks = []
        self.layer_timings = defaultdict(list)
        
    def register_hooks(self, model: nn.Module):
        """为模型的每一层注册前向钩子"""
        
        def create_pre_hook(layer_name):
            def pre_hook(module, input):
                if self.profiler.monitoring_active:
                    self.profiler._start_layer_timing(layer_name)
            return pre_hook
            
        def create_post_hook(layer_name):
            def post_hook(module, input, output):
                if self.profiler.monitoring_active:
                    self.profiler._end_layer_timing(layer_name, input, output)
            return post_hook
            
        # 遍历模型的所有层
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                # 注册前向钩子
                pre_hook = module.register_forward_pre_hook(create_pre_hook(name))
                post_hook = module.register_forward_hook(create_post_hook(name))
                
                self.hooks.extend([pre_hook, post_hook])
                
        logger.info(f"Registered monitoring hooks for {len(self.hooks)//2} layers")
        
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("Removed all monitoring hooks")


class VLLMPerformanceMonitor:
    """vLLM专用的性能监控器"""
    
    def __init__(self, 
                 model_runner = None,
                 device_id: int = 0,
                 enable_realtime_monitoring: bool = True,
                 enable_detailed_profiling: bool = False,
                 sample_interval: float = 0.1,
                 max_history_size: int = 10000):
        """
        初始化vLLM性能监控器
        
        Args:
            model_runner: vLLM的ModelRunner实例
            device_id: GPU设备ID
            enable_realtime_monitoring: 是否启用实时硬件监控
            enable_detailed_profiling: 是否启用详细的torch profiling
            sample_interval: 硬件监控采样间隔(秒)
            max_history_size: 保存的最大历史记录数
        """
        self.model_runner = model_runner
        self.device_id = device_id
        self.enable_realtime_monitoring = enable_realtime_monitoring
        self.enable_detailed_profiling = enable_detailed_profiling
        self.max_history_size = max_history_size
        
        # 监控状态
        self.monitoring_active = False
        self.step_counter = 0
        
        # 初始化监控组件
        if enable_realtime_monitoring:
            self.gpu_monitor = GPUHardwareMonitor(device_id, sample_interval)
        else:
            self.gpu_monitor = None
            
        if enable_detailed_profiling:
            self.torch_profiler = TorchOpProfiler()
        else:
            self.torch_profiler = None
            
        # 性能数据存储
        self.inference_metrics = deque(maxlen=max_history_size)
        self.layer_timings = defaultdict(list)
        self.current_step_start = None
        self.current_layer_timings = {}
        
        # 模型钩子管理器
        self.hooks_manager = VLLMModelHooks(self)
        
    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
            
        self.monitoring_active = True
        
        # 启动GPU硬件监控
        if self.gpu_monitor:
            self.gpu_monitor.start()
            
        # 为模型添加钩子
        if self.model_runner and hasattr(self.model_runner, 'model'):
            self.hooks_manager.register_hooks(self.model_runner.model)
            
        logger.info("vLLM performance monitoring started")
        
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        
        # 停止GPU硬件监控
        if self.gpu_monitor:
            self.gpu_monitor.stop()
            
        # 移除模型钩子
        self.hooks_manager.remove_hooks()
        
        logger.info("vLLM performance monitoring stopped")
        
    @contextmanager
    def profile_inference_step(self, 
                              batch_size: Optional[int] = None,
                              sequence_length: Optional[int] = None,
                              num_tokens: Optional[int] = None):
        """
        监控一次推理步骤的上下文管理器
        
        Args:
            batch_size: 批量大小
            sequence_length: 序列长度  
            num_tokens: token数量
            
        Yields:
            InferenceStepMetrics: 当前推理步骤的性能指标
        """
        step_id = self.step_counter
        self.step_counter += 1
        
        # 记录步骤开始
        step_start_time = time.time()
        self.current_step_start = step_start_time
        
        # 记录GPU状态
        gpu_start = None
        if self.gpu_monitor:
            gpu_start = self.gpu_monitor.get_current_metrics()
            
        # 记录内存状态
        torch.cuda.reset_peak_memory_stats()
        memory_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 开始详细profiling（如果启用）
        profiler_context = None
        if self.torch_profiler and self.enable_detailed_profiling:
            profiler_context = self.torch_profiler.profile_context()
            profiler_context.__enter__()
        
        try:
            # 创建step metrics对象
            step_metrics = InferenceStepMetrics(
                step_id=step_id,
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_tokens=num_tokens,
                start_time=step_start_time
            )
            
            yield step_metrics
            
        finally:
            # 记录步骤结束
            step_end_time = time.time()
            step_duration = (step_end_time - step_start_time) * 1000  # 转换为毫秒
            
            # 记录内存使用
            memory_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (memory_peak - memory_start) / 1e6  # 转换为MB
            
            # 记录GPU状态
            gpu_end = None
            if self.gpu_monitor:
                gpu_end = self.gpu_monitor.get_current_metrics()
                
            # 结束详细profiling
            if profiler_context:
                profiler_context.__exit__(None, None, None)
                
            # 更新step metrics
            step_metrics.end_time = step_end_time
            step_metrics.total_duration_ms = step_duration
            step_metrics.memory_used_mb = memory_used
            step_metrics.gpu_start = gpu_start
            step_metrics.gpu_end = gpu_end
            step_metrics.layer_timings = dict(self.current_layer_timings)
            
            # 保存到历史记录
            self.inference_metrics.append(step_metrics)
            
            # 清理当前步骤状态
            self.current_layer_timings.clear()
            self.current_step_start = None
            
            # 日志输出
            if step_id % 100 == 0:  # 每100步输出一次
                self._log_performance_summary(step_metrics)
                
    def _start_layer_timing(self, layer_name: str):
        """开始计时某一层"""
        if not self.monitoring_active:
            return
        self.current_layer_timings[layer_name] = {
            'start_time': time.time(),
            'start_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        
    def _end_layer_timing(self, layer_name: str, input_tensors, output_tensors):
        """结束计时某一层"""
        if not self.monitoring_active or layer_name not in self.current_layer_timings:
            return
            
        timing_info = self.current_layer_timings[layer_name]
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        duration_ms = (end_time - timing_info['start_time']) * 1000
        memory_delta_mb = (end_memory - timing_info['start_memory']) / 1e6
        
        # 获取张量形状信息
        input_shapes = self._extract_tensor_shapes(input_tensors)
        output_shapes = self._extract_tensor_shapes(output_tensors)
        
        self.current_layer_timings[layer_name] = {
            'duration_ms': duration_ms,
            'memory_delta_mb': memory_delta_mb,
            'input_shapes': input_shapes,
            'output_shapes': output_shapes
        }
        
    def _extract_tensor_shapes(self, tensors) -> List[List[int]]:
        """提取张量形状信息"""
        shapes = []
        
        if isinstance(tensors, torch.Tensor):
            shapes.append(list(tensors.shape))
        elif isinstance(tensors, (list, tuple)):
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor):
                    shapes.append(list(tensor.shape))
        
        return shapes
        
    def _log_performance_summary(self, step_metrics: 'InferenceStepMetrics'):
        """输出性能摘要日志"""
        gpu_util = 0.0
        memory_util = 0.0
        
        if step_metrics.gpu_end:
            gpu_util = step_metrics.gpu_end.gpu_utilization
            memory_util = step_metrics.gpu_end.memory_utilization
            
        logger.info(
            f"Step {step_metrics.step_id}: "
            f"Duration={step_metrics.total_duration_ms:.1f}ms, "
            f"GPU={gpu_util:.1f}%, "
            f"Memory={memory_util:.1f}%, "
            f"MemUsed={step_metrics.memory_used_mb:.1f}MB, "
            f"Layers={len(step_metrics.layer_timings)}"
        )
        
    def get_performance_summary(self, last_n_steps: Optional[int] = None) -> Dict[str, Any]:
        """获取性能摘要统计"""
        if not self.inference_metrics:
            return {}
            
        # 选择要分析的步骤
        steps_to_analyze = list(self.inference_metrics)
        if last_n_steps:
            steps_to_analyze = steps_to_analyze[-last_n_steps:]
            
        # 计算统计信息
        durations = [s.total_duration_ms for s in steps_to_analyze if s.total_duration_ms]
        memory_usage = [s.memory_used_mb for s in steps_to_analyze if s.memory_used_mb]
        
        summary = {
            'total_steps': len(steps_to_analyze),
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0,
            'avg_memory_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'max_memory_mb': max(memory_usage) if memory_usage else 0,
        }
        
        # 层级性能统计
        layer_stats = defaultdict(list)
        for step in steps_to_analyze:
            for layer_name, layer_info in step.layer_timings.items():
                if isinstance(layer_info, dict) and 'duration_ms' in layer_info:
                    layer_stats[layer_name].append(layer_info['duration_ms'])
                    
        summary['layer_performance'] = {
            layer_name: {
                'avg_duration_ms': sum(durations) / len(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'call_count': len(durations)
            } for layer_name, durations in layer_stats.items()
        }
        
        return summary
        
    def export_performance_report(self, output_file: Union[str, Path]):
        """导出详细的性能报告"""
        output_file = Path(output_file)
        
        # 收集所有监控数据
        report = {
            'metadata': {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'total_steps': len(self.inference_metrics),
                'monitoring_duration_sec': 0
            },
            'summary': self.get_performance_summary(),
            'detailed_steps': []
        }
        
        # 计算监控总时长
        if self.inference_metrics:
            first_step = self.inference_metrics[0]
            last_step = self.inference_metrics[-1]
            if first_step.end_time and last_step.end_time:
                report['metadata']['monitoring_duration_sec'] = \
                    last_step.end_time - first_step.start_time
        
        # 添加详细的步骤数据（限制数量避免文件过大）
        steps_to_export = list(self.inference_metrics)[-1000:]  # 最后1000步
        
        for step in steps_to_export:
            step_data = {
                'step_id': step.step_id,
                'batch_size': step.batch_size,
                'sequence_length': step.sequence_length,
                'num_tokens': step.num_tokens,
                'duration_ms': step.total_duration_ms,
                'memory_used_mb': step.memory_used_mb,
                'layer_timings': step.layer_timings
            }
            
            # 添加GPU指标
            if step.gpu_start and step.gpu_end:
                step_data['gpu_metrics'] = {
                    'start_utilization': step.gpu_start.gpu_utilization,
                    'end_utilization': step.gpu_end.gpu_utilization,
                    'start_memory_util': step.gpu_start.memory_utilization,
                    'end_memory_util': step.gpu_end.memory_utilization
                }
                
            report['detailed_steps'].append(step_data)
            
        # 添加GPU监控历史（如果可用）
        if self.gpu_monitor:
            gpu_history = self.gpu_monitor.get_metrics_history(300.0)  # 最近5分钟
            report['gpu_history'] = [
                {
                    'timestamp': m.timestamp,
                    'gpu_utilization': m.gpu_utilization,
                    'memory_utilization': m.memory_utilization,
                    'memory_used': m.memory_used,
                    'temperature': m.temperature,
                    'power_draw': m.power_draw
                } for m in gpu_history
            ]
        
        # 写入文件
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance report exported to {output_file}")
        
    def __enter__(self):
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()


class InferenceStepMetrics:
    """单次推理步骤的性能指标"""
    
    def __init__(self, 
                 step_id: int,
                 batch_size: Optional[int] = None,
                 sequence_length: Optional[int] = None,
                 num_tokens: Optional[int] = None,
                 start_time: Optional[float] = None):
        self.step_id = step_id
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        self.start_time = start_time
        
        # 将在推理过程中填充的字段
        self.end_time: Optional[float] = None
        self.total_duration_ms: Optional[float] = None
        self.memory_used_mb: Optional[float] = None
        self.gpu_start: Optional[GPUMetrics] = None
        self.gpu_end: Optional[GPUMetrics] = None
        self.layer_timings: Dict[str, Any] = {}


# 便捷函数和装饰器
def monitor_vllm_model_runner(model_runner,
                             enable_realtime: bool = True,
                             enable_detailed: bool = False,
                             output_dir: Optional[str] = None):
    """
    为vLLM ModelRunner添加性能监控的装饰器
    
    使用方法:
    ```python
    model_runner = monitor_vllm_model_runner(
        model_runner, 
        enable_realtime=True,
        output_dir="./profiling_results"
    )
    ```
    """
    monitor = VLLMPerformanceMonitor(
        model_runner, 
        enable_realtime_monitoring=enable_realtime,
        enable_detailed_profiling=enable_detailed
    )
    
    # 为model_runner的execute_model方法添加监控
    original_execute_model = model_runner.execute_model
    
    def monitored_execute_model(*args, **kwargs):
        scheduler_output = args[0] if args else None
        batch_size = getattr(scheduler_output, 'num_scheduled_tokens', None) if scheduler_output else None
        
        with monitor.profile_inference_step(batch_size=batch_size):
            return original_execute_model(*args, **kwargs)
    
    model_runner.execute_model = monitored_execute_model
    model_runner._performance_monitor = monitor
    
    # 启动监控
    monitor.start_monitoring()
    
    # 添加清理方法
    def stop_monitoring():
        monitor.stop_monitoring()
        if output_dir:
            output_file = Path(output_dir) / f"vllm_performance_{int(time.time())}.json"
            monitor.export_performance_report(output_file)
    
    model_runner.stop_performance_monitoring = stop_monitoring
    
    return model_runner


@contextmanager
def profile_vllm_inference(model_runner, output_file: Optional[str] = None):
    """
    用于临时性能分析的上下文管理器
    
    使用方法:
    ```python
    with profile_vllm_inference(model_runner, "performance.json"):
        # 执行推理
        for i in range(100):
            outputs = model_runner.execute_model(scheduler_output)
    ```
    """
    monitor = VLLMPerformanceMonitor(
        model_runner,
        enable_realtime_monitoring=True,
        enable_detailed_profiling=True
    )
    
    with monitor:
        yield monitor
        
    if output_file:
        monitor.export_performance_report(output_file)
