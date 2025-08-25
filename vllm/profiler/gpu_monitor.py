# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import subprocess
import psutil

import torch
import torch.nn as nn
from torch._C._autograd import DeviceType, _KinetoEvent
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from vllm.logger import init_logger
from vllm.utils import is_hip

logger = init_logger(__name__)

@dataclass
class GPUMetrics:
    """GPU硬件层面的监控指标"""
    timestamp: float
    gpu_utilization: float  # GPU利用率 (0-100%)
    memory_utilization: float  # 显存利用率 (0-100%)
    memory_used: float  # 已使用显存 (MB)
    memory_total: float  # 总显存 (MB)
    temperature: Optional[float] = None  # GPU温度 (°C)
    power_draw: Optional[float] = None  # 功耗 (W)
    compute_mode: Optional[str] = None  # 计算模式

@dataclass 
class KernelMetrics:
    """CUDA内核层面的监控指标"""
    kernel_name: str
    timestamp: float
    duration_us: float  # 执行时间 (微秒)
    grid_size: tuple  # Grid大小
    block_size: tuple  # Block大小
    registers_per_thread: Optional[int] = None
    shared_memory_per_block: Optional[int] = None
    occupancy: Optional[float] = None  # 占用率

@dataclass
class TorchOpMetrics:
    """PyTorch操作层面的监控指标"""
    op_name: str
    layer_name: Optional[str]
    timestamp: float
    cpu_time_us: float  # CPU时间
    cuda_time_us: float  # CUDA时间
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    memory_delta: float  # 内存变化 (MB)
    flops: Optional[float] = None  # 浮点运算数

@dataclass
class LayerMetrics:
    """模型层面的监控指标"""
    layer_name: str
    layer_type: str
    timestamp: float
    forward_time_us: float  # 前向传播时间
    total_memory_mb: float  # 该层总内存使用
    peak_memory_mb: float  # 该层峰值内存
    torch_ops: List[TorchOpMetrics] = field(default_factory=list)


class GPUHardwareMonitor:
    """GPU硬件层面的实时监控"""
    
    def __init__(self, device_id: int = 0, sample_interval: float = 0.1):
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.running = False
        self.metrics_buffer = deque(maxlen=10000)  # 保存最近10000个样本
        self.monitor_thread = None
        
    def start(self):
        """开始监控"""
        if self.running:
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started GPU hardware monitoring for device {self.device_id}")
        
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped GPU hardware monitoring")
        
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """获取最新的GPU指标"""
        if not self.metrics_buffer:
            return None
        return self.metrics_buffer[-1]
        
    def get_metrics_history(self, duration_sec: Optional[float] = None) -> List[GPUMetrics]:
        """获取历史指标"""
        if duration_sec is None:
            return list(self.metrics_buffer)
        
        current_time = time.time()
        cutoff_time = current_time - duration_sec
        return [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = self._collect_gpu_metrics()
                if metrics:
                    self.metrics_buffer.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
            time.sleep(self.sample_interval)
            
    def _collect_gpu_metrics(self) -> Optional[GPUMetrics]:
        """收集GPU指标"""
        try:
            # 使用nvidia-smi或rocm-smi获取GPU指标
            if is_hip():
                return self._collect_rocm_metrics()
            else:
                return self._collect_nvidia_metrics()
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            return None
            
    def _collect_nvidia_metrics(self) -> Optional[GPUMetrics]:
        """使用nvidia-smi收集NVIDIA GPU指标"""
        try:
            cmd = [
                "nvidia-smi", 
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                f"--id={self.device_id}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            values = result.stdout.strip().split(', ')
            return GPUMetrics(
                timestamp=time.time(),
                gpu_utilization=float(values[0]) if values[0] != '[Not Supported]' else 0.0,
                memory_used=float(values[1]),
                memory_total=float(values[2]),
                memory_utilization=(float(values[1]) / float(values[2])) * 100,
                temperature=float(values[3]) if values[3] != '[Not Supported]' else None,
                power_draw=float(values[4]) if values[4] != '[Not Supported]' else None
            )
        except (subprocess.CalledProcessError, ValueError, IndexError) as e:
            logger.warning(f"Failed to parse nvidia-smi output: {e}")
            return None
            
    def _collect_rocm_metrics(self) -> Optional[GPUMetrics]:
        """使用rocm-smi收集AMD GPU指标"""
        try:
            cmd = ["rocm-smi", "--gpu", str(self.device_id), "--showuse", "--showmemuse"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 解析rocm-smi输出 (这里需要根据实际输出格式调整)
            # 这是一个简化的实现
            return GPUMetrics(
                timestamp=time.time(),
                gpu_utilization=0.0,  # 需要解析实际输出
                memory_utilization=0.0,
                memory_used=0.0,
                memory_total=0.0
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to run rocm-smi: {e}")
            return None


class TorchOpProfiler:
    """PyTorch操作级别的性能分析器"""
    
    def __init__(self, 
                 trace_gpu: bool = True,
                 trace_memory: bool = True,
                 with_stack: bool = True,
                 profile_memory: bool = True):
        self.trace_gpu = trace_gpu
        self.trace_memory = trace_memory
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.torch_ops_metrics: List[TorchOpMetrics] = []
        self.kernel_metrics: List[KernelMetrics] = []
        
    @contextmanager
    def profile_context(self, output_dir: Optional[str] = None):
        """PyTorch profiler上下文管理器"""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available() and self.trace_gpu:
            activities.append(ProfilerActivity.CUDA)
            
        # 配置profiler
        profiler_config = {
            'activities': activities,
            'record_shapes': True,
            'profile_memory': self.profile_memory,
            'with_stack': self.with_stack,
            'with_flops': True,
            'with_modules': True,
        }
        
        if output_dir:
            profiler_config['on_trace_ready'] = tensorboard_trace_handler(output_dir)
            
        with profile(**profiler_config) as prof:
            yield prof
            
        # 处理profiling结果
        self._process_profiler_results(prof)
        
    def _process_profiler_results(self, prof):
        """处理profiler结果"""
        # 按层级处理每个事件
        events = prof.events()
        current_time = time.time()
        
        for event in events:
            # 提取torch操作指标
            if event.device_type == DeviceType.CUDA:
                torch_op = TorchOpMetrics(
                    op_name=event.name,
                    layer_name=self._extract_layer_name(event),
                    timestamp=current_time,
                    cpu_time_us=event.cpu_time_total,
                    cuda_time_us=event.cuda_time_total,
                    input_shapes=event.input_shapes if hasattr(event, 'input_shapes') else [],
                    output_shapes=[],  # 需要从event中提取
                    memory_delta=event.cpu_memory_usage if hasattr(event, 'cpu_memory_usage') else 0.0,
                    flops=event.flops if hasattr(event, 'flops') else None
                )
                self.torch_ops_metrics.append(torch_op)
                
    def _extract_layer_name(self, event) -> Optional[str]:
        """从事件中提取层名称"""
        if hasattr(event, 'stack') and event.stack:
            # 从调用栈中提取层信息
            for frame in event.stack:
                if 'nn.modules' in frame.filename:
                    return frame.name
        return None
        
    def get_layer_summary(self) -> Dict[str, LayerMetrics]:
        """获取按层汇总的性能指标"""
        layer_stats = defaultdict(lambda: {
            'total_time': 0,
            'total_memory': 0,
            'ops': []
        })
        
        for op in self.torch_ops_metrics:
            layer_name = op.layer_name or 'unknown'
            layer_stats[layer_name]['total_time'] += op.cuda_time_us
            layer_stats[layer_name]['total_memory'] += op.memory_delta
            layer_stats[layer_name]['ops'].append(op)
            
        result = {}
        for layer_name, stats in layer_stats.items():
            result[layer_name] = LayerMetrics(
                layer_name=layer_name,
                layer_type='unknown',  # 需要从模型定义中获取
                timestamp=time.time(),
                forward_time_us=stats['total_time'],
                total_memory_mb=stats['total_memory'],
                peak_memory_mb=stats['total_memory'],  # 简化处理
                torch_ops=stats['ops']
            )
            
        return result


class LayerwiseProfiler:
    """层级性能分析器，集成GPU硬件监控和torch操作监控"""
    
    def __init__(self, 
                 model: nn.Module,
                 device_id: int = 0,
                 sample_interval: float = 0.1,
                 enable_hardware_monitoring: bool = True):
        self.model = model
        self.device_id = device_id
        self.enable_hardware_monitoring = enable_hardware_monitoring
        
        # 初始化监控器
        if enable_hardware_monitoring:
            self.gpu_monitor = GPUHardwareMonitor(device_id, sample_interval)
        else:
            self.gpu_monitor = None
            
        self.torch_profiler = TorchOpProfiler()
        
        # 存储监控结果
        self.layer_metrics: Dict[str, LayerMetrics] = {}
        self.profiling_active = False
        
    def start_monitoring(self):
        """开始监控"""
        if self.profiling_active:
            return
            
        self.profiling_active = True
        
        # 启动GPU硬件监控
        if self.gpu_monitor:
            self.gpu_monitor.start()
            
        logger.info("Started layerwise profiling")
        
    def stop_monitoring(self):
        """停止监控"""
        if not self.profiling_active:
            return
            
        self.profiling_active = False
        
        # 停止GPU硬件监控
        if self.gpu_monitor:
            self.gpu_monitor.stop()
            
        logger.info("Stopped layerwise profiling")
        
    @contextmanager
    def profile_forward_pass(self, output_dir: Optional[str] = None):
        """分析一次前向传播"""
        if not self.profiling_active:
            self.start_monitoring()
            
        with self.torch_profiler.profile_context(output_dir) as prof:
            yield prof
            
        # 更新层级指标
        self.layer_metrics.update(self.torch_profiler.get_layer_summary())
        
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控总结"""
        summary = {
            'timestamp': time.time(),
            'layer_metrics': {name: {
                'forward_time_us': metrics.forward_time_us,
                'memory_mb': metrics.total_memory_mb,
                'num_ops': len(metrics.torch_ops)
            } for name, metrics in self.layer_metrics.items()}
        }
        
        if self.gpu_monitor:
            current_gpu = self.gpu_monitor.get_current_metrics()
            if current_gpu:
                summary['gpu_metrics'] = {
                    'utilization': current_gpu.gpu_utilization,
                    'memory_utilization': current_gpu.memory_utilization,
                    'memory_used_mb': current_gpu.memory_used,
                    'temperature': current_gpu.temperature,
                    'power_draw': current_gpu.power_draw
                }
                
        return summary
        
    def export_detailed_report(self, output_file: str):
        """导出详细报告"""
        report = {
            'metadata': {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'model_name': self.model.__class__.__name__
            },
            'layer_metrics': [
                {
                    'layer_name': metrics.layer_name,
                    'layer_type': metrics.layer_type,
                    'forward_time_us': metrics.forward_time_us,
                    'memory_mb': metrics.total_memory_mb,
                    'torch_ops': [
                        {
                            'op_name': op.op_name,
                            'cpu_time_us': op.cpu_time_us,
                            'cuda_time_us': op.cuda_time_us,
                            'memory_delta': op.memory_delta,
                            'input_shapes': op.input_shapes,
                            'flops': op.flops
                        } for op in metrics.torch_ops
                    ]
                } for metrics in self.layer_metrics.values()
            ]
        }
        
        if self.gpu_monitor:
            gpu_history = self.gpu_monitor.get_metrics_history(60.0)  # 最近1分钟
            report['gpu_history'] = [
                {
                    'timestamp': m.timestamp,
                    'gpu_utilization': m.gpu_utilization,
                    'memory_utilization': m.memory_utilization,
                    'temperature': m.temperature,
                    'power_draw': m.power_draw
                } for m in gpu_history
            ]
            
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Detailed profiling report exported to {output_file}")


# 便捷的装饰器和上下文管理器
@contextmanager
def monitor_vllm_execution(model: nn.Module, 
                          output_dir: Optional[str] = None,
                          device_id: int = 0):
    """监控vLLM模型执行的上下文管理器"""
    profiler = LayerwiseProfiler(model, device_id=device_id)
    profiler.start_monitoring()
    
    try:
        with profiler.profile_forward_pass(output_dir) as prof:
            yield profiler
    finally:
        profiler.stop_monitoring()


def profile_vllm_layer(layer_name: str = None):
    """装饰器：监控特定层的执行"""
    def decorator(forward_func):
        def wrapper(*args, **kwargs):
            # 在这里添加层级监控逻辑
            start_time = time.time()
            
            # 记录内存使用
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # 执行原始函数
            result = forward_func(*args, **kwargs)
            
            # 记录性能指标
            end_time = time.time()
            execution_time = (end_time - start_time) * 1e6  # 转换为微秒
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_delta = (peak_memory - start_memory) / 1e6  # 转换为MB
                
                logger.debug(f"Layer {layer_name or forward_func.__name__}: "
                           f"{execution_time:.2f}μs, {memory_delta:.2f}MB")
            
            return result
        return wrapper
    return decorator
