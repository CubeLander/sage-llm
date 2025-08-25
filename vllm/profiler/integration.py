# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM性能监控集成补丁

这个模块提供了将GPU性能监控无缝集成到现有vLLM代码中的方法，
无需修改核心vLLM代码，通过monkey patching的方式添加监控功能。

使用方法：
```python
from vllm.profiler.integration import enable_vllm_profiling, disable_vllm_profiling

# 启用监控
enable_vllm_profiling(
    enable_gpu_monitoring=True,
    enable_layer_profiling=True,
    output_dir="./profiling_results"
)

# 正常使用vLLM
from vllm import LLM
llm = LLM("facebook/opt-125m")
outputs = llm.generate(["Hello world"], sampling_params)

# 禁用监控并导出结果
disable_vllm_profiling()
```
"""

import functools
import json
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import torch

from vllm.logger import init_logger
from vllm.profiler.vllm_monitor import VLLMPerformanceMonitor

logger = init_logger(__name__)

# 全局监控状态
_global_monitors: Dict[str, VLLMPerformanceMonitor] = {}
_monitoring_enabled = False
_original_methods = {}
_profiling_config = {}
_lock = threading.Lock()


def enable_vllm_profiling(
    enable_gpu_monitoring: bool = True,
    enable_layer_profiling: bool = False,
    sample_interval: float = 0.1,
    output_dir: Optional[str] = None,
    auto_export_interval: Optional[float] = None
):
    """
    启用vLLM性能监控
    
    Args:
        enable_gpu_monitoring: 启用GPU硬件监控
        enable_layer_profiling: 启用详细的层级profiling
        sample_interval: GPU监控采样间隔(秒)
        output_dir: 结果输出目录
        auto_export_interval: 自动导出间隔(秒)，None表示不自动导出
    """
    global _monitoring_enabled, _profiling_config
    
    with _lock:
        if _monitoring_enabled:
            logger.warning("vLLM profiling is already enabled")
            return
            
        _profiling_config = {
            'enable_gpu_monitoring': enable_gpu_monitoring,
            'enable_layer_profiling': enable_layer_profiling,
            'sample_interval': sample_interval,
            'output_dir': output_dir,
            'auto_export_interval': auto_export_interval
        }
        
        # 创建输出目录
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
        # 应用monkey patches
        _patch_vllm_classes()
        
        _monitoring_enabled = True
        
        # 启动自动导出线程
        if auto_export_interval:
            _start_auto_export_thread(auto_export_interval)
            
        logger.info("vLLM performance monitoring enabled")


def disable_vllm_profiling(export_results: bool = True):
    """
    禁用vLLM性能监控并可选导出结果
    
    Args:
        export_results: 是否在禁用前导出监控结果
    """
    global _monitoring_enabled
    
    with _lock:
        if not _monitoring_enabled:
            logger.warning("vLLM profiling is not enabled")
            return
            
        # 导出结果
        if export_results:
            _export_all_results()
            
        # 停止所有监控器
        for monitor in _global_monitors.values():
            monitor.stop_monitoring()
            
        # 恢复原始方法
        _restore_original_methods()
        
        # 清理状态
        _global_monitors.clear()
        _monitoring_enabled = False
        
        logger.info("vLLM performance monitoring disabled")


def get_monitoring_summary() -> Dict[str, Any]:
    """获取当前监控摘要"""
    if not _monitoring_enabled:
        return {}
        
    summary = {
        'total_monitors': len(_global_monitors),
        'monitoring_enabled': _monitoring_enabled,
        'config': _profiling_config,
        'monitors': {}
    }
    
    for name, monitor in _global_monitors.items():
        monitor_summary = monitor.get_performance_summary()
        summary['monitors'][name] = monitor_summary
        
    return summary


def _patch_vllm_classes():
    """为vLLM关键类应用monkey patches"""
    
    # Patch GPUModelRunner.execute_model
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        _patch_gpu_model_runner(GPUModelRunner)
    except ImportError:
        logger.warning("Could not import GPUModelRunner from v1")
        
    try:
        from vllm.worker.model_runner import GPUModelRunnerBase
        _patch_gpu_model_runner_base(GPUModelRunnerBase)
    except ImportError:
        logger.warning("Could not import GPUModelRunnerBase")
    
    # Patch LLM class
    try:
        from vllm import LLM
        _patch_llm_class(LLM)
    except ImportError:
        logger.warning("Could not import LLM class")


def _patch_gpu_model_runner(cls):
    """为GPUModelRunner添加监控"""
    original_execute_model = cls.execute_model
    original_init = cls.__init__
    
    def monitored_init(self, *args, **kwargs):
        # 调用原始初始化
        original_init(self, *args, **kwargs)
        
        # 添加性能监控器
        monitor_name = f"GPUModelRunner_{id(self)}"
        monitor = VLLMPerformanceMonitor(
            model_runner=self,
            enable_realtime_monitoring=_profiling_config.get('enable_gpu_monitoring', True),
            enable_detailed_profiling=_profiling_config.get('enable_layer_profiling', False),
            sample_interval=_profiling_config.get('sample_interval', 0.1)
        )
        
        _global_monitors[monitor_name] = monitor
        self._performance_monitor = monitor
        
        # 启动监控
        monitor.start_monitoring()
        
    def monitored_execute_model(self, *args, **kwargs):
        # 获取scheduler_output信息
        scheduler_output = args[0] if args else None
        batch_size = None
        num_tokens = None
        
        if scheduler_output:
            if hasattr(scheduler_output, 'total_num_scheduled_tokens'):
                num_tokens = scheduler_output.total_num_scheduled_tokens
            if hasattr(scheduler_output, 'num_scheduled_tokens'):
                # 计算批量大小
                batch_size = len(scheduler_output.num_scheduled_tokens) if \
                    hasattr(scheduler_output.num_scheduled_tokens, '__len__') else 1
        
        # 使用监控上下文执行
        if hasattr(self, '_performance_monitor'):
            with self._performance_monitor.profile_inference_step(
                batch_size=batch_size,
                num_tokens=num_tokens
            ):
                return original_execute_model(self, *args, **kwargs)
        else:
            return original_execute_model(self, *args, **kwargs)
    
    # 应用patches
    cls.__init__ = monitored_init
    cls.execute_model = monitored_execute_model
    
    # 保存原始方法
    _original_methods[f'{cls.__name__}.__init__'] = original_init
    _original_methods[f'{cls.__name__}.execute_model'] = original_execute_model


def _patch_gpu_model_runner_base(cls):
    """为GPUModelRunnerBase添加监控"""
    # 类似于GPUModelRunner的实现，但适配不同的类结构
    if hasattr(cls, 'execute_model'):
        original_execute_model = cls.execute_model
        
        def monitored_execute_model(self, *args, **kwargs):
            monitor_name = f"GPUModelRunnerBase_{id(self)}"
            
            if monitor_name not in _global_monitors and _monitoring_enabled:
                # 为这个实例创建监控器
                monitor = VLLMPerformanceMonitor(
                    model_runner=self,
                    enable_realtime_monitoring=_profiling_config.get('enable_gpu_monitoring', True),
                    enable_detailed_profiling=_profiling_config.get('enable_layer_profiling', False)
                )
                _global_monitors[monitor_name] = monitor
                monitor.start_monitoring()
                
            if monitor_name in _global_monitors:
                with _global_monitors[monitor_name].profile_inference_step():
                    return original_execute_model(self, *args, **kwargs)
            else:
                return original_execute_model(self, *args, **kwargs)
                
        cls.execute_model = monitored_execute_model
        _original_methods[f'{cls.__name__}.execute_model'] = original_execute_model


def _patch_llm_class(cls):
    """为LLM类添加监控"""
    original_generate = cls.generate
    original_init = cls.__init__
    
    def monitored_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # 为LLM实例添加监控标识
        self._profiling_enabled = True
        
    def monitored_generate(self, prompts, *args, **kwargs):
        # 记录生成请求的信息
        batch_size = len(prompts) if isinstance(prompts, (list, tuple)) else 1
        
        # 获取采样参数
        sampling_params = None
        if args:
            sampling_params = args[0]
        elif 'sampling_params' in kwargs:
            sampling_params = kwargs['sampling_params']
            
        max_tokens = None
        if sampling_params and hasattr(sampling_params, 'max_tokens'):
            max_tokens = sampling_params.max_tokens
            
        # 使用全局监控器记录生成请求
        monitor_name = f"LLM_{id(self)}"
        if monitor_name not in _global_monitors and _monitoring_enabled:
            # 为LLM实例创建一个简单的监控器
            from vllm.profiler.gpu_monitor import GPUHardwareMonitor
            gpu_monitor = GPUHardwareMonitor(
                sample_interval=_profiling_config.get('sample_interval', 0.1)
            )
            gpu_monitor.start()
            _global_monitors[monitor_name] = gpu_monitor
        
        logger.debug(f"LLM.generate called with {batch_size} prompts, max_tokens={max_tokens}")
        
        # 执行原始生成方法
        return original_generate(self, prompts, *args, **kwargs)
    
    # 应用patches
    cls.__init__ = monitored_init  
    cls.generate = monitored_generate
    
    # 保存原始方法
    _original_methods[f'{cls.__name__}.__init__'] = original_init
    _original_methods[f'{cls.__name__}.generate'] = original_generate


def _restore_original_methods():
    """恢复所有被patch的原始方法"""
    for method_name, original_method in _original_methods.items():
        try:
            class_name, method_name = method_name.split('.', 1)
            
            # 根据类名获取类对象
            if class_name == 'GPUModelRunner':
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner
                cls = GPUModelRunner
            elif class_name == 'GPUModelRunnerBase':
                from vllm.worker.model_runner import GPUModelRunnerBase
                cls = GPUModelRunnerBase
            elif class_name == 'LLM':
                from vllm import LLM
                cls = LLM
            else:
                continue
                
            # 恢复原始方法
            setattr(cls, method_name, original_method)
            
        except Exception as e:
            logger.warning(f"Failed to restore method {method_name}: {e}")
    
    _original_methods.clear()


def _export_all_results():
    """导出所有监控结果"""
    if not _profiling_config.get('output_dir'):
        return
        
    output_dir = Path(_profiling_config['output_dir'])
    timestamp = int(time.time())
    
    # 导出全局摘要
    summary = get_monitoring_summary()
    summary_file = output_dir / f"global_monitoring_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    # 导出各个监控器的详细报告
    for name, monitor in _global_monitors.items():
        if hasattr(monitor, 'export_performance_report'):
            report_file = output_dir / f"monitor_{name}_{timestamp}.json"
            monitor.export_performance_report(report_file)
            
    logger.info(f"Monitoring results exported to {output_dir}")


def _start_auto_export_thread(interval: float):
    """启动自动导出线程"""
    def auto_export_loop():
        while _monitoring_enabled:
            time.sleep(interval)
            if _monitoring_enabled:  # 再次检查
                try:
                    _export_all_results()
                    logger.debug("Auto-exported monitoring results")
                except Exception as e:
                    logger.warning(f"Auto-export failed: {e}")
                    
    thread = threading.Thread(target=auto_export_loop, daemon=True)
    thread.start()


# 上下文管理器版本
class VLLMProfilingContext:
    """vLLM性能监控上下文管理器"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def __enter__(self):
        enable_vllm_profiling(**self.config)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        disable_vllm_profiling()
        
    def get_summary(self):
        return get_monitoring_summary()


# 装饰器版本
def profile_vllm_function(
    enable_gpu_monitoring: bool = True,
    enable_layer_profiling: bool = False,
    output_dir: Optional[str] = None
):
    """
    装饰器：为函数添加vLLM性能监控
    
    使用方法:
    @profile_vllm_function(enable_gpu_monitoring=True, output_dir="./results")
    def run_inference():
        llm = LLM("facebook/opt-125m")
        outputs = llm.generate(["Hello world"])
        return outputs
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with VLLMProfilingContext(
                enable_gpu_monitoring=enable_gpu_monitoring,
                enable_layer_profiling=enable_layer_profiling,
                output_dir=output_dir
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 命令行工具支持
def setup_cli_profiling():
    """为命令行工具设置性能监控"""
    import argparse
    import sys
    
    # 检查是否有profiling相关的命令行参数
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable vLLM performance profiling')
    parser.add_argument('--profiling-output-dir', type=str, 
                       default='./vllm_profiling_results',
                       help='Profiling output directory')
    parser.add_argument('--enable-gpu-monitoring', action='store_true',
                       help='Enable GPU hardware monitoring')
    parser.add_argument('--enable-layer-profiling', action='store_true', 
                       help='Enable detailed layer profiling')
    
    # 解析已知参数
    args, unknown = parser.parse_known_args()
    
    if args.enable_profiling:
        enable_vllm_profiling(
            enable_gpu_monitoring=args.enable_gpu_monitoring,
            enable_layer_profiling=args.enable_layer_profiling,
            output_dir=args.profiling_output_dir
        )
        
        # 设置退出时的清理
        import atexit
        atexit.register(lambda: disable_vllm_profiling(export_results=True))
        
        logger.info("CLI profiling enabled")


# 自动检测环境变量
if os.getenv('VLLM_ENABLE_PROFILING', '').lower() in ('1', 'true', 'yes'):
    enable_vllm_profiling(
        enable_gpu_monitoring=os.getenv('VLLM_ENABLE_GPU_MONITORING', '').lower() in ('1', 'true', 'yes'),
        enable_layer_profiling=os.getenv('VLLM_ENABLE_LAYER_PROFILING', '').lower() in ('1', 'true', 'yes'),
        output_dir=os.getenv('VLLM_PROFILING_OUTPUT_DIR', './vllm_profiling_results')
    )
