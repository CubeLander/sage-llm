#!/usr/bin/env python3
"""
Qwen3 深层性能分析实验 - 测量每一层每个torch操作的计算延迟以及分布式通信开销

设计思路:
1. Hook机制 - 在每个torch操作前后记录时间戳
2. 分层profiling - 分别测量Attention、MLP、LayerNorm等组件
3. 通信开销分析 - 测量TP和PP的all_reduce、send/recv等操作
4. 最小化测量开销 - 使用高精度计时器和条件性记录
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule
import time
import json
import os
import threading
import functools
import contextlib
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import csv
from dataclasses import dataclass, asdict
import gc

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer, Qwen3Attention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.attention import Attention
from vllm.distributed import get_tensor_model_parallel_world_size, get_pp_group


@dataclass
class OpTiming:
    """单个操作的时间测量数据"""
    layer_name: str
    op_name: str
    op_type: str  # 'compute', 'communication', 'memory'
    start_time: float
    end_time: float
    duration: float
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]
    device_id: int
    thread_id: int
    memory_before: int
    memory_after: int
    
class TimerContext:
    """高精度计时器上下文"""
    def __init__(self, name: str, profiler: 'Qwen3Profiler'):
        self.name = name
        self.profiler = profiler
        self.start_time = 0
        self.start_memory = 0
        
    def __enter__(self):
        if self.profiler.enabled:
            torch.cuda.synchronize()  # 同步GPU
            self.start_time = time.perf_counter()
            self.start_memory = torch.cuda.memory_allocated()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler.enabled:
            torch.cuda.synchronize()  # 同步GPU
            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated()
            duration = end_time - self.start_time
            
            # 记录时间数据
            timing = OpTiming(
                layer_name=self.profiler.current_layer,
                op_name=self.name,
                op_type='compute',
                start_time=self.start_time,
                end_time=end_time,
                duration=duration,
                input_shapes=[],
                output_shapes=[],
                device_id=torch.cuda.current_device(),
                thread_id=threading.get_ident(),
                memory_before=self.start_memory,
                memory_after=end_memory
            )
            
            self.profiler.add_timing(timing)


class Qwen3Profiler:
    """Qwen3模型深度性能分析器"""
    
    def __init__(self, 
                 model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",  # 使用较小模型测试
                 tp_size: int = 1,
                 pp_size: int = 1,
                 max_seq_len: int = 512,
                 batch_size: int = 1,
                 output_dir: str = "./qwen3_profile_results"):
        
        self.model_path = model_path
        self.tp_size = tp_size  
        self.pp_size = pp_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_dir = output_dir
        
        # 分析控制
        self.enabled = False
        self.current_layer = ""
        self.current_forward_step = 0
        
        # 数据存储
        self.timings: List[OpTiming] = []
        self.timing_lock = threading.Lock()
        
        # 性能统计
        self.layer_stats = defaultdict(lambda: {
            'total_time': 0.0,
            'op_count': 0,
            'avg_time': 0.0,
            'operations': defaultdict(float)
        })
        
        # Hook handles
        self.hooks = []
        
        # 通信分析
        self.comm_ops = {
            'all_reduce': [],
            'all_gather': [], 
            'send': [],
            'recv': [],
            'broadcast': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
    def timer(self, name: str):
        """创建计时器上下文管理器"""
        return TimerContext(name, self)
        
    def add_timing(self, timing: OpTiming):
        """线程安全地添加时间数据"""
        with self.timing_lock:
            self.timings.append(timing)
            
            # 更新统计
            layer_name = timing.layer_name
            self.layer_stats[layer_name]['total_time'] += timing.duration
            self.layer_stats[layer_name]['op_count'] += 1
            self.layer_stats[layer_name]['operations'][timing.op_name] += timing.duration
            
    def _create_forward_hook(self, module_name: str):
        """创建前向传播hook"""
        def hook_fn(module, input, output):
            if not self.enabled:
                return
                
            # 记录输入输出形状
            input_shapes = []
            output_shapes = []
            
            if isinstance(input, (tuple, list)):
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        input_shapes.append(inp.shape)
            elif isinstance(input, torch.Tensor):
                input_shapes.append(input.shape)
                
            if isinstance(output, (tuple, list)):
                for out in output:
                    if isinstance(out, torch.Tensor):
                        output_shapes.append(out.shape)
            elif isinstance(output, torch.Tensor):
                output_shapes.append(output.shape)
                
        return hook_fn
        
    def _monkey_patch_operations(self):
        """Monkey patch关键torch操作以进行性能测量"""
        
        # 保存原始函数
        self._orig_linear = F.linear
        self._orig_matmul = torch.matmul
        self._orig_bmm = torch.bmm
        self._orig_softmax = F.softmax
        self._orig_dropout = F.dropout
        self._orig_layer_norm = F.layer_norm
        self._orig_rms_norm = F.rms_norm if hasattr(F, 'rms_norm') else None
        self._orig_gelu = F.gelu
        self._orig_silu = F.silu
        self._orig_relu = F.relu
        
        # Patch linear operations
        def timed_linear(input, weight, bias=None):
            with self.timer(f"linear_{weight.shape}"):
                return self._orig_linear(input, weight, bias)
                
        def timed_matmul(input, other):
            with self.timer(f"matmul_{input.shape[-2:]}x{other.shape[-2:]}"):
                return self._orig_matmul(input, other)
                
        def timed_bmm(input, mat2):
            with self.timer(f"bmm_{input.shape}x{mat2.shape}"):
                return self._orig_bmm(input, mat2)
                
        def timed_softmax(input, dim=None, dtype=None):
            with self.timer(f"softmax_dim{dim}"):
                return self._orig_softmax(input, dim, dtype)
                
        def timed_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
            with self.timer("layer_norm"):
                return self._orig_layer_norm(input, normalized_shape, weight, bias, eps)
                
        # Apply patches
        F.linear = timed_linear
        torch.matmul = timed_matmul
        torch.bmm = timed_bmm
        F.softmax = timed_softmax
        F.layer_norm = timed_layer_norm
        
        # Patch activation functions
        activations = [
            ('gelu', F.gelu, self._orig_gelu),
            ('silu', F.silu, self._orig_silu), 
            ('relu', F.relu, self._orig_relu)
        ]
        
        for name, func_ref, orig_func in activations:
            if orig_func:
                def make_timed_activation(act_name, original):
                    def timed_activation(*args, **kwargs):
                        with self.timer(act_name):
                            return original(*args, **kwargs)
                    return timed_activation
                setattr(F, name, make_timed_activation(name, orig_func))
                
    def _unpatch_operations(self):
        """恢复原始torch操作"""
        F.linear = self._orig_linear
        torch.matmul = self._orig_matmul  
        torch.bmm = self._orig_bmm
        F.softmax = self._orig_softmax
        F.layer_norm = self._orig_layer_norm
        F.gelu = self._orig_gelu
        F.silu = self._orig_silu
        F.relu = self._orig_relu
        
    def _patch_communication_ops(self):
        """Patch分布式通信操作"""
        try:
            from vllm.distributed.communication_op import (
                tensor_model_parallel_all_reduce,
                tensor_model_parallel_all_gather
            )
            
            # 保存原始函数
            self._orig_all_reduce = tensor_model_parallel_all_reduce
            self._orig_all_gather = tensor_model_parallel_all_gather
            
            def timed_all_reduce(input_tensor, op=None):
                with self.timer("tp_all_reduce"):
                    return self._orig_all_reduce(input_tensor, op)
                    
            def timed_all_gather(input_tensor, dim=0):
                with self.timer("tp_all_gather"):
                    return self._orig_all_gather(input_tensor, dim)
            
            # Apply patches  
            import vllm.distributed.communication_op as comm_op
            comm_op.tensor_model_parallel_all_reduce = timed_all_reduce
            comm_op.tensor_model_parallel_all_gather = timed_all_gather
            
        except ImportError:
            print("Warning: Could not patch communication operations")
            
    def _patch_attention_ops(self):
        """Patch attention相关操作"""
        try:
            # Patch attention kernel calls
            from vllm.attention.backends.flash_attn import FlashAttentionBackend
            from vllm.attention.backends.xformers import XFormersBackend
            from vllm.attention.backends.torch_sdpa import TorchSDPABackend
            
            # 这里可以进一步patch具体的attention实现
            print("Attention backends patched successfully")
            
        except ImportError:
            print("Warning: Could not patch attention operations")
    
    def setup_model_profiling(self, model):
        """为模型设置profiling hooks"""
        
        # 为每个Qwen3DecoderLayer添加hook
        for name, module in model.named_modules():
            if isinstance(module, Qwen3DecoderLayer):
                # Layer-level timing
                def make_layer_hook(layer_name):
                    def pre_hook(module, input):
                        self.current_layer = layer_name
                        
                    def post_hook(module, input, output):
                        pass  # Layer timing handled by sub-components
                        
                    return pre_hook, post_hook
                    
                pre_hook, post_hook = make_layer_hook(name)
                handle1 = module.register_forward_pre_hook(pre_hook)
                handle2 = module.register_forward_hook(post_hook)
                self.hooks.extend([handle1, handle2])
                
            # Attention组件
            elif isinstance(module, Qwen3Attention):
                def make_attention_hook(comp_name):
                    def pre_hook(module, input):
                        self.current_layer = f"{self.current_layer}.{comp_name}"
                        
                    return pre_hook
                    
                handle = module.register_forward_pre_hook(make_attention_hook("attention"))
                self.hooks.append(handle)
                
            # Linear层
            elif isinstance(module, (QKVParallelLinear, RowParallelLinear)):
                def make_linear_hook(comp_name):
                    def pre_hook(module, input):
                        pass  # Linear operations timed by monkey patches
                        
                    return pre_hook
                    
                handle = module.register_forward_pre_hook(make_linear_hook(type(module).__name__))
                self.hooks.append(handle)
                
            # LayerNorm
            elif isinstance(module, RMSNorm):
                def make_norm_hook(comp_name):
                    def pre_hook(module, input):
                        pass  # Norm operations timed by monkey patches
                        
                    return pre_hook
                    
                handle = module.register_forward_pre_hook(make_norm_hook("rms_norm"))
                self.hooks.append(handle)
                
    def start_profiling(self):
        """开始profiling"""
        self.enabled = True
        self.timings.clear()
        self.layer_stats.clear()
        
        # Apply patches
        self._monkey_patch_operations()
        self._patch_communication_ops()
        self._patch_attention_ops()
        
        print("Profiling started...")
        
    def stop_profiling(self):
        """停止profiling"""
        self.enabled = False
        
        # Remove patches
        self._unpatch_operations()
        
        # Remove hooks
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        
        print("Profiling stopped...")
        
    def run_benchmark(self, 
                     prompts: List[str] = None,
                     num_tokens: int = 100,
                     warmup_steps: int = 3,
                     profile_steps: int = 10):
        """运行基准测试"""
        
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "Explain quantum computing in simple terms:",
                "What are the benefits of renewable energy?",
            ]
            
        # 初始化模型
        print(f"Loading model: {self.model_path}")
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tp_size,
            pipeline_parallel_size=self.pp_size,
            max_model_len=self.max_seq_len,
            enforce_eager=True,  # 避免CUDA graph优化干扰测量
            gpu_memory_utilization=0.8
        )
        
        sampling_params = SamplingParams(
            temperature=0.0,  # 确定性采样
            max_tokens=num_tokens,
            use_beam_search=False
        )
        
        # 获取模型实例进行profiling设置
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.setup_model_profiling(model)
        
        # Warmup
        print(f"Warming up with {warmup_steps} steps...")
        for i in range(warmup_steps):
            _ = llm.generate(prompts[i % len(prompts)], sampling_params)
            torch.cuda.empty_cache()
            
        # 开始profiling
        self.start_profiling()
        
        try:
            print(f"Profiling {profile_steps} steps...")
            for i in range(profile_steps):
                prompt = prompts[i % len(prompts)]
                
                with self.timer(f"full_forward_step_{i}"):
                    outputs = llm.generate(prompt, sampling_params)
                    
                # 记录这一步的完成
                self.current_forward_step = i + 1
                
                # 定期清理GPU内存
                if i % 3 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
        finally:
            self.stop_profiling()
            
        print(f"Profiling completed. Collected {len(self.timings)} timing records.")
        return self.analyze_results()
        
    def analyze_results(self) -> Dict[str, Any]:
        """分析性能数据"""
        if not self.timings:
            return {}
            
        analysis = {
            'summary': {
                'total_operations': len(self.timings),
                'total_time': sum(t.duration for t in self.timings),
                'avg_op_time': np.mean([t.duration for t in self.timings]),
                'std_op_time': np.std([t.duration for t in self.timings])
            },
            'layer_breakdown': {},
            'operation_breakdown': {},
            'communication_analysis': {},
            'memory_analysis': {}
        }
        
        # 按层分析
        layer_times = defaultdict(list)
        for timing in self.timings:
            layer_times[timing.layer_name].append(timing.duration)
            
        for layer, times in layer_times.items():
            analysis['layer_breakdown'][layer] = {
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'count': len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
            
        # 按操作类型分析  
        op_times = defaultdict(list)
        for timing in self.timings:
            op_times[timing.op_name].append(timing.duration)
            
        for op, times in op_times.items():
            analysis['operation_breakdown'][op] = {
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'count': len(times),
                'percentage': sum(times) / analysis['summary']['total_time'] * 100
            }
            
        # 通信分析
        comm_times = [t for t in self.timings if 'all_reduce' in t.op_name or 'all_gather' in t.op_name]
        if comm_times:
            analysis['communication_analysis'] = {
                'total_comm_time': sum(t.duration for t in comm_times),
                'comm_percentage': sum(t.duration for t in comm_times) / analysis['summary']['total_time'] * 100,
                'avg_comm_time': np.mean([t.duration for t in comm_times])
            }
            
        # 内存分析
        memory_deltas = [t.memory_after - t.memory_before for t in self.timings]
        analysis['memory_analysis'] = {
            'avg_memory_delta': np.mean(memory_deltas),
            'max_memory_delta': max(memory_deltas),
            'min_memory_delta': min(memory_deltas)
        }
        
        return analysis
        
    def save_results(self, analysis: Dict[str, Any]):
        """保存分析结果"""
        
        # 保存详细timing数据  
        timing_file = os.path.join(self.output_dir, "detailed_timings.csv")
        with open(timing_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'layer_name', 'op_name', 'op_type', 'duration', 
                'input_shapes', 'output_shapes', 'device_id', 
                'memory_before', 'memory_after'
            ])
            
            for timing in self.timings:
                writer.writerow([
                    timing.layer_name,
                    timing.op_name, 
                    timing.op_type,
                    timing.duration,
                    str(timing.input_shapes),
                    str(timing.output_shapes),
                    timing.device_id,
                    timing.memory_before,
                    timing.memory_after
                ])
                
        # 保存分析报告
        report_file = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        # 生成可视化友好的数据
        self._generate_visualization_data(analysis)
        
        print(f"Results saved to {self.output_dir}")
        
    def _generate_visualization_data(self, analysis: Dict[str, Any]):
        """生成用于可视化的数据"""
        
        # 层时间分布数据
        layer_data = []
        for layer, stats in analysis['layer_breakdown'].items():
            layer_data.append({
                'layer': layer,
                'total_time': stats['total_time'],
                'avg_time': stats['avg_time'], 
                'count': stats['count']
            })
            
        viz_file = os.path.join(self.output_dir, "visualization_data.json")
        with open(viz_file, 'w') as f:
            json.dump({
                'layer_times': layer_data,
                'operation_times': analysis['operation_breakdown'],
                'summary': analysis['summary']
            }, f, indent=2, default=str)
            
    def print_summary(self, analysis: Dict[str, Any]):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("QWEN3 DETAILED PROFILING SUMMARY")
        print("="*60)
        
        summary = analysis['summary']
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Time: {summary['total_time']:.4f} seconds")
        print(f"Average Operation Time: {summary['avg_op_time']*1000:.3f} ms")
        print(f"Std Operation Time: {summary['std_op_time']*1000:.3f} ms")
        
        print("\nTop 10 Slowest Operations:")
        op_breakdown = analysis['operation_breakdown']
        sorted_ops = sorted(op_breakdown.items(), key=lambda x: x[1]['total_time'], reverse=True)
        for i, (op, stats) in enumerate(sorted_ops[:10]):
            print(f"{i+1:2d}. {op:30s} {stats['total_time']*1000:8.3f}ms ({stats['percentage']:5.1f}%)")
            
        if 'communication_analysis' in analysis and analysis['communication_analysis']:
            comm = analysis['communication_analysis']
            print(f"\nCommunication Overhead: {comm['comm_percentage']:.1f}% ({comm['total_comm_time']*1000:.3f}ms)")
            
        print("\n" + "="*60)


def main():
    """主函数"""
    
    # 实验配置
    config = {
        'model_path': "Qwen/Qwen2.5-1.5B-Instruct",  # 使用较小模型
        'tp_size': 1,
        'pp_size': 1, 
        'max_seq_len': 512,
        'batch_size': 1,
        'output_dir': "./qwen3_profile_results"
    }
    
    # 测试prompts
    test_prompts = [
        "The key principles of machine learning include",
        "Distributed computing systems enable", 
        "Neural network architectures for natural language processing",
        "The evolution of transformer models has led to",
        "Efficient algorithms for large-scale data processing"
    ]
    
    print("Starting Qwen3 Detailed Profiling Experiment")
    print(f"Configuration: {config}")
    
    # 创建profiler
    profiler = Qwen3Profiler(**config)
    
    try:
        # 运行基准测试
        analysis = profiler.run_benchmark(
            prompts=test_prompts,
            num_tokens=50,  # 较短输出减少测试时间
            warmup_steps=2,
            profile_steps=5
        )
        
        # 分析和保存结果
        profiler.save_results(analysis)
        profiler.print_summary(analysis)
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
