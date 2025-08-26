#!/usr/bin/env python3
"""
Qwen3 真实模型性能测试 - 添加轻量级hook进行性能监控

在真实的Qwen3模型上添加hook来测量各层的性能表现
"""

import os
import sys
import time
import torch
import json
import threading
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import gc

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig, ModelConfig
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer, Qwen3Attention
    from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.attention import Attention
    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: vLLM not available: {e}")
    VLLM_AVAILABLE = False


@dataclass
class LayerTiming:
    """单层性能数据"""
    layer_name: str
    forward_time: float
    attention_time: float
    mlp_time: float
    layernorm_time: float
    memory_before: int
    memory_after: int
    input_shape: tuple
    timestamp: float


class LightweightProfiler:
    """轻量级性能profiler，最小化对模型性能的影响"""
    
    def __init__(self, enable_detailed=False):
        self.enable_detailed = enable_detailed
        self.enabled = False
        self.timings = deque(maxlen=10000)  # 限制内存使用
        self.layer_stats = defaultdict(list)
        self.lock = threading.Lock()
        
        # 性能计数器
        self.operation_counts = defaultdict(int)
        self.total_forward_time = 0.0
        self.total_attention_time = 0.0
        self.total_mlp_time = 0.0
        
    def enable(self):
        """启用profiling"""
        self.enabled = True
        self.start_time = time.perf_counter()
        
    def disable(self):
        """停用profiling"""
        self.enabled = False
        
    @contextmanager
    def timer(self, name: str):
        """轻量级计时器"""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            if self.enabled:
                end_time = time.perf_counter()
                duration = end_time - start_time
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                with self.lock:
                    self.timings.append({
                        'name': name,
                        'duration': duration,
                        'memory_delta': end_memory - start_memory,
                        'timestamp': end_time
                    })
                    self.operation_counts[name] += 1
                    
                    # 累计统计
                    if 'forward' in name.lower():
                        self.total_forward_time += duration
                    elif 'attention' in name.lower() or 'attn' in name.lower():
                        self.total_attention_time += duration
                    elif 'mlp' in name.lower():
                        self.total_mlp_time += duration
                        
    def add_layer_timing(self, timing: LayerTiming):
        """添加层级计时数据"""
        if not self.enabled:
            return
            
        with self.lock:
            self.layer_stats[timing.layer_name].append(timing)
            
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.timings:
            return {}
            
        durations = [t['duration'] for t in self.timings]
        
        return {
            'total_operations': len(self.timings),
            'total_time': sum(durations),
            'avg_time': np.mean(durations),
            'std_time': np.std(durations),
            'min_time': min(durations),
            'max_time': max(durations),
            'forward_time': self.total_forward_time,
            'attention_time': self.total_attention_time, 
            'mlp_time': self.total_mlp_time,
            'operation_counts': dict(self.operation_counts)
        }
        
    def clear(self):
        """清空统计数据"""
        with self.lock:
            self.timings.clear()
            self.layer_stats.clear()
            self.operation_counts.clear()
            self.total_forward_time = 0.0
            self.total_attention_time = 0.0
            self.total_mlp_time = 0.0


# 全局profiler实例
profiler = LightweightProfiler()


def add_qwen3_attention_hooks(attention_module, layer_name):
    """为Qwen3Attention添加性能hook"""
    
    original_forward = attention_module.forward
    
    def timed_forward(positions, hidden_states):
        with profiler.timer(f"{layer_name}_attention"):
            # QKV projection timing
            with profiler.timer(f"{layer_name}_qkv_proj"):
                qkv, _ = attention_module.qkv_proj(hidden_states)
                
            q, k, v = qkv.split([attention_module.q_size, attention_module.kv_size, attention_module.kv_size], dim=-1)
            
            # Q/K normalization
            with profiler.timer(f"{layer_name}_qk_norm"):
                q_by_head = q.view(*q.shape[:-1], q.shape[-1] // attention_module.head_dim, attention_module.head_dim)
                q_by_head = attention_module.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                
                k_by_head = k.view(*k.shape[:-1], k.shape[-1] // attention_module.head_dim, attention_module.head_dim)
                k_by_head = attention_module.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
                
            # RoPE
            with profiler.timer(f"{layer_name}_rope"):
                q, k = attention_module.rotary_emb(positions, q, k)
                
            # Attention computation
            with profiler.timer(f"{layer_name}_attention_compute"):
                attn_output = attention_module.attn(q, k, v)
                
            # Output projection
            with profiler.timer(f"{layer_name}_o_proj"):
                output, _ = attention_module.o_proj(attn_output)
                
            return output
    
    # 替换forward方法
    attention_module.forward = timed_forward
    return attention_module


def add_qwen3_decoder_hooks(decoder_layer, layer_name):
    """为Qwen3DecoderLayer添加性能hook"""
    
    # Hook attention
    add_qwen3_attention_hooks(decoder_layer.self_attn, f"{layer_name}.self_attn")
    
    # Hook MLP
    original_mlp_forward = decoder_layer.mlp.forward
    
    def timed_mlp_forward(x):
        with profiler.timer(f"{layer_name}_mlp"):
            return original_mlp_forward(x)
    
    decoder_layer.mlp.forward = timed_mlp_forward
    
    # Hook layer norms
    original_input_norm_forward = decoder_layer.input_layernorm.forward
    original_post_norm_forward = decoder_layer.post_attention_layernorm.forward
    
    def timed_input_norm_forward(*args, **kwargs):
        with profiler.timer(f"{layer_name}_input_layernorm"):
            return original_input_norm_forward(*args, **kwargs)
            
    def timed_post_norm_forward(*args, **kwargs):
        with profiler.timer(f"{layer_name}_post_attention_layernorm"):
            return original_post_norm_forward(*args, **kwargs)
    
    decoder_layer.input_layernorm.forward = timed_input_norm_forward
    decoder_layer.post_attention_layernorm.forward = timed_post_norm_forward
    
    # Hook整个layer的forward
    original_forward = decoder_layer.forward
    
    def timed_layer_forward(positions, hidden_states, residual=None):
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with profiler.timer(f"{layer_name}_forward"):
            result = original_forward(positions, hidden_states, residual)
            
        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 记录层级性能数据
        if profiler.enabled:
            layer_timing = LayerTiming(
                layer_name=layer_name,
                forward_time=end_time - start_time,
                attention_time=0.0,  # 会在子hook中累计
                mlp_time=0.0,       # 会在子hook中累计
                layernorm_time=0.0, # 会在子hook中累计
                memory_before=start_memory,
                memory_after=end_memory,
                input_shape=hidden_states.shape,
                timestamp=end_time
            )
            profiler.add_layer_timing(layer_timing)
            
        return result
    
    decoder_layer.forward = timed_layer_forward
    return decoder_layer


def instrument_qwen3_model(model):
    """为Qwen3模型添加性能监控hook"""
    print("Adding performance hooks to Qwen3 model...")
    
    hook_count = 0
    
    # 遍历所有模块，找到Qwen3DecoderLayer
    for name, module in model.named_modules():
        if isinstance(module, Qwen3DecoderLayer):
            print(f"  Adding hooks to {name}")
            add_qwen3_decoder_hooks(module, name)
            hook_count += 1
            
    print(f"Successfully added hooks to {hook_count} decoder layers")
    return hook_count


def run_performance_benchmark(model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
                            max_tokens: int = 100,
                            num_prompts: int = 5,
                            tp_size: int = 1):
    """运行性能基准测试"""
    
    if not VLLM_AVAILABLE:
        print("vLLM not available, cannot run benchmark")
        return
        
    print(f"Loading model: {model_path}")
    print(f"Configuration: TP={tp_size}, max_tokens={max_tokens}")
    
    try:
        # 初始化LLM
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=1024,  # 较短以减少内存需求
            enforce_eager=True,  # 禁用CUDA graph以便准确计时
            gpu_memory_utilization=0.85,
            trust_remote_code=True
        )
        
        # 获取底层模型实例
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        
        # 添加性能监控hook
        hook_count = instrument_qwen3_model(model)
        
        if hook_count == 0:
            print("Warning: No decoder layers found, hooks not added")
            return
            
        # 准备测试prompts
        prompts = [
            "The future of artificial intelligence is",
            "Explain quantum computing in simple terms:",
            "What are the main challenges in renewable energy?",
            "Describe the process of photosynthesis:",
            "How do neural networks learn patterns in data?"
        ][:num_prompts]
        
        sampling_params = SamplingParams(
            temperature=0.0,  # 确定性
            max_tokens=max_tokens,
            use_beam_search=False
        )
        
        print("\nRunning warmup...")
        # Warmup
        profiler.enable()
        for i in range(2):
            _ = llm.generate(prompts[0], sampling_params)
            profiler.clear()  # 清除warmup数据
        
        print("Starting performance measurement...")
        profiler.enable()
        
        results = []
        total_start = time.perf_counter()
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            
            prompt_start = time.perf_counter()
            outputs = llm.generate(prompt, sampling_params)
            prompt_end = time.perf_counter()
            
            # 记录结果
            output_text = outputs[0].outputs[0].text
            results.append({
                'prompt': prompt,
                'output': output_text,
                'prompt_time': prompt_end - prompt_start,
                'output_length': len(output_text.split())
            })
            
            # 定期清理GPU内存
            if i % 2 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        total_end = time.perf_counter()
        profiler.disable()
        
        # 分析结果
        total_time = total_end - total_start
        avg_prompt_time = np.mean([r['prompt_time'] for r in results])
        total_tokens = sum(r['output_length'] for r in results)
        throughput = total_tokens / total_time
        
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average prompt time: {avg_prompt_time:.3f} seconds") 
        print(f"Total output tokens: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tokens/second")
        
        # Profiler摘要
        summary = profiler.get_summary()
        if summary:
            print(f"\nDetailed Operation Timing:")
            print(f"Total operations: {summary['total_operations']}")
            print(f"Total operation time: {summary['total_time']*1000:.2f} ms")
            print(f"Average operation time: {summary['avg_time']*1000:.3f} ms")
            print(f"Forward time: {summary['forward_time']*1000:.2f} ms") 
            print(f"Attention time: {summary['attention_time']*1000:.2f} ms")
            print(f"MLP time: {summary['mlp_time']*1000:.2f} ms")
            
            # 显示操作计数
            print(f"\nTop Operations by Count:")
            op_counts = summary.get('operation_counts', {})
            sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
            for op, count in sorted_ops[:10]:
                print(f"  {op:40s}: {count:4d}")
        
        # 保存详细结果
        output_dir = "./qwen3_real_model_results"
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_results = {
            'benchmark_info': {
                'model_path': model_path,
                'tp_size': tp_size,
                'max_tokens': max_tokens,
                'num_prompts': num_prompts
            },
            'performance_metrics': {
                'total_time': total_time,
                'avg_prompt_time': avg_prompt_time,
                'total_tokens': total_tokens,
                'throughput': throughput
            },
            'profiler_summary': summary,
            'prompt_results': results
        }
        
        with open(f"{output_dir}/benchmark_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to {output_dir}/benchmark_results.json")
        
        return detailed_results
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def quick_test():
    """快速测试，用小模型验证hook功能"""
    print("Running quick test with small model...")
    
    # 使用一个更可能可用的小模型
    model_candidates = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct", 
        "microsoft/DialoGPT-small"  # 备选方案
    ]
    
    for model_path in model_candidates:
        try:
            print(f"Trying model: {model_path}")
            result = run_performance_benchmark(
                model_path=model_path,
                max_tokens=20,
                num_prompts=2,
                tp_size=1
            )
            
            if result:
                print("✅ Quick test completed successfully!")
                return result
                
        except Exception as e:
            print(f"Failed with {model_path}: {e}")
            continue
    
    print("❌ All model candidates failed")
    return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3 Real Model Performance Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Model path or name")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--num-prompts", type=int, default=5,
                       help="Number of test prompts")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with small model")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_performance_benchmark(
            model_path=args.model,
            max_tokens=args.max_tokens,
            num_prompts=args.num_prompts,
            tp_size=args.tp_size
        )


if __name__ == "__main__":
    main()
