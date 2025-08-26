#!/usr/bin/env python3
"""
简化版Qwen3性能分析实验 - 使用本地小模型或mock模型进行演示

这个版本不依赖网络下载大模型，可以在离线环境中演示profiling框架的工作原理
"""

import torch
import torch.nn as nn
import time
import json
import os
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np

class MockQwen3Attention(nn.Module):
    """模拟Qwen3 Attention层"""
    
    def __init__(self, hidden_size=256, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # QKV projection (模拟QKVParallelLinear)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        # Output projection (模拟RowParallelLinear)  
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer norms (模拟RMSNorm)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)

class MockQwen3MLP(nn.Module):
    """模拟Qwen3 MLP层"""
    
    def __init__(self, hidden_size=256, intermediate_size=1024):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MockQwen3DecoderLayer(nn.Module):
    """模拟Qwen3 Decoder层"""
    
    def __init__(self, hidden_size=256):
        super().__init__()
        self.self_attn = MockQwen3Attention(hidden_size)
        self.mlp = MockQwen3MLP(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states):
        # Self attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states)
        hidden_states = residual + attn_output
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states

class MockQwen3Model(nn.Module):
    """模拟完整的Qwen3模型"""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MockQwen3DecoderLayer(hidden_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return self.norm(hidden_states)

class SimplifiedProfiler:
    """简化版的性能profiler"""
    
    def __init__(self, output_dir="./simplified_profile_results"):
        self.output_dir = output_dir
        self.enabled = False
        self.timings = []
        self.current_layer = ""
        
        os.makedirs(output_dir, exist_ok=True)
        
    def timer(self, name):
        """创建计时器上下文"""
        class TimerContext:
            def __init__(self, op_name, profiler):
                self.op_name = op_name
                self.profiler = profiler
                self.start_time = 0
                
            def __enter__(self):
                if self.profiler.enabled:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.profiler.enabled:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    duration = end_time - self.start_time
                    
                    self.profiler.timings.append({
                        'layer_name': self.profiler.current_layer,
                        'op_name': self.op_name,
                        'duration': duration,
                        'duration_ms': duration * 1000
                    })
                    
        return TimerContext(name, self)
        
    def patch_operations(self):
        """Patch torch操作进行计时"""
        # 保存原始函数
        self._orig_linear = torch.nn.functional.linear
        self._orig_softmax = torch.nn.functional.softmax
        self._orig_layer_norm = torch.nn.functional.layer_norm
        
        def timed_linear(input, weight, bias=None):
            with self.timer(f"linear_{list(weight.shape)}"):
                return self._orig_linear(input, weight, bias)
                
        def timed_softmax(input, dim=None, dtype=None):
            with self.timer("softmax"):
                return self._orig_softmax(input, dim, dtype)
                
        def timed_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
            with self.timer("layer_norm"):
                return self._orig_layer_norm(input, normalized_shape, weight, bias, eps)
        
        # Apply patches
        torch.nn.functional.linear = timed_linear
        torch.nn.functional.softmax = timed_softmax
        torch.nn.functional.layer_norm = timed_layer_norm
        
    def unpatch_operations(self):
        """恢复原始操作"""
        torch.nn.functional.linear = self._orig_linear
        torch.nn.functional.softmax = self._orig_softmax
        torch.nn.functional.layer_norm = self._orig_layer_norm
        
    def setup_hooks(self, model):
        """为模型设置profiling hooks"""
        self.hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, MockQwen3DecoderLayer):
                def make_layer_hook(layer_name):
                    def pre_hook(module, input):
                        self.current_layer = layer_name
                    return pre_hook
                    
                handle = module.register_forward_pre_hook(make_layer_hook(name))
                self.hooks.append(handle)
        
    def start_profiling(self):
        """开始profiling"""
        self.enabled = True
        self.timings.clear()
        self.patch_operations()
        
    def stop_profiling(self):
        """停止profiling"""
        self.enabled = False
        self.unpatch_operations()
        
        # 清理hooks
        for handle in self.hooks:
            handle.remove()
            
    def analyze_results(self):
        """分析profiling结果"""
        if not self.timings:
            return {}
            
        # 按操作类型聚合
        op_times = defaultdict(list)
        layer_times = defaultdict(list)
        
        for timing in self.timings:
            op_times[timing['op_name']].append(timing['duration'])
            layer_times[timing['layer_name']].append(timing['duration'])
            
        analysis = {
            'summary': {
                'total_operations': len(self.timings),
                'total_time': sum(t['duration'] for t in self.timings),
                'avg_time': np.mean([t['duration'] for t in self.timings])
            },
            'by_operation': {},
            'by_layer': {}
        }
        
        # 操作统计
        for op, times in op_times.items():
            analysis['by_operation'][op] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'percentage': sum(times) / analysis['summary']['total_time'] * 100
            }
            
        # 层统计
        for layer, times in layer_times.items():
            if layer:  # 忽略空层名
                analysis['by_layer'][layer] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times)
                }
        
        return analysis
        
    def save_results(self, analysis):
        """保存结果"""
        # 保存详细数据
        with open(os.path.join(self.output_dir, "detailed_timings.json"), 'w') as f:
            json.dump(self.timings, f, indent=2)
            
        # 保存分析
        with open(os.path.join(self.output_dir, "analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        print(f"Results saved to {self.output_dir}")
        
    def print_summary(self, analysis):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("SIMPLIFIED QWEN3 PROFILING RESULTS")
        print("="*60)
        
        summary = analysis['summary']
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Time: {summary['total_time']*1000:.3f} ms")
        print(f"Average Time: {summary['avg_time']*1000:.3f} ms")
        
        print(f"\nTop Operations by Time:")
        op_breakdown = analysis['by_operation']
        sorted_ops = sorted(op_breakdown.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for i, (op, stats) in enumerate(sorted_ops[:10]):
            print(f"{i+1:2d}. {op:30s} {stats['total_time']*1000:8.3f}ms ({stats['percentage']:5.1f}%)")
            
        if analysis['by_layer']:
            print(f"\nBy Layer:")
            layer_breakdown = analysis['by_layer']
            sorted_layers = sorted(layer_breakdown.items(), key=lambda x: x[1]['total_time'], reverse=True)
            
            for layer, stats in sorted_layers[:5]:
                print(f"  {layer:40s} {stats['total_time']*1000:8.3f}ms")
        
        print("="*60)

def run_simplified_experiment():
    """运行简化实验"""
    print("Starting Simplified Qwen3 Profiling Experiment")
    print("Using mock model to demonstrate profiling capabilities")
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 创建模型
    model = MockQwen3Model(vocab_size=1000, hidden_size=256, num_layers=4).to(device)
    model.eval()
    
    # 创建profiler
    profiler = SimplifiedProfiler()
    
    # 设置hooks
    profiler.setup_hooks(model)
    
    # 准备输入数据
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"Input shape: {input_ids.shape}")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)
    
    # 开始profiling
    print("Starting profiling...")
    profiler.start_profiling()
    
    try:
        with torch.no_grad():
            for step in range(5):
                profiler.current_layer = f"forward_step_{step}"
                output = model(input_ids)
                
                # 模拟一些额外操作
                with profiler.timer("additional_computation"):
                    result = torch.nn.functional.softmax(output, dim=-1)
                    loss = torch.mean(result)
                    
    finally:
        profiler.stop_profiling()
    
    # 分析结果
    print("Analyzing results...")
    analysis = profiler.analyze_results()
    
    # 保存和显示结果
    profiler.save_results(analysis)
    profiler.print_summary(analysis)
    
    return analysis

if __name__ == "__main__":
    try:
        analysis = run_simplified_experiment()
        print("\n✅ Simplified experiment completed successfully!")
        print("This demonstrates the profiling framework capabilities.")
        print("For full Qwen3 analysis, use the complete experiment scripts with real models.")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
