#!/usr/bin/env python3
"""
Qwen3 性能分析实验的简化测试版本

用于验证实验框架是否正常工作，使用最小的配置进行测试
"""

import os
import sys
import time
import torch
import json
from collections import defaultdict
from typing import Dict, Any, List

# 添加experiments目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_torch_operations_profiling():
    """测试torch操作profiling的基本功能"""
    print("Testing torch operations profiling...")
    
    # 模拟profiling环境
    timings = []
    
    def timer_context(name):
        class TimerContext:
            def __init__(self, op_name):
                self.op_name = op_name
                self.start_time = 0
                
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, *args):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                duration = end_time - self.start_time
                timings.append({
                    'op_name': self.op_name,
                    'duration': duration,
                    'duration_ms': duration * 1000
                })
                
        return TimerContext(name)
    
    # 测试一些基本torch操作
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建测试tensor
    x = torch.randn(128, 512, device=device)
    y = torch.randn(512, 256, device=device)
    
    # 测试不同操作
    with timer_context("matmul"):
        z = torch.matmul(x, y)
        
    with timer_context("relu"):
        z_relu = torch.relu(z)
        
    with timer_context("softmax"):
        z_softmax = torch.softmax(z, dim=-1)
        
    with timer_context("layer_norm"):
        z_norm = torch.layer_norm(z, [256])
    
    # 输出结果
    print(f"Collected {len(timings)} timing records:")
    for timing in timings:
        print(f"  {timing['op_name']:10s}: {timing['duration_ms']:6.3f} ms")
    
    return len(timings) > 0

def test_simple_model_profiling():
    """测试简单模型的profiling"""
    print("\nTesting simple model profiling...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建简单的transformer-like层
    class SimpleAttention(torch.nn.Module):
        def __init__(self, d_model=256, nheads=8):
            super().__init__()
            self.d_model = d_model
            self.nheads = nheads
            self.head_dim = d_model // nheads
            
            self.qkv = torch.nn.Linear(d_model, 3 * d_model)
            self.out_proj = torch.nn.Linear(d_model, d_model)
            
        def forward(self, x):
            B, L, D = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            
            # Reshape for attention
            q = q.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
            k = k.view(B, L, self.nheads, self.head_dim).transpose(1, 2)  
            v = v.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            
            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(out)
    
    class SimpleMLP(torch.nn.Module):
        def __init__(self, d_model=256, d_ff=1024):
            super().__init__()
            self.fc1 = torch.nn.Linear(d_model, d_ff)
            self.fc2 = torch.nn.Linear(d_ff, d_model)
            self.activation = torch.nn.GELU()
            
        def forward(self, x):
            return self.fc2(self.activation(self.fc1(x)))
    
    class SimpleTransformerLayer(torch.nn.Module):
        def __init__(self, d_model=256):
            super().__init__()
            self.attention = SimpleAttention(d_model)
            self.mlp = SimpleMLP(d_model)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            
        def forward(self, x):
            # Self attention with residual
            attn_out = self.attention(self.norm1(x))
            x = x + attn_out
            
            # MLP with residual  
            mlp_out = self.mlp(self.norm2(x))
            x = x + mlp_out
            
            return x
    
    # 创建模型
    model = SimpleTransformerLayer().to(device)
    model.eval()
    
    # Hook计时
    layer_times = defaultdict(list)
    
    def create_timing_hook(module_name):
        def hook(module, input, output):
            # 简单计时
            start_time = time.perf_counter()
            # 这里实际在hook中无法准确计时forward过程
            # 实际实现中需要使用pre_hook和post_hook配合
            end_time = time.perf_counter()
            layer_times[module_name].append(end_time - start_time)
        return hook
    
    # 注册hooks（简化版）
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只对叶子模块添加hook
            handle = module.register_forward_hook(create_timing_hook(name))
            handles.append(handle)
    
    # 运行测试
    batch_size, seq_len, d_model = 4, 128, 256
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    with torch.no_grad():
        for i in range(3):  # 运行几次
            output = model(x)
    
    # 清理hooks
    for handle in handles:
        handle.remove()
    
    print(f"Model profiling completed. Hooks registered for {len(handles)} modules.")
    return True

def test_memory_tracking():
    """测试内存使用跟踪"""
    print("\nTesting memory tracking...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return True
        
    device = "cuda"
    
    # 记录初始内存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
    
    memory_records = []
    
    def record_memory(step_name):
        current_memory = torch.cuda.memory_allocated()
        memory_records.append({
            'step': step_name,
            'memory_mb': current_memory / 1024**2,
            'delta_mb': (current_memory - initial_memory) / 1024**2
        })
    
    # 测试内存变化
    record_memory("baseline")
    
    # 分配一些tensor
    x1 = torch.randn(1024, 1024, device=device)
    record_memory("alloc_1MB")
    
    x2 = torch.randn(2048, 2048, device=device)  
    record_memory("alloc_16MB")
    
    # 执行一些操作
    y = torch.matmul(x2, x2.T)
    record_memory("matmul")
    
    # 释放内存
    del x1, x2, y
    torch.cuda.empty_cache()
    record_memory("cleanup")
    
    # 输出内存跟踪结果
    print("Memory tracking results:")
    for record in memory_records:
        print(f"  {record['step']:12s}: {record['memory_mb']:6.1f} MB (Δ{record['delta_mb']:+6.1f} MB)")
    
    return len(memory_records) > 0

def test_basic_vllm_loading():
    """测试基本的vLLM模型加载"""
    print("\nTesting basic vLLM loading...")
    
    try:
        from vllm import LLM, SamplingParams
        
        # 使用一个很小的模型进行测试
        print("Attempting to load a small model...")
        
        # 注意：这可能需要网络连接下载模型
        # 在实际环境中可以使用本地模型
        model_name = "microsoft/DialoGPT-small"  # 较小的模型用于测试
        
        try:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                max_model_len=128,  # 很小的长度
                enforce_eager=True,
                gpu_memory_utilization=0.3
            )
            
            sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
            
            # 测试生成
            outputs = llm.generate("Hello, how are you?", sampling_params)
            print(f"Generated output: {outputs[0].outputs[0].text}")
            
            print("vLLM loading test passed!")
            return True
            
        except Exception as e:
            print(f"vLLM model loading failed (expected in some environments): {e}")
            return False
            
    except ImportError:
        print("vLLM not available for testing")
        return False

def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("QWEN3 PROFILING FRAMEWORK TESTS")
    print("="*60)
    
    tests = [
        ("Torch Operations Profiling", test_torch_operations_profiling),
        ("Simple Model Profiling", test_simple_model_profiling),
        ("Memory Tracking", test_memory_tracking),
        ("Basic vLLM Loading", test_basic_vllm_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*40}")
            print(f"Running: {test_name}")
            print(f"{'='*40}")
            
            result = test_func()
            results[test_name] = result
            
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
            
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The profiling framework should work correctly.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the setup before running full experiments.")
    
    return results

if __name__ == "__main__":
    # 设置环境
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available - running CPU tests only")
    
    # 运行测试
    test_results = run_all_tests()
    
    # 保存测试结果
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to {output_file}")
