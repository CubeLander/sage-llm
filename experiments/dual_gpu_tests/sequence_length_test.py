#!/usr/bin/env python3
"""
序列长度与推理时间关系测试
测试不同序列长度下的推理延迟和吞吐量
"""

import torch
import time
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# 添加vllm路径
sys.path.append('/home/tjy/hotLLM')

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
except ImportError:
    print("Warning: vLLM not available, using mock implementation")
    
    class MockLLM:
        def __init__(self, *args, **kwargs):
            self.model_name = kwargs.get('model', 'mock-model')
            
        def generate(self, prompts, sampling_params):
            # 模拟推理时间，与序列长度成正比
            time.sleep(len(prompts[0]) * 0.0001)  # 简单模拟
            return [f"Generated response for prompt"] * len(prompts)
    
    LLM = MockLLM
    SamplingParams = lambda **kwargs: kwargs

@dataclass
class TestResult:
    sequence_length: int
    inference_time: float
    tokens_per_second: float
    memory_usage: float
    gpu_utilization: float
    timestamp: str

class SequenceLengthTester:
    def __init__(self, model_name: str = "facebook/opt-125m", 
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1):
        """
        初始化序列长度测试器
        
        Args:
            model_name: 模型名称
            tensor_parallel_size: TP并行度
            pipeline_parallel_size: PP并行度
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.results = []
        
        # 检查GPU可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device_count = torch.cuda.device_count()
        print(f"Available GPUs: {self.device_count}")
        
        # 初始化模型
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                gpu_memory_utilization=0.8,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Model initialization failed: {e}")
            self.llm = LLM(model=model_name)  # fallback
    
    def generate_prompt(self, length: int) -> str:
        """生成指定长度的测试prompt"""
        base_prompt = "Please analyze the following text and provide insights: "
        # 使用重复的文本来达到目标长度
        filler = "This is a sample sentence for testing purposes. " * (length // 50 + 1)
        prompt = base_prompt + filler
        return prompt[:length]
    
    def measure_gpu_metrics(self) -> Dict[str, float]:
        """测量GPU指标"""
        metrics = {}
        
        for i in range(self.device_count):
            try:
                # 内存使用率
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                
                # GPU利用率（简单估算）
                torch.cuda.synchronize(i)
                
                metrics[f'gpu_{i}_memory_allocated'] = memory_allocated
                metrics[f'gpu_{i}_memory_reserved'] = memory_reserved
                
            except Exception as e:
                print(f"Error measuring GPU {i} metrics: {e}")
                
        return metrics
    
    def run_inference_test(self, sequence_length: int, num_samples: int = 5) -> TestResult:
        """运行单次推理测试"""
        print(f"\nTesting sequence length: {sequence_length}")
        
        # 生成测试prompt
        prompt = self.generate_prompt(sequence_length)
        actual_length = len(prompt.split())
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50  # 固定输出长度
        )
        
        # 预热
        for _ in range(2):
            try:
                _ = self.llm.generate([prompt[:100]], sampling_params)
            except:
                pass
        
        # 正式测试
        inference_times = []
        
        for i in range(num_samples):
            # 清空GPU缓存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 测量推理时间
            start_time = time.perf_counter()
            
            try:
                outputs = self.llm.generate([prompt], sampling_params)
                torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                print(f"  Sample {i+1}: {inference_time:.4f}s")
                
            except Exception as e:
                print(f"  Sample {i+1} failed: {e}")
                continue
        
        if not inference_times:
            raise RuntimeError("All inference attempts failed")
        
        # 计算统计信息
        avg_inference_time = np.mean(inference_times)
        tokens_per_second = actual_length / avg_inference_time if avg_inference_time > 0 else 0
        
        # 测量GPU指标
        gpu_metrics = self.measure_gpu_metrics()
        avg_memory = np.mean([v for k, v in gpu_metrics.items() if 'memory_allocated' in k])
        
        result = TestResult(
            sequence_length=actual_length,
            inference_time=avg_inference_time,
            tokens_per_second=tokens_per_second,
            memory_usage=avg_memory,
            gpu_utilization=0.0,  # 简化版本
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def run_sequence_scaling_test(self, 
                                min_length: int = 100, 
                                max_length: int = 4000, 
                                steps: int = 10) -> List[TestResult]:
        """运行序列长度扩展测试"""
        print(f"Running sequence scaling test from {min_length} to {max_length} tokens")
        print(f"Model: {self.model_name}")
        print(f"TP: {self.tensor_parallel_size}, PP: {self.pipeline_parallel_size}")
        
        # 生成测试序列长度
        lengths = np.logspace(
            np.log10(min_length), 
            np.log10(max_length), 
            steps,
            dtype=int
        )
        
        results = []
        
        for length in lengths:
            try:
                result = self.run_inference_test(length)
                results.append(result)
                
                print(f"Length: {result.sequence_length:4d}, "
                      f"Time: {result.inference_time:.4f}s, "
                      f"TPS: {result.tokens_per_second:.2f}")
                
            except Exception as e:
                print(f"Failed at length {length}: {e}")
                continue
        
        return results
    
    def save_results(self, output_dir: str = "results"):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sequence_length_test_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        results_dict = {
            'test_config': {
                'model_name': self.model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'pipeline_parallel_size': self.pipeline_parallel_size,
                'device_count': self.device_count
            },
            'results': [
                {
                    'sequence_length': r.sequence_length,
                    'inference_time': r.inference_time,
                    'tokens_per_second': r.tokens_per_second,
                    'memory_usage': r.memory_usage,
                    'gpu_utilization': r.gpu_utilization,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def cleanup(self):
        """清理资源"""
        try:
            destroy_model_parallel()
        except:
            pass
        
        torch.cuda.empty_cache()

def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequence Length Scaling Test")
    parser.add_argument("--model", default="facebook/opt-125m", 
                       help="Model name or path")
    parser.add_argument("--tp", type=int, default=1, 
                       help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, 
                       help="Pipeline parallel size")
    parser.add_argument("--min-length", type=int, default=100, 
                       help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=4000, 
                       help="Maximum sequence length")
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of test points")
    parser.add_argument("--output-dir", default="results", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # 运行测试
    tester = SequenceLengthTester(
        model_name=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp
    )
    
    try:
        results = tester.run_sequence_scaling_test(
            min_length=args.min_length,
            max_length=args.max_length,
            steps=args.steps
        )
        
        # 保存结果
        tester.save_results(args.output_dir)
        
        # 打印总结
        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests run: {len(results)}")
        if results:
            avg_tps = np.mean([r.tokens_per_second for r in results])
            print(f"Average tokens/second: {avg_tps:.2f}")
            
            print("\nScaling characteristics:")
            for i in range(1, len(results)):
                prev_result = results[i-1]
                curr_result = results[i]
                scaling_factor = (curr_result.sequence_length / prev_result.sequence_length)
                time_factor = (curr_result.inference_time / prev_result.inference_time)
                print(f"  Length {prev_result.sequence_length:4d} -> {curr_result.sequence_length:4d}: "
                      f"{scaling_factor:.2f}x length, {time_factor:.2f}x time")
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
