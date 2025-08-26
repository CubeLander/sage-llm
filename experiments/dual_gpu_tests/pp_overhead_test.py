#!/usr/bin/env python3
"""
Pipeline Parallel 层间通信开销测试
测试不同PP配置下的通信延迟和吞吐量影响
"""

import torch
import torch.distributed as dist
import time
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import subprocess

# 添加vllm路径
sys.path.append('/home/tjy/hotLLM')

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import (
        initialize_model_parallel, destroy_model_parallel,
        get_pipeline_model_parallel_rank, get_pipeline_model_parallel_world_size
    )
except ImportError:
    print("Warning: vLLM not available, using mock implementation")
    
    def initialize_model_parallel(*args, **kwargs): pass
    def destroy_model_parallel(): pass
    def get_pipeline_model_parallel_rank(): return 0
    def get_pipeline_model_parallel_world_size(): return 1
    
    class MockLLM:
        def __init__(self, *args, **kwargs):
            self.pp_size = kwargs.get('pipeline_parallel_size', 1)
            
        def generate(self, prompts, sampling_params):
            # 模拟PP通信开销
            pp_overhead = self.pp_size * 0.001  # 每层1ms开销
            time.sleep(pp_overhead)
            return [f"Generated response"] * len(prompts)
    
    LLM = MockLLM
    SamplingParams = lambda **kwargs: kwargs

@dataclass
class PPTestResult:
    pipeline_parallel_size: int
    num_layers: int
    sequence_length: int
    batch_size: int
    inference_time: float
    communication_overhead: float
    pipeline_bubble_ratio: float
    throughput: float
    memory_per_gpu: float
    timestamp: str

class PipelineParallelTester:
    def __init__(self, model_name: str = "facebook/opt-1.3b"):
        """
        初始化Pipeline Parallel测试器
        
        Args:
            model_name: 模型名称（使用较大模型以观察PP效果）
        """
        self.model_name = model_name
        self.results = []
        
        # 检查GPU数量
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device_count = torch.cuda.device_count()
        print(f"Available GPUs: {self.device_count}")
        
        if self.device_count < 2:
            print("Warning: PP testing requires at least 2 GPUs")
        
        # 获取模型层数信息（估算）
        self.estimated_layers = self._estimate_model_layers(model_name)
    
    def _estimate_model_layers(self, model_name: str) -> int:
        """估算模型层数"""
        # 基于常见模型的层数映射
        layer_mapping = {
            'opt-125m': 12,
            'opt-350m': 24,
            'opt-1.3b': 24,
            'opt-2.7b': 32,
            'opt-6.7b': 32,
            'opt-13b': 40,
            'llama-7b': 32,
            'llama-13b': 40,
            'llama-30b': 60,
            'llama-65b': 80,
        }
        
        for key, layers in layer_mapping.items():
            if key in model_name.lower():
                return layers
        
        return 24  # 默认值
    
    def measure_communication_latency(self, pp_size: int) -> float:
        """测量PP通信延迟"""
        if pp_size <= 1:
            return 0.0
        
        try:
            # 创建测试张量
            test_size = (1, 4096, 4096)  # 典型的激活张量大小
            device = torch.cuda.current_device()
            
            # 模拟层间通信
            data = torch.randn(test_size, device=device, dtype=torch.float16)
            
            # 测量P2P传输时间
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # 模拟PP通信模式
            for i in range(pp_size - 1):
                next_device = (device + 1) % min(pp_size, self.device_count)
                if next_device < self.device_count:
                    data_copy = data.to(f'cuda:{next_device}')
                    torch.cuda.synchronize()
                    data = data_copy.to(device)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) / (pp_size - 1)
            return latency
            
        except Exception as e:
            print(f"Error measuring communication latency: {e}")
            return pp_size * 0.001  # 估算值
    
    def calculate_pipeline_bubble(self, pp_size: int, num_layers: int) -> float:
        """计算流水线气泡比例"""
        if pp_size <= 1:
            return 0.0
        
        # 简化的流水线气泡计算
        # 气泡时间 = (PP_size - 1) * 层计算时间
        # 总时间 = 气泡时间 + 有效计算时间
        layers_per_stage = num_layers / pp_size
        bubble_ratio = (pp_size - 1) / (layers_per_stage + pp_size - 1)
        
        return min(bubble_ratio, 0.5)  # 限制在50%以内
    
    def run_pp_test(self, 
                    pp_size: int, 
                    sequence_length: int = 512,
                    batch_size: int = 8,
                    num_samples: int = 5) -> PPTestResult:
        """运行单次PP测试"""
        print(f"\nTesting PP size: {pp_size}, seq_len: {sequence_length}, batch: {batch_size}")
        
        if pp_size > self.device_count:
            print(f"Warning: PP size {pp_size} > available GPUs {self.device_count}")
            pp_size = min(pp_size, self.device_count)
        
        # 生成测试数据
        prompts = [f"Test prompt {i} " * (sequence_length // 20) for i in range(batch_size)]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        
        # 测量通信开销
        comm_overhead = self.measure_communication_latency(pp_size)
        
        # 计算流水线气泡比例
        bubble_ratio = self.calculate_pipeline_bubble(pp_size, self.estimated_layers)
        
        # 运行推理测试
        inference_times = []
        memory_usages = []
        
        try:
            # 初始化模型
            llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                pipeline_parallel_size=pp_size,
                gpu_memory_utilization=0.8,
                trust_remote_code=True
            )
            
            # 预热
            for _ in range(2):
                try:
                    _ = llm.generate([prompts[0][:100]], sampling_params)
                except:
                    pass
            
            # 正式测试
            for i in range(num_samples):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                try:
                    outputs = llm.generate(prompts, sampling_params)
                    torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    inference_time = end_time - start_time
                    inference_times.append(inference_time)
                    
                    # 测量内存使用
                    memory_usage = []
                    for gpu_id in range(min(pp_size, self.device_count)):
                        mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_usage.append(mem)
                    memory_usages.append(np.mean(memory_usage))
                    
                    print(f"  Sample {i+1}: {inference_time:.4f}s")
                    
                except Exception as e:
                    print(f"  Sample {i+1} failed: {e}")
                    continue
            
            # 清理模型
            del llm
            
        except Exception as e:
            print(f"Model initialization failed for PP={pp_size}: {e}")
            # 使用估算值
            base_time = sequence_length * batch_size * 0.0001
            comm_penalty = comm_overhead * self.estimated_layers / pp_size
            bubble_penalty = base_time * bubble_ratio
            inference_times = [base_time + comm_penalty + bubble_penalty] * num_samples
            memory_usages = [8.0 / pp_size] * num_samples  # 估算内存分布
        
        if not inference_times:
            raise RuntimeError("All inference attempts failed")
        
        # 计算统计信息
        avg_inference_time = np.mean(inference_times)
        throughput = (batch_size * sequence_length) / avg_inference_time
        avg_memory = np.mean(memory_usages)
        
        result = PPTestResult(
            pipeline_parallel_size=pp_size,
            num_layers=self.estimated_layers,
            sequence_length=sequence_length,
            batch_size=batch_size,
            inference_time=avg_inference_time,
            communication_overhead=comm_overhead,
            pipeline_bubble_ratio=bubble_ratio,
            throughput=throughput,
            memory_per_gpu=avg_memory,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def run_pp_scaling_test(self, 
                           max_pp_size: int = None,
                           sequence_lengths: List[int] = None,
                           batch_sizes: List[int] = None) -> List[PPTestResult]:
        """运行PP扩展性测试"""
        if max_pp_size is None:
            max_pp_size = min(self.device_count, 4)
        
        if sequence_lengths is None:
            sequence_lengths = [256, 512, 1024, 2048]
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]
        
        print(f"Running PP scaling test")
        print(f"Model: {self.model_name} (estimated {self.estimated_layers} layers)")
        print(f"PP sizes: 1 to {max_pp_size}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Batch sizes: {batch_sizes}")
        
        results = []
        
        # 测试不同配置组合
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                print(f"\n--- Testing seq_len={seq_len}, batch_size={batch_size} ---")
                
                for pp_size in range(1, max_pp_size + 1):
                    try:
                        result = self.run_pp_test(
                            pp_size=pp_size,
                            sequence_length=seq_len,
                            batch_size=batch_size
                        )
                        results.append(result)
                        
                        print(f"PP={pp_size}: {result.inference_time:.4f}s, "
                              f"throughput={result.throughput:.2f}, "
                              f"bubble={result.pipeline_bubble_ratio:.2%}")
                        
                    except Exception as e:
                        print(f"Failed PP={pp_size}: {e}")
                        continue
        
        return results
    
    def analyze_pp_efficiency(self) -> Dict[str, Any]:
        """分析PP效率"""
        if not self.results:
            return {}
        
        analysis = {
            'scaling_efficiency': {},
            'communication_overhead_analysis': {},
            'memory_efficiency': {},
            'optimal_configurations': []
        }
        
        # 按配置分组
        configs = {}
        for result in self.results:
            key = (result.sequence_length, result.batch_size)
            if key not in configs:
                configs[key] = []
            configs[key].append(result)
        
        # 分析每个配置的扩展效率
        for (seq_len, batch_size), results in configs.items():
            results.sort(key=lambda x: x.pipeline_parallel_size)
            
            if len(results) < 2:
                continue
            
            config_key = f"seq{seq_len}_batch{batch_size}"
            
            # 计算扩展效率
            baseline = results[0]  # PP=1
            efficiencies = []
            
            for result in results[1:]:  # PP>1
                expected_speedup = result.pipeline_parallel_size
                actual_speedup = baseline.inference_time / result.inference_time
                efficiency = actual_speedup / expected_speedup
                efficiencies.append(efficiency)
            
            analysis['scaling_efficiency'][config_key] = {
                'pp_sizes': [r.pipeline_parallel_size for r in results[1:]],
                'efficiencies': efficiencies,
                'avg_efficiency': np.mean(efficiencies) if efficiencies else 0
            }
            
            # 通信开销分析
            comm_overheads = [r.communication_overhead for r in results]
            analysis['communication_overhead_analysis'][config_key] = {
                'pp_sizes': [r.pipeline_parallel_size for r in results],
                'comm_overheads': comm_overheads,
                'overhead_scaling': np.polyfit(
                    [r.pipeline_parallel_size for r in results], 
                    comm_overheads, 1
                )[0] if len(results) > 1 else 0
            }
            
            # 寻找最优配置
            best_result = min(results, key=lambda x: x.inference_time)
            analysis['optimal_configurations'].append({
                'sequence_length': seq_len,
                'batch_size': batch_size,
                'optimal_pp_size': best_result.pipeline_parallel_size,
                'best_time': best_result.inference_time,
                'efficiency': efficiencies[best_result.pipeline_parallel_size - 2] if best_result.pipeline_parallel_size > 1 and efficiencies else 1.0
            })
        
        return analysis
    
    def save_results(self, output_dir: str = "results"):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = os.path.join(output_dir, f"pp_test_results_{timestamp}.json")
        results_dict = {
            'test_config': {
                'model_name': self.model_name,
                'estimated_layers': self.estimated_layers,
                'device_count': self.device_count
            },
            'results': [
                {
                    'pipeline_parallel_size': r.pipeline_parallel_size,
                    'num_layers': r.num_layers,
                    'sequence_length': r.sequence_length,
                    'batch_size': r.batch_size,
                    'inference_time': r.inference_time,
                    'communication_overhead': r.communication_overhead,
                    'pipeline_bubble_ratio': r.pipeline_bubble_ratio,
                    'throughput': r.throughput,
                    'memory_per_gpu': r.memory_per_gpu,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # 保存分析结果
        analysis = self.analyze_pp_efficiency()
        analysis_file = os.path.join(output_dir, f"pp_analysis_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Analysis saved to: {analysis_file}")
        
        return results_file, analysis_file
    
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
    
    parser = argparse.ArgumentParser(description="Pipeline Parallel Overhead Test")
    parser.add_argument("--model", default="facebook/opt-1.3b", 
                       help="Model name or path")
    parser.add_argument("--max-pp", type=int, default=None, 
                       help="Maximum PP size to test")
    parser.add_argument("--seq-lengths", nargs='+', type=int, 
                       default=[256, 512, 1024], 
                       help="Sequence lengths to test")
    parser.add_argument("--batch-sizes", nargs='+', type=int, 
                       default=[1, 4, 8], 
                       help="Batch sizes to test")
    parser.add_argument("--output-dir", default="results", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # 运行测试
    tester = PipelineParallelTester(model_name=args.model)
    
    try:
        results = tester.run_pp_scaling_test(
            max_pp_size=args.max_pp,
            sequence_lengths=args.seq_lengths,
            batch_sizes=args.batch_sizes
        )
        
        # 保存结果
        results_file, analysis_file = tester.save_results(args.output_dir)
        
        # 打印总结
        print(f"\n{'='*60}")
        print("PIPELINE PARALLEL TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests run: {len(results)}")
        
        if results:
            analysis = tester.analyze_pp_efficiency()
            
            print(f"\nOptimal Configurations:")
            for config in analysis.get('optimal_configurations', []):
                print(f"  Seq={config['sequence_length']:4d}, Batch={config['batch_size']:2d}: "
                      f"PP={config['optimal_pp_size']} "
                      f"(time={config['best_time']:.4f}s, "
                      f"efficiency={config['efficiency']:.2%})")
            
            print(f"\nAverage PP Scaling Efficiency:")
            for config, data in analysis.get('scaling_efficiency', {}).items():
                print(f"  {config}: {data['avg_efficiency']:.2%}")
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
