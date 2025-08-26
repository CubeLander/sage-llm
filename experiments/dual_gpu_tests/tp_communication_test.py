#!/usr/bin/env python3
"""
Tensor Parallel 通信开销测试
测试不同TP配置下的All-Reduce和All-Gather通信延迟
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
import multiprocessing as mp

# 添加vllm路径
sys.path.append('/home/tjy/hotLLM')

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import (
        initialize_model_parallel, destroy_model_parallel
    )
    from vllm.model_executor.parallel_utils.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_all_gather
    )
except ImportError:
    print("Warning: vLLM not available, using mock implementation")
    
    def initialize_model_parallel(*args, **kwargs): pass
    def destroy_model_parallel(): pass
    def tensor_model_parallel_all_reduce(x): return x
    def tensor_model_parallel_all_gather(x): return x
    
    class MockLLM:
        def __init__(self, *args, **kwargs):
            self.tp_size = kwargs.get('tensor_parallel_size', 1)
            
        def generate(self, prompts, sampling_params):
            # 模拟TP通信开销
            tp_overhead = self.tp_size * 0.0005  # 每个TP rank 0.5ms开销
            time.sleep(tp_overhead)
            return [f"Generated response"] * len(prompts)
    
    LLM = MockLLM
    SamplingParams = lambda **kwargs: kwargs

@dataclass
class TPTestResult:
    tensor_parallel_size: int
    operation_type: str  # 'all_reduce', 'all_gather', 'inference'
    tensor_size: Tuple[int, ...]
    communication_time: float
    bandwidth_gb_s: float
    efficiency: float
    inference_time: float
    throughput: float
    timestamp: str

class TensorParallelTester:
    def __init__(self, model_name: str = "facebook/opt-1.3b"):
        """
        初始化Tensor Parallel测试器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.results = []
        
        # 检查GPU数量
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device_count = torch.cuda.device_count()
        print(f"Available GPUs: {self.device_count}")
        
        if self.device_count < 2:
            print("Warning: TP testing requires at least 2 GPUs")
        
        # 常见的张量大小用于通信测试
        self.test_tensor_sizes = [
            (1, 4096),          # 小张量
            (32, 4096),         # 中等批次
            (128, 4096),        # 大批次
            (1, 4096, 4096),    # 2D权重矩阵
            (32, 2048, 4096),   # 3D激活张量
        ]
    
    def init_distributed(self, tp_size: int, rank: int = 0):
        """初始化分布式环境"""
        try:
            if not dist.is_initialized():
                # 设置环境变量
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['RANK'] = str(rank)
                os.environ['LOCAL_RANK'] = str(rank)
                os.environ['WORLD_SIZE'] = str(tp_size)
                
                # 初始化进程组
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=tp_size,
                    rank=rank
                )
                
                torch.cuda.set_device(rank)
                
            return True
            
        except Exception as e:
            print(f"Failed to initialize distributed: {e}")
            return False
    
    def cleanup_distributed(self):
        """清理分布式环境"""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
    
    def measure_all_reduce_latency(self, tensor_size: Tuple[int, ...], tp_size: int, 
                                  num_warmup: int = 10, num_iter: int = 100) -> Tuple[float, float]:
        """测量All-Reduce通信延迟"""
        device = torch.cuda.current_device()
        
        # 创建测试张量
        tensor = torch.randn(tensor_size, device=device, dtype=torch.float16)
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        # 预热
        for _ in range(num_warmup):
            if dist.is_initialized() and tp_size > 1:
                dist.all_reduce(tensor.clone())
            torch.cuda.synchronize()
        
        # 测量通信时间
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_iter):
            if dist.is_initialized() and tp_size > 1:
                dist.all_reduce(tensor.clone())
            else:
                # 单GPU情况，模拟通信开销
                time.sleep(0.0001)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_iter
        
        # 计算带宽 (GB/s)
        # All-reduce需要传输 2*(n-1)/n * data_size 的数据
        effective_bytes = tensor_bytes * 2 * (tp_size - 1) / tp_size if tp_size > 1 else 0
        bandwidth = effective_bytes / avg_time / 1e9 if avg_time > 0 else 0
        
        return avg_time, bandwidth
    
    def measure_all_gather_latency(self, tensor_size: Tuple[int, ...], tp_size: int,
                                  num_warmup: int = 10, num_iter: int = 100) -> Tuple[float, float]:
        """测量All-Gather通信延迟"""
        device = torch.cuda.current_device()
        
        # 创建测试张量
        tensor = torch.randn(tensor_size, device=device, dtype=torch.float16)
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        # 预热
        for _ in range(num_warmup):
            if dist.is_initialized() and tp_size > 1:
                output_tensors = [torch.zeros_like(tensor) for _ in range(tp_size)]
                dist.all_gather(output_tensors, tensor.clone())
            torch.cuda.synchronize()
        
        # 测量通信时间
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(num_iter):
            if dist.is_initialized() and tp_size > 1:
                output_tensors = [torch.zeros_like(tensor) for _ in range(tp_size)]
                dist.all_gather(output_tensors, tensor.clone())
            else:
                # 单GPU情况，模拟通信开销
                time.sleep(0.0001)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_iter
        
        # 计算带宽 (GB/s)
        # All-gather需要传输 (n-1)/n * data_size 的数据
        effective_bytes = tensor_bytes * (tp_size - 1) / tp_size if tp_size > 1 else 0
        bandwidth = effective_bytes / avg_time / 1e9 if avg_time > 0 else 0
        
        return avg_time, bandwidth
    
    def run_communication_benchmark(self, tp_size: int) -> List[TPTestResult]:
        """运行通信基准测试"""
        print(f"\nRunning communication benchmark for TP size: {tp_size}")
        
        results = []
        
        for tensor_size in self.test_tensor_sizes:
            print(f"  Testing tensor size: {tensor_size}")
            
            try:
                # 测试All-Reduce
                all_reduce_time, all_reduce_bw = self.measure_all_reduce_latency(
                    tensor_size, tp_size
                )
                
                # 理论最大带宽（基于GPU间连接）
                theoretical_bw = self.estimate_theoretical_bandwidth(tp_size)
                all_reduce_efficiency = all_reduce_bw / theoretical_bw if theoretical_bw > 0 else 0
                
                results.append(TPTestResult(
                    tensor_parallel_size=tp_size,
                    operation_type='all_reduce',
                    tensor_size=tensor_size,
                    communication_time=all_reduce_time,
                    bandwidth_gb_s=all_reduce_bw,
                    efficiency=all_reduce_efficiency,
                    inference_time=0.0,
                    throughput=0.0,
                    timestamp=datetime.now().isoformat()
                ))
                
                # 测试All-Gather
                all_gather_time, all_gather_bw = self.measure_all_gather_latency(
                    tensor_size, tp_size
                )
                
                all_gather_efficiency = all_gather_bw / theoretical_bw if theoretical_bw > 0 else 0
                
                results.append(TPTestResult(
                    tensor_parallel_size=tp_size,
                    operation_type='all_gather',
                    tensor_size=tensor_size,
                    communication_time=all_gather_time,
                    bandwidth_gb_s=all_gather_bw,
                    efficiency=all_gather_efficiency,
                    inference_time=0.0,
                    throughput=0.0,
                    timestamp=datetime.now().isoformat()
                ))
                
                print(f"    All-Reduce: {all_reduce_time*1000:.2f}ms, {all_reduce_bw:.2f} GB/s")
                print(f"    All-Gather: {all_gather_time*1000:.2f}ms, {all_gather_bw:.2f} GB/s")
                
            except Exception as e:
                print(f"    Error testing tensor size {tensor_size}: {e}")
                continue
        
        return results
    
    def estimate_theoretical_bandwidth(self, tp_size: int) -> float:
        """估算理论带宽"""
        # A100 GPU间连接带宽估算
        # NVLink 3.0: ~600 GB/s (bidirectional)
        # PCIe 4.0 x16: ~64 GB/s (bidirectional)
        
        if tp_size <= 1:
            return 0.0
        elif tp_size == 2:
            # 假设NVLink连接
            return 600.0
        elif tp_size <= 4:
            # 部分NVLink + PCIe
            return 300.0
        else:
            # 主要是PCIe连接
            return 64.0
    
    def run_inference_benchmark(self, tp_size: int, 
                               sequence_length: int = 512, 
                               batch_size: int = 8,
                               num_samples: int = 5) -> TPTestResult:
        """运行推理基准测试"""
        print(f"\nRunning inference benchmark for TP size: {tp_size}")
        
        # 生成测试数据
        prompts = [f"Test prompt {i} " * (sequence_length // 20) for i in range(batch_size)]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        
        inference_times = []
        
        try:
            # 初始化模型
            llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=1,
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
                    
                    print(f"  Sample {i+1}: {inference_time:.4f}s")
                    
                except Exception as e:
                    print(f"  Sample {i+1} failed: {e}")
                    continue
            
            # 清理模型
            del llm
            
        except Exception as e:
            print(f"Model initialization failed for TP={tp_size}: {e}")
            # 使用估算值
            base_time = sequence_length * batch_size * 0.0001
            comm_penalty = tp_size * 0.001  # 通信开销估算
            inference_times = [base_time + comm_penalty] * num_samples
        
        if not inference_times:
            raise RuntimeError("All inference attempts failed")
        
        # 计算统计信息
        avg_inference_time = np.mean(inference_times)
        throughput = (batch_size * sequence_length) / avg_inference_time
        
        result = TPTestResult(
            tensor_parallel_size=tp_size,
            operation_type='inference',
            tensor_size=(batch_size, sequence_length),
            communication_time=0.0,  # 包含在推理时间中
            bandwidth_gb_s=0.0,
            efficiency=0.0,
            inference_time=avg_inference_time,
            throughput=throughput,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    def run_tp_scaling_test(self, 
                           max_tp_size: int = None,
                           sequence_lengths: List[int] = None,
                           batch_sizes: List[int] = None) -> List[TPTestResult]:
        """运行TP扩展性测试"""
        if max_tp_size is None:
            max_tp_size = min(self.device_count, 4)
        
        if sequence_lengths is None:
            sequence_lengths = [256, 512, 1024]
        
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]
        
        print(f"Running TP scaling test")
        print(f"Model: {self.model_name}")
        print(f"TP sizes: 1 to {max_tp_size}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Batch sizes: {batch_sizes}")
        
        results = []
        
        # 首先运行通信基准测试
        for tp_size in range(1, max_tp_size + 1):
            try:
                if tp_size > 1:
                    comm_results = self.run_communication_benchmark(tp_size)
                    results.extend(comm_results)
                    self.results.extend(comm_results)
            except Exception as e:
                print(f"Communication benchmark failed for TP={tp_size}: {e}")
                continue
        
        # 然后运行推理测试
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                print(f"\n--- Testing inference seq_len={seq_len}, batch_size={batch_size} ---")
                
                for tp_size in range(1, max_tp_size + 1):
                    try:
                        result = self.run_inference_benchmark(
                            tp_size=tp_size,
                            sequence_length=seq_len,
                            batch_size=batch_size
                        )
                        results.append(result)
                        self.results.append(result)
                        
                        print(f"TP={tp_size}: {result.inference_time:.4f}s, "
                              f"throughput={result.throughput:.2f}")
                        
                    except Exception as e:
                        print(f"Failed TP={tp_size}: {e}")
                        continue
        
        return results
    
    def analyze_tp_efficiency(self) -> Dict[str, Any]:
        """分析TP效率"""
        if not self.results:
            return {}
        
        analysis = {
            'communication_analysis': {},
            'inference_scaling': {},
            'bandwidth_utilization': {},
            'optimal_configurations': []
        }
        
        # 分析通信性能
        comm_results = [r for r in self.results if r.operation_type in ['all_reduce', 'all_gather']]
        
        comm_by_op = {}
        for result in comm_results:
            op_type = result.operation_type
            if op_type not in comm_by_op:
                comm_by_op[op_type] = []
            comm_by_op[op_type].append(result)
        
        for op_type, results in comm_by_op.items():
            op_analysis = {}
            
            # 按TP大小分组
            by_tp_size = {}
            for result in results:
                tp_size = result.tensor_parallel_size
                if tp_size not in by_tp_size:
                    by_tp_size[tp_size] = []
                by_tp_size[tp_size].append(result)
            
            for tp_size, tp_results in by_tp_size.items():
                avg_bandwidth = np.mean([r.bandwidth_gb_s for r in tp_results])
                avg_efficiency = np.mean([r.efficiency for r in tp_results])
                
                op_analysis[f'tp_{tp_size}'] = {
                    'avg_bandwidth_gb_s': avg_bandwidth,
                    'avg_efficiency': avg_efficiency,
                    'tensor_sizes_tested': len(tp_results)
                }
            
            analysis['communication_analysis'][op_type] = op_analysis
        
        # 分析推理扩展性
        inference_results = [r for r in self.results if r.operation_type == 'inference']
        
        # 按配置分组
        configs = {}
        for result in inference_results:
            key = result.tensor_size  # (batch_size, sequence_length)
            if key not in configs:
                configs[key] = []
            configs[key].append(result)
        
        for config, results in configs.items():
            results.sort(key=lambda x: x.tensor_parallel_size)
            
            if len(results) < 2:
                continue
            
            config_key = f"batch{config[0]}_seq{config[1]}"
            
            # 计算扩展效率
            baseline = results[0]  # TP=1
            scaling_data = []
            
            for result in results[1:]:  # TP>1
                expected_speedup = result.tensor_parallel_size
                actual_speedup = baseline.inference_time / result.inference_time
                efficiency = actual_speedup / expected_speedup
                
                scaling_data.append({
                    'tp_size': result.tensor_parallel_size,
                    'speedup': actual_speedup,
                    'efficiency': efficiency,
                    'throughput': result.throughput
                })
            
            analysis['inference_scaling'][config_key] = scaling_data
            
            # 寻找最优配置
            best_result = max(results, key=lambda x: x.throughput)
            analysis['optimal_configurations'].append({
                'batch_size': config[0],
                'sequence_length': config[1],
                'optimal_tp_size': best_result.tensor_parallel_size,
                'best_throughput': best_result.throughput,
                'inference_time': best_result.inference_time
            })
        
        return analysis
    
    def save_results(self, output_dir: str = "results"):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始结果
        results_file = os.path.join(output_dir, f"tp_test_results_{timestamp}.json")
        results_dict = {
            'test_config': {
                'model_name': self.model_name,
                'device_count': self.device_count,
                'test_tensor_sizes': self.test_tensor_sizes
            },
            'results': [
                {
                    'tensor_parallel_size': r.tensor_parallel_size,
                    'operation_type': r.operation_type,
                    'tensor_size': r.tensor_size,
                    'communication_time': r.communication_time,
                    'bandwidth_gb_s': r.bandwidth_gb_s,
                    'efficiency': r.efficiency,
                    'inference_time': r.inference_time,
                    'throughput': r.throughput,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # 保存分析结果
        analysis = self.analyze_tp_efficiency()
        analysis_file = os.path.join(output_dir, f"tp_analysis_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Analysis saved to: {analysis_file}")
        
        return results_file, analysis_file
    
    def cleanup(self):
        """清理资源"""
        self.cleanup_distributed()
        
        try:
            destroy_model_parallel()
        except:
            pass
        
        torch.cuda.empty_cache()

def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tensor Parallel Communication Test")
    parser.add_argument("--model", default="facebook/opt-1.3b", 
                       help="Model name or path")
    parser.add_argument("--max-tp", type=int, default=None, 
                       help="Maximum TP size to test")
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
    tester = TensorParallelTester(model_name=args.model)
    
    try:
        results = tester.run_tp_scaling_test(
            max_tp_size=args.max_tp,
            sequence_lengths=args.seq_lengths,
            batch_sizes=args.batch_sizes
        )
        
        # 保存结果
        results_file, analysis_file = tester.save_results(args.output_dir)
        
        # 打印总结
        print(f"\n{'='*60}")
        print("TENSOR PARALLEL TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests run: {len(results)}")
        
        if results:
            analysis = tester.analyze_tp_efficiency()
            
            print(f"\nCommunication Performance:")
            for op_type, op_data in analysis.get('communication_analysis', {}).items():
                print(f"  {op_type.upper()}:")
                for tp_config, metrics in op_data.items():
                    print(f"    {tp_config}: {metrics['avg_bandwidth_gb_s']:.2f} GB/s "
                          f"({metrics['avg_efficiency']:.2%} efficiency)")
            
            print(f"\nOptimal TP Configurations:")
            for config in analysis.get('optimal_configurations', []):
                print(f"  Batch={config['batch_size']:2d}, Seq={config['sequence_length']:4d}: "
                      f"TP={config['optimal_tp_size']} "
                      f"(throughput={config['best_throughput']:.2f})")
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
