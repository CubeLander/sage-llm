#!/usr/bin/env python3
"""
Qwen3 分布式通信开销专项分析

专门测量TP(Tensor Parallel)和PP(Pipeline Parallel)的通信开销:
1. All-reduce operations (RowParallelLinear输出合并)  
2. All-gather operations (处理KV heads复制)
3. Pipeline send/recv (PP层间传输)
4. Memory copies和同步开销
"""

import torch
import torch.distributed as dist
import time
import json
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import threading
import functools
import os

class DistributedCommProfiler:
    """分布式通信profiler"""
    
    def __init__(self, output_dir: str = "./comm_profile_results"):
        self.output_dir = output_dir
        self.enabled = False
        self.comm_records = []
        self.lock = threading.Lock()
        
        # 通信模式统计
        self.comm_patterns = {
            'all_reduce': defaultdict(list),
            'all_gather': defaultdict(list), 
            'send': defaultdict(list),
            'recv': defaultdict(list),
            'broadcast': defaultdict(list),
            'synchronize': defaultdict(list)
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
    def record_comm_op(self, op_type: str, tensor_size: int, 
                      duration: float, metadata: Dict[str, Any] = None):
        """记录通信操作"""
        if not self.enabled:
            return
            
        with self.lock:
            record = {
                'timestamp': time.time(),
                'op_type': op_type,
                'tensor_size': tensor_size,
                'duration': duration,
                'metadata': metadata or {}
            }
            self.comm_records.append(record)
            self.comm_patterns[op_type]['sizes'].append(tensor_size)
            self.comm_patterns[op_type]['times'].append(duration)
            
    def start_profiling(self):
        """开始通信profiling"""
        self.enabled = True
        self.comm_records.clear()
        for pattern in self.comm_patterns.values():
            pattern.clear()
            
    def stop_profiling(self):
        """停止通信profiling"""
        self.enabled = False
        
    def patch_communication_ops(self):
        """Patch vLLM的通信操作"""
        try:
            # Patch tensor parallel operations
            import vllm.distributed.communication_op as comm_op
            
            # 保存原始函数
            self._orig_all_reduce = comm_op.tensor_model_parallel_all_reduce
            self._orig_all_gather = comm_op.tensor_model_parallel_all_gather
            
            def timed_all_reduce(tensor, op=None):
                if not self.enabled:
                    return self._orig_all_reduce(tensor, op)
                    
                start_time = time.perf_counter()
                torch.cuda.synchronize()
                
                result = self._orig_all_reduce(tensor, op)
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                self.record_comm_op(
                    'all_reduce',
                    tensor.numel() * tensor.element_size(),
                    end_time - start_time,
                    {
                        'tensor_shape': list(tensor.shape),
                        'tensor_dtype': str(tensor.dtype),
                        'device': str(tensor.device),
                        'op': str(op) if op else None
                    }
                )
                
                return result
                
            def timed_all_gather(tensor, dim=0):
                if not self.enabled:
                    return self._orig_all_gather(tensor, dim)
                    
                start_time = time.perf_counter()
                torch.cuda.synchronize()
                
                result = self._orig_all_gather(tensor, dim)
                
                torch.cuda.synchronize() 
                end_time = time.perf_counter()
                
                self.record_comm_op(
                    'all_gather',
                    tensor.numel() * tensor.element_size(),
                    end_time - start_time,
                    {
                        'tensor_shape': list(tensor.shape),
                        'tensor_dtype': str(tensor.dtype),
                        'device': str(tensor.device),
                        'dim': dim
                    }
                )
                
                return result
                
            # Apply patches
            comm_op.tensor_model_parallel_all_reduce = timed_all_reduce
            comm_op.tensor_model_parallel_all_gather = timed_all_gather
            
            print("Communication operations patched successfully")
            
        except ImportError as e:
            print(f"Warning: Could not patch communication operations: {e}")
            
    def patch_pipeline_ops(self):
        """Patch pipeline parallel operations"""
        try:
            from vllm.distributed.pipeline_parallel import pp_group
            
            if hasattr(pp_group, 'send') and hasattr(pp_group, 'recv'):
                # 保存原始函数
                self._orig_pp_send = pp_group.send
                self._orig_pp_recv = pp_group.recv
                
                def timed_pp_send(tensor, dst=None):
                    if not self.enabled:
                        return self._orig_pp_send(tensor, dst)
                        
                    start_time = time.perf_counter()
                    torch.cuda.synchronize()
                    
                    result = self._orig_pp_send(tensor, dst)
                    
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    
                    self.record_comm_op(
                        'send',
                        tensor.numel() * tensor.element_size(),
                        end_time - start_time,
                        {
                            'tensor_shape': list(tensor.shape),
                            'tensor_dtype': str(tensor.dtype),
                            'dst': dst
                        }
                    )
                    
                    return result
                    
                def timed_pp_recv(size, dtype, src=None):
                    if not self.enabled:
                        return self._orig_pp_recv(size, dtype, src)
                        
                    start_time = time.perf_counter()
                    torch.cuda.synchronize()
                    
                    result = self._orig_pp_recv(size, dtype, src)
                    
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    
                    tensor_size = size if isinstance(size, int) else np.prod(size) * torch.tensor([], dtype=dtype).element_size()
                    
                    self.record_comm_op(
                        'recv',
                        tensor_size,
                        end_time - start_time,
                        {
                            'size': size,
                            'dtype': str(dtype),
                            'src': src
                        }
                    )
                    
                    return result
                    
                # Apply patches
                pp_group.send = timed_pp_send
                pp_group.recv = timed_pp_recv
                
                print("Pipeline parallel operations patched successfully")
                
        except ImportError as e:
            print(f"Warning: Could not patch pipeline operations: {e}")
            
    def analyze_communication_patterns(self) -> Dict[str, Any]:
        """分析通信模式"""
        if not self.comm_records:
            return {}
            
        analysis = {
            'summary': {
                'total_comm_ops': len(self.comm_records),
                'total_comm_time': sum(r['duration'] for r in self.comm_records),
                'avg_comm_time': np.mean([r['duration'] for r in self.comm_records]),
                'total_data_transferred': sum(r['tensor_size'] for r in self.comm_records)
            },
            'by_operation_type': {},
            'bandwidth_analysis': {},
            'latency_analysis': {},
            'scaling_analysis': {}
        }
        
        # 按操作类型分析
        op_groups = defaultdict(list)
        for record in self.comm_records:
            op_groups[record['op_type']].append(record)
            
        for op_type, records in op_groups.items():
            durations = [r['duration'] for r in records]
            sizes = [r['tensor_size'] for r in records]
            
            analysis['by_operation_type'][op_type] = {
                'count': len(records),
                'total_time': sum(durations),
                'avg_time': np.mean(durations),
                'std_time': np.std(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'total_data': sum(sizes),
                'avg_bandwidth_gbps': (sum(sizes) / sum(durations)) / 1e9 if sum(durations) > 0 else 0
            }
            
        # 带宽分析
        self._analyze_bandwidth_patterns(analysis)
        
        # 延迟分析
        self._analyze_latency_patterns(analysis)
        
        return analysis
        
    def _analyze_bandwidth_patterns(self, analysis: Dict[str, Any]):
        """分析带宽模式"""
        bandwidth_data = []
        
        for record in self.comm_records:
            if record['duration'] > 0:
                bandwidth_gbps = (record['tensor_size'] / record['duration']) / 1e9
                bandwidth_data.append({
                    'op_type': record['op_type'],
                    'tensor_size': record['tensor_size'],
                    'duration': record['duration'],
                    'bandwidth_gbps': bandwidth_gbps
                })
                
        if bandwidth_data:
            bandwidths = [b['bandwidth_gbps'] for b in bandwidth_data]
            analysis['bandwidth_analysis'] = {
                'avg_bandwidth_gbps': np.mean(bandwidths),
                'max_bandwidth_gbps': max(bandwidths), 
                'min_bandwidth_gbps': min(bandwidths),
                'std_bandwidth_gbps': np.std(bandwidths),
                'by_size_range': self._group_by_size_range(bandwidth_data)
            }
            
    def _analyze_latency_patterns(self, analysis: Dict[str, Any]):
        """分析延迟模式"""
        # 按tensor size分组分析延迟
        size_groups = defaultdict(list)
        
        for record in self.comm_records:
            size_category = self._categorize_tensor_size(record['tensor_size'])
            size_groups[size_category].append(record['duration'])
            
        latency_by_size = {}
        for size_cat, durations in size_groups.items():
            latency_by_size[size_cat] = {
                'avg_latency': np.mean(durations),
                'std_latency': np.std(durations),
                'count': len(durations)
            }
            
        analysis['latency_analysis'] = {
            'by_tensor_size': latency_by_size,
            'baseline_latency_us': self._estimate_baseline_latency()
        }
        
    def _group_by_size_range(self, bandwidth_data: List[Dict]) -> Dict[str, Any]:
        """按size范围分组带宽数据"""
        size_ranges = {
            'small_1kb': (0, 1024),
            'medium_1mb': (1024, 1024*1024), 
            'large_100mb': (1024*1024, 100*1024*1024),
            'xlarge_1gb': (100*1024*1024, float('inf'))
        }
        
        grouped = {}
        for range_name, (min_size, max_size) in size_ranges.items():
            range_data = [b for b in bandwidth_data 
                         if min_size <= b['tensor_size'] < max_size]
            if range_data:
                bandwidths = [b['bandwidth_gbps'] for b in range_data]
                grouped[range_name] = {
                    'count': len(range_data),
                    'avg_bandwidth': np.mean(bandwidths),
                    'max_bandwidth': max(bandwidths)
                }
                
        return grouped
        
    def _categorize_tensor_size(self, size: int) -> str:
        """按tensor大小分类"""
        if size < 1024:
            return 'tiny'
        elif size < 1024 * 1024:
            return 'small'  
        elif size < 100 * 1024 * 1024:
            return 'medium'
        else:
            return 'large'
            
    def _estimate_baseline_latency(self) -> float:
        """估算基础延迟(μs)"""
        # 找最小的通信操作来估算基础延迟
        small_ops = [r for r in self.comm_records if r['tensor_size'] < 1024]
        if small_ops:
            return min(r['duration'] for r in small_ops) * 1e6  # 转换为微秒
        return 0.0
        
    def save_analysis(self, analysis: Dict[str, Any]):
        """保存通信分析结果"""
        
        # 保存详细记录
        records_file = os.path.join(self.output_dir, "comm_records.json")
        with open(records_file, 'w') as f:
            json.dump(self.comm_records, f, indent=2, default=str)
            
        # 保存分析结果
        analysis_file = os.path.join(self.output_dir, "comm_analysis.json") 
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        print(f"Communication analysis saved to {self.output_dir}")
        
    def print_communication_summary(self, analysis: Dict[str, Any]):
        """打印通信分析摘要"""
        print("\n" + "="*60)
        print("DISTRIBUTED COMMUNICATION ANALYSIS")
        print("="*60)
        
        summary = analysis['summary']
        print(f"Total Communication Operations: {summary['total_comm_ops']}")
        print(f"Total Communication Time: {summary['total_comm_time']*1000:.3f} ms")
        print(f"Average Operation Time: {summary['avg_comm_time']*1000:.3f} ms")
        print(f"Total Data Transferred: {summary['total_data_transferred']/1024/1024:.2f} MB")
        
        print("\nBy Operation Type:")
        for op_type, stats in analysis['by_operation_type'].items():
            print(f"  {op_type:12s}: {stats['count']:4d} ops, "
                  f"{stats['total_time']*1000:7.2f}ms total, "
                  f"{stats['avg_bandwidth_gbps']:6.2f} GB/s avg")
                  
        if 'bandwidth_analysis' in analysis:
            bw = analysis['bandwidth_analysis']
            print(f"\nBandwidth Analysis:")
            print(f"  Average: {bw['avg_bandwidth_gbps']:.2f} GB/s")
            print(f"  Peak:    {bw['max_bandwidth_gbps']:.2f} GB/s") 
            
        print("="*60)


# 集成到主profiler中的通信专项测试
def run_communication_benchmark():
    """运行通信基准测试"""
    
    comm_profiler = DistributedCommProfiler()
    
    # Patch通信操作
    comm_profiler.patch_communication_ops()
    comm_profiler.patch_pipeline_ops()
    
    # 运行基准测试 (需要在分布式环境中运行)
    print("Running communication-focused benchmark...")
    
    # 这里可以集成到主要的profiling流程中
    # 或者单独运行通信测试
    
    return comm_profiler


if __name__ == "__main__":
    print("Communication profiler can be used as a component in the main profiling script")
    profiler = run_communication_benchmark()
