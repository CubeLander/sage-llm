#!/usr/bin/env python3
"""
GPU间带宽测试脚本

测试A100 GPU之间的点对点通信带宽和集合通信性能。
"""

import os
import time
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

def test_p2p_bandwidth():
    """测试点对点传输带宽"""
    print("Testing P2P bandwidth...")
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for P2P testing")
        return {}
    
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")
    
    # 启用P2P访问
    try:
        torch.cuda.set_device(0)
        if torch.cuda.can_device_access_peer(0, 1):
            torch.cuda.set_device(1)
            torch.cuda.set_device(0)
            print("P2P access enabled")
        else:
            print("Warning: P2P access not available")
    except:
        print("P2P setup failed, continuing with regular transfers")
    
    results = {}
    data_sizes_mb = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    for size_mb in data_sizes_mb:
        print(f"  Testing {size_mb} MB...")
        size_elements = size_mb * 1024 * 1024 // 4  # float32 elements
        
        # 创建测试数据
        data_gpu0 = torch.randn(size_elements, device=device_0, dtype=torch.float32)
        
        # 预热
        for _ in range(10):
            data_gpu1 = data_gpu0.to(device_1)
            torch.cuda.synchronize()
        
        # 测试 GPU0 -> GPU1
        times_0_to_1 = []
        for _ in range(20):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            data_gpu1 = data_gpu0.to(device_1)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times_0_to_1.append(end_time - start_time)
        
        # 测试 GPU1 -> GPU0
        data_gpu1 = torch.randn(size_elements, device=device_1, dtype=torch.float32)
        times_1_to_0 = []
        for _ in range(20):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            data_gpu0_copy = data_gpu1.to(device_0)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times_1_to_0.append(end_time - start_time)
        
        # 计算带宽
        avg_time_0_to_1 = np.mean(times_0_to_1)
        avg_time_1_to_0 = np.mean(times_1_to_0)
        
        bandwidth_0_to_1 = (size_mb / 1024) / avg_time_0_to_1  # GB/s
        bandwidth_1_to_0 = (size_mb / 1024) / avg_time_1_to_0  # GB/s
        
        results[f"{size_mb}MB"] = {
            "size_mb": size_mb,
            "gpu0_to_gpu1_time_ms": avg_time_0_to_1 * 1000,
            "gpu1_to_gpu0_time_ms": avg_time_1_to_0 * 1000,
            "gpu0_to_gpu1_bandwidth_gbps": bandwidth_0_to_1,
            "gpu1_to_gpu0_bandwidth_gbps": bandwidth_1_to_0,
            "bidirectional_avg_bandwidth_gbps": (bandwidth_0_to_1 + bandwidth_1_to_0) / 2
        }
    
    return results

def test_all_reduce_bandwidth():
    """测试All-Reduce集合通信带宽"""
    print("Testing All-Reduce bandwidth...")
    
    if torch.cuda.device_count() < 2:
        return {}
    
    # 检查NCCL是否可用
    nccl_available = hasattr(torch.distributed, 'all_reduce')
    
    results = {}
    data_sizes_mb = [1, 10, 50, 100, 500, 1000]
    
    for size_mb in data_sizes_mb:
        print(f"  Testing {size_mb} MB...")
        size_elements = size_mb * 1024 * 1024 // 4
        
        # 简单的手动all-reduce实现（不使用NCCL）
        data_0 = torch.randn(size_elements, device="cuda:0", dtype=torch.float32)
        data_1 = torch.randn(size_elements, device="cuda:1", dtype=torch.float32)
        
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # 模拟all-reduce: 每个GPU获得所有数据的和
            sum_result = data_0 + data_1.to("cuda:0")
            result_0 = sum_result
            result_1 = sum_result.to("cuda:1")
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        # All-reduce的有效带宽计算：总传输数据量 / 时间
        effective_bandwidth = (size_mb * 2 / 1024) / avg_time  # GB/s
        
        results[f"{size_mb}MB"] = {
            "size_mb": size_mb,
            "all_reduce_time_ms": avg_time * 1000,
            "effective_bandwidth_gbps": effective_bandwidth
        }
    
    return results

def test_memory_bandwidth():
    """测试单GPU内存带宽作为参考"""
    print("Testing single GPU memory bandwidth...")
    
    device = torch.device("cuda:0")
    results = {}
    data_sizes_mb = [10, 100, 500, 1000, 2000]
    
    for size_mb in data_sizes_mb:
        print(f"  Testing {size_mb} MB...")
        size_elements = size_mb * 1024 * 1024 // 4
        
        # GPU内存分配和复制
        src = torch.randn(size_elements, device=device, dtype=torch.float32)
        
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            dst = src.clone()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        bandwidth = (size_mb / 1024) / avg_time  # GB/s
        
        results[f"{size_mb}MB"] = {
            "size_mb": size_mb,
            "memory_copy_time_ms": avg_time * 1000,
            "memory_bandwidth_gbps": bandwidth
        }
    
    return results

def test_nvlink_topology():
    """检测NVLink拓扑"""
    print("Checking NVLink topology...")
    
    topology_info = {
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [],
        "p2p_access_matrix": [],
        "nvlink_info": "Not available through PyTorch directly"
    }
    
    for i in range(torch.cuda.device_count()):
        topology_info["gpu_names"].append(torch.cuda.get_device_name(i))
    
    # 检查P2P访问矩阵
    gpu_count = torch.cuda.device_count()
    p2p_matrix = []
    
    for i in range(gpu_count):
        row = []
        for j in range(gpu_count):
            if i == j:
                row.append(True)  # 自己总是可以访问
            else:
                try:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    row.append(can_access)
                except:
                    row.append(False)
        p2p_matrix.append(row)
    
    topology_info["p2p_access_matrix"] = p2p_matrix
    
    return topology_info

def main():
    print("Starting GPU Bandwidth Benchmark")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }
    
    # 检查拓扑
    results["topology"] = test_nvlink_topology()
    
    # 测试单GPU内存带宽（参考基准）
    results["single_gpu_memory"] = test_memory_bandwidth()
    
    if torch.cuda.device_count() >= 2:
        # 测试P2P带宽
        results["p2p_bandwidth"] = test_p2p_bandwidth()
        
        # 测试All-Reduce带宽
        results["all_reduce_bandwidth"] = test_all_reduce_bandwidth()
    else:
        print("Skipping multi-GPU tests (need at least 2 GPUs)")
        results["p2p_bandwidth"] = {}
        results["all_reduce_bandwidth"] = {}
    
    # 分析结果
    analysis = analyze_bandwidth_results(results)
    results["analysis"] = analysis
    
    # 保存结果
    os.makedirs("experiments/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/results/bandwidth_test_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # 打印总结
    print_bandwidth_summary(results)

def analyze_bandwidth_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """分析带宽测试结果"""
    analysis = {}
    
    # 分析P2P带宽
    if results["p2p_bandwidth"]:
        p2p_bandwidths = [v["bidirectional_avg_bandwidth_gbps"] for v in results["p2p_bandwidth"].values()]
        analysis["p2p_analysis"] = {
            "peak_bandwidth_gbps": max(p2p_bandwidths),
            "avg_bandwidth_gbps": np.mean(p2p_bandwidths),
            "bandwidth_range": {
                "min": min(p2p_bandwidths),
                "max": max(p2p_bandwidths)
            }
        }
    
    # 分析内存带宽
    if results["single_gpu_memory"]:
        mem_bandwidths = [v["memory_bandwidth_gbps"] for v in results["single_gpu_memory"].values()]
        analysis["memory_analysis"] = {
            "peak_memory_bandwidth_gbps": max(mem_bandwidths),
            "avg_memory_bandwidth_gbps": np.mean(mem_bandwidths)
        }
    
    # 效率分析
    if "p2p_analysis" in analysis and "memory_analysis" in analysis:
        analysis["efficiency_analysis"] = {
            "p2p_vs_memory_ratio": analysis["p2p_analysis"]["peak_bandwidth_gbps"] / analysis["memory_analysis"]["peak_memory_bandwidth_gbps"],
            "theoretical_nvlink_utilization": analysis["p2p_analysis"]["peak_bandwidth_gbps"] / 300.0  # A100 NVLink理论带宽约300GB/s
        }
    
    return analysis

def print_bandwidth_summary(results: Dict[str, Any]):
    """打印带宽测试总结"""
    print("\n=== Bandwidth Test Summary ===")
    
    if results.get("analysis", {}).get("p2p_analysis"):
        p2p = results["analysis"]["p2p_analysis"]
        print(f"Peak P2P Bandwidth: {p2p['peak_bandwidth_gbps']:.2f} GB/s")
        print(f"Average P2P Bandwidth: {p2p['avg_bandwidth_gbps']:.2f} GB/s")
    
    if results.get("analysis", {}).get("memory_analysis"):
        mem = results["analysis"]["memory_analysis"]
        print(f"Peak Memory Bandwidth: {mem['peak_memory_bandwidth_gbps']:.2f} GB/s")
    
    if results.get("analysis", {}).get("efficiency_analysis"):
        eff = results["analysis"]["efficiency_analysis"]
        print(f"P2P vs Memory Ratio: {eff['p2p_vs_memory_ratio']:.2f}")
        print(f"NVLink Utilization: {eff['theoretical_nvlink_utilization']*100:.1f}%")

if __name__ == "__main__":
    main()
