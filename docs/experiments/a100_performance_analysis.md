# A100 双卡性能测试实验设计

## 实验概述

本实验旨在系统性地测量和分析双A100 GPU环境下的推理性能，重点关注：
1. 推理时间与序列长度的关系
2. Pipeline Parallelism (PP) 层间通信开销
3. Tensor Parallelism (TP) 通信开销
4. GPU间带宽测试

## 实验环境

- **硬件配置**: 2 × NVIDIA A100 80GB
- **软件环境**: vLLM框架
- **测试模型**: 建议使用Llama2-7B, Llama2-13B作为基准模型
- **CUDA版本**: 11.8+
- **驱动版本**: 最新稳定版

## 快速开始

我们提供了完整的实验脚本和工具，可以一键运行所有测试：

```bash
# 进入实验目录
cd experiments/dual_gpu_tests

# 运行快速测试（推荐首次使用）
./quick_start.sh quick

# 运行特定测试
./quick_start.sh bandwidth          # GPU带宽测试
./quick_start.sh sequence          # 序列长度测试
./quick_start.sh pp                # PP开销测试
./quick_start.sh tp                # TP通信测试

# 运行完整综合测试
./quick_start.sh comprehensive
```

**实验脚本说明**：
- `gpu_bandwidth_test.py`: GPU间带宽测试
- `sequence_length_test.py`: 序列长度扩展性测试  
- `pp_overhead_test.py`: Pipeline Parallel开销测试
- `tp_communication_test.py`: Tensor Parallel通信测试
- `run_comprehensive_test.py`: 综合测试控制器
- `visualization_generator.py`: 结果可视化
- `quick_start.sh`: 一键启动脚本
- `config.ini`: 测试配置文件

## 实验设计

### 实验1: 推理时间与序列长度关系测试

#### 目标
量化分析推理延迟随输入序列长度的变化规律，识别性能瓶颈点。

#### 测试参数
```bash
# 序列长度测试范围
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# 批次大小
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

# 输出长度
OUTPUT_LENGTHS = [50, 100, 200, 500, 1000]

# 测试配置
TENSOR_PARALLEL_SIZE = [1, 2]  # 单卡vs双卡TP
PIPELINE_PARALLEL_SIZE = [1, 2]  # 单stage vs双stage PP
```

#### 测试脚本示例
```python
# experiments/latency_vs_sequence_length.py
import time
import torch
import json
from vllm import LLM, SamplingParams
from typing import List, Dict

def benchmark_sequence_lengths():
    results = {}
    
    for tp_size in [1, 2]:
        for pp_size in [1, 2]:
            if tp_size * pp_size > 2:  # 总GPU数不超过2
                continue
                
            llm = LLM(
                model="meta-llama/Llama-2-7b-hf",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                gpu_memory_utilization=0.8,
                max_model_len=32768
            )
            
            config_key = f"tp{tp_size}_pp{pp_size}"
            results[config_key] = {}
            
            for seq_len in SEQUENCE_LENGTHS:
                for batch_size in BATCH_SIZES:
                    # 生成固定长度的输入
                    prompts = ["A" * seq_len] * batch_size
                    
                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=100,
                        ignore_eos=True
                    )
                    
                    # 预热
                    _ = llm.generate(prompts[:1], sampling_params)
                    
                    # 实际测试
                    start_time = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params)
                    end_time = time.perf_counter()
                    
                    latency = end_time - start_time
                    tokens_per_sec = sum(len(out.outputs[0].token_ids) for out in outputs) / latency
                    
                    key = f"seq{seq_len}_batch{batch_size}"
                    results[config_key][key] = {
                        "latency": latency,
                        "tokens_per_sec": tokens_per_sec,
                        "memory_usage": torch.cuda.max_memory_allocated() / 1024**3
                    }
            
            del llm
            torch.cuda.empty_cache()
    
    return results
```

### 实验2: Pipeline Parallelism (PP) 开销测试

#### 目标
测量PP模式下的层间通信开销，分析不同模型大小和配置的影响。

#### 测试策略
```python
# experiments/pipeline_overhead.py
def benchmark_pipeline_overhead():
    """
    比较PP=1 vs PP=2的性能差异，量化通信开销
    """
    results = {}
    
    models = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf"
    ]
    
    for model_name in models:
        results[model_name] = {}
        
        # 测试PP=1 (单GPU流水线)
        llm_single = LLM(
            model=model_name,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # 测试PP=2 (双GPU流水线)
        llm_pipeline = LLM(
            model=model_name,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            gpu_memory_utilization=0.8
        )
        
        for seq_len in [1024, 2048, 4096]:
            for batch_size in [1, 4, 8]:
                prompt = "A" * seq_len
                prompts = [prompt] * batch_size
                
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=200
                )
                
                # 单GPU测试
                start_time = time.perf_counter()
                _ = llm_single.generate(prompts, sampling_params)
                single_latency = time.perf_counter() - start_time
                
                # PP双GPU测试
                start_time = time.perf_counter()
                _ = llm_pipeline.generate(prompts, sampling_params)
                pipeline_latency = time.perf_counter() - start_time
                
                # 计算开销
                overhead_ratio = (pipeline_latency / single_latency - 1.0) * 100
                
                key = f"seq{seq_len}_batch{batch_size}"
                results[model_name][key] = {
                    "single_gpu_latency": single_latency,
                    "pipeline_latency": pipeline_latency,
                    "overhead_percentage": overhead_ratio
                }
    
    return results
```

### 实验3: Tensor Parallelism (TP) 开销测试

#### 目标
评估TP模式下的通信开销和扩展效率。

#### 测试方法
```python
# experiments/tensor_parallel_overhead.py
def benchmark_tensor_parallel_overhead():
    """
    比较TP=1 vs TP=2的性能和通信开销
    """
    results = {}
    
    configurations = [
        {"tp": 1, "pp": 1, "name": "single_gpu"},
        {"tp": 2, "pp": 1, "name": "tensor_parallel"}
    ]
    
    for config in configurations:
        llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            tensor_parallel_size=config["tp"],
            pipeline_parallel_size=config["pp"],
            gpu_memory_utilization=0.8
        )
        
        config_results = {}
        
        for seq_len in [512, 1024, 2048, 4096]:
            batch_sizes = [1, 2, 4, 8, 16] if config["tp"] == 2 else [1, 2, 4, 8]
            
            for batch_size in batch_sizes:
                prompts = ["Test prompt " + "A" * seq_len] * batch_size
                
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=100
                )
                
                # 多次测试取平均
                latencies = []
                for _ in range(5):
                    start_time = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params)
                    latency = time.perf_counter() - start_time
                    latencies.append(latency)
                
                avg_latency = sum(latencies) / len(latencies)
                throughput = len(prompts) * 100 / avg_latency  # tokens/sec approximate
                
                key = f"seq{seq_len}_batch{batch_size}"
                config_results[key] = {
                    "avg_latency": avg_latency,
                    "throughput": throughput,
                    "memory_per_gpu": torch.cuda.max_memory_allocated() / 1024**3
                }
        
        results[config["name"]] = config_results
        del llm
        torch.cuda.empty_cache()
    
    # 计算TP效率
    efficiency_results = {}
    for key in results["single_gpu"].keys():
        single_throughput = results["single_gpu"][key]["throughput"]
        tp_throughput = results["tensor_parallel"][key]["throughput"]
        
        # 理想情况下TP=2应该有2倍性能提升
        efficiency = (tp_throughput / single_throughput) / 2.0 * 100
        
        efficiency_results[key] = {
            "single_gpu_throughput": single_throughput,
            "tp_throughput": tp_throughput,
            "scaling_efficiency": efficiency
        }
    
    results["efficiency_analysis"] = efficiency_results
    return results
```

### 实验4: GPU间带宽测试

#### 目标
测量两块A100之间的实际通信带宽。

#### 测试脚本
```python
# experiments/gpu_bandwidth_test.py
import torch
import time

def benchmark_gpu_bandwidth():
    """
    测试GPU间的点对点带宽和全归约带宽
    """
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")
    
    results = {}
    
    # 测试不同数据大小的传输
    data_sizes_mb = [1, 10, 100, 500, 1000, 2000]  # MB
    
    for size_mb in data_sizes_mb:
        size_elements = size_mb * 1024 * 1024 // 4  # float32
        
        # P2P传输测试
        data_gpu0 = torch.randn(size_elements, device=device_0, dtype=torch.float32)
        
        # 预热
        for _ in range(10):
            data_gpu1 = data_gpu0.to(device_1)
            torch.cuda.synchronize()
        
        # 实际测试
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(100):
            data_gpu1 = data_gpu0.to(device_1)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        transfer_time = (end_time - start_time) / 100
        bandwidth_gbps = (size_mb / 1024) / transfer_time
        
        results[f"{size_mb}MB"] = {
            "transfer_time_ms": transfer_time * 1000,
            "bandwidth_GB_per_sec": bandwidth_gbps
        }
    
    # 全归约测试 (模拟集合通信)
    all_reduce_results = {}
    
    for size_mb in [10, 100, 500, 1000]:
        size_elements = size_mb * 1024 * 1024 // 4
        
        # 创建测试数据
        data_list = [
            torch.randn(size_elements, device=f"cuda:{i}", dtype=torch.float32) 
            for i in range(2)
        ]
        
        # 模拟all-reduce操作
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # 简单的reduce实现
        result = data_list[0] + data_list[1].to(device_0)
        result_gpu1 = result.to(device_1)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        all_reduce_time = end_time - start_time
        all_reduce_results[f"{size_mb}MB"] = {
            "all_reduce_time_ms": all_reduce_time * 1000,
            "effective_bandwidth_GB_per_sec": (size_mb * 2 / 1024) / all_reduce_time
        }
    
    results["all_reduce"] = all_reduce_results
    return results
```

## 实验执行计划

### 第1阶段: 基础性能建立 (预计2天)
1. 环境设置和验证
2. 单GPU基准测试
3. 简单的序列长度vs延迟关系

### 第2阶段: 并行性能测试 (预计3天)
1. TP并行效率测试
2. PP通信开销测试
3. 混合并行配置测试

### 第3阶段: 深度分析 (预计2天)
1. GPU间带宽详细测试
2. 内存使用分析
3. 热点识别和瓶颈分析

## 预期结果和分析

### 关键指标
1. **延迟分析**:
   - Time to First Token (TTFT)
   - Time per Token (TPT) 
   - 总推理时间

2. **吞吐量分析**:
   - Tokens per second
   - Requests per second
   - 批处理效率

3. **扩展性分析**:
   - TP扩展效率 (理想值100%)
   - PP开销百分比
   - 内存利用率

4. **通信分析**:
   - P2P带宽利用率
   - 集合通信效率
   - 通信延迟分布

### 预期发现
- 序列长度对内存和计算的非线性影响
- TP vs PP在不同工负载下的性能权衡
- GPU间通信瓶颈识别
- 最优配置推荐

## 自动化测试脚本

```bash
#!/bin/bash
# run_all_experiments.sh

echo "Starting A100 Performance Analysis Experiments..."

# 实验1: 序列长度vs延迟
echo "Running Experiment 1: Sequence Length Analysis..."
python experiments/latency_vs_sequence_length.py > results/exp1_sequence_length.json

# 实验2: PP开销测试
echo "Running Experiment 2: Pipeline Parallelism Overhead..."
python experiments/pipeline_overhead.py > results/exp2_pp_overhead.json

# 实验3: TP开销测试
echo "Running Experiment 3: Tensor Parallelism Overhead..."
python experiments/tensor_parallel_overhead.py > results/exp3_tp_overhead.json

# 实验4: 带宽测试
echo "Running Experiment 4: GPU Bandwidth Test..."
python experiments/gpu_bandwidth_test.py > results/exp4_bandwidth.json

# 生成综合报告
python experiments/generate_report.py

echo "All experiments completed. Results saved in results/ directory."
```

## 结果可视化

建议使用matplotlib和seaborn创建以下图表:
1. 序列长度vs延迟的散点图和拟合曲线
2. TP/PP配置的性能对比柱状图
3. 带宽测试的传输大小vs速度曲线
4. 内存使用随配置变化的热力图

## 注意事项

1. **GPU温度监控**: 长时间测试需要监控温度，避免降频
2. **内存清理**: 每次测试间要清理GPU内存
3. **多次运行**: 每个配置至少运行5次取平均值
4. **系统负载**: 确保系统其他进程不影响测试结果
5. **版本记录**: 记录vLLM、CUDA、驱动的具体版本

这个实验设计将为您的A100双卡环境提供全面的性能基线和优化指导。

## 实验执行

完整的实验实现已准备就绪，包含：

### 自动化脚本
- **完整测试套件**: `experiments/run_all_experiments.sh` - 一键运行所有实验
- **序列长度分析**: `experiments/latency_vs_sequence_length.py`
- **并行化开销测试**: `experiments/parallelism_overhead.py`  
- **GPU带宽测试**: `experiments/gpu_bandwidth_test.py`
- **结果可视化**: `experiments/visualize_results.py`

### 快速开始
```bash
# 进入项目根目录
cd /home/tjy/hotLLM

# 运行完整测试套件
./experiments/run_all_experiments.sh

# 生成可视化报告  
python experiments/visualize_results.py
```

详细使用说明请参见: `experiments/README.md`
