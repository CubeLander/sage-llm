#!/usr/bin/env python3
"""
A100 双卡推理延迟与序列长度关系测试

测试不同序列长度下的推理性能，分析延迟随序列长度的变化规律。
"""

import os
import time
import json
import argparse
import torch
import numpy as np
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from datetime import datetime

# 测试配置
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 2, 4, 8]
OUTPUT_LENGTHS = [50, 100, 200]
WARMUP_RUNS = 3
TEST_RUNS = 5

def setup_model(model_name: str, tp_size: int, pp_size: int, max_model_len: int = 16384):
    """初始化vLLM模型"""
    print(f"Setting up model {model_name} with TP={tp_size}, PP={pp_size}")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        disable_log_stats=True,
        enforce_eager=True  # 避免编译开销影响测试
    )
    
    return llm

def generate_test_prompts(seq_length: int, batch_size: int) -> List[str]:
    """生成固定长度的测试输入"""
    # 使用重复的文本内容确保长度一致
    base_text = "The quick brown fox jumps over the lazy dog. " * 20
    
    # 精确控制序列长度
    if len(base_text) >= seq_length:
        prompt = base_text[:seq_length]
    else:
        repeat_count = (seq_length // len(base_text)) + 1
        prompt = (base_text * repeat_count)[:seq_length]
    
    return [prompt] * batch_size

def run_latency_test(llm: LLM, prompts: List[str], output_length: int) -> Dict[str, float]:
    """运行延迟测试"""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_length,
        ignore_eos=True
    )
    
    # 预热
    for _ in range(WARMUP_RUNS):
        _ = llm.generate(prompts[:1], sampling_params)
    
    # 实际测试
    latencies = []
    memory_usage = []
    
    for _ in range(TEST_RUNS):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
        
        # 记录内存使用
        memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
    
    # 计算统计数据
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    total_output_tokens = len(prompts) * output_length
    avg_throughput = total_output_tokens / avg_latency
    
    return {
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "throughput_tokens_per_sec": avg_throughput,
        "avg_memory_gb": np.mean(memory_usage),
        "latencies": latencies
    }

def benchmark_sequence_lengths(model_name: str, results_dir: str):
    """主测试函数"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "configurations": {}
    }
    
    # 测试不同的并行配置
    test_configs = [
        {"tp_size": 1, "pp_size": 1, "name": "single_gpu"},
        {"tp_size": 2, "pp_size": 1, "name": "tensor_parallel_2"},
    ]
    
    if torch.cuda.device_count() >= 2:
        test_configs.append({"tp_size": 1, "pp_size": 2, "name": "pipeline_parallel_2"})
    
    for config in test_configs:
        print(f"\n=== Testing Configuration: {config['name']} ===")
        
        try:
            llm = setup_model(
                model_name, 
                config["tp_size"], 
                config["pp_size"],
                max_model_len=max(SEQUENCE_LENGTHS) + max(OUTPUT_LENGTHS)
            )
            
            config_results = {}
            
            for seq_len in SEQUENCE_LENGTHS:
                print(f"\nTesting sequence length: {seq_len}")
                
                for batch_size in BATCH_SIZES:
                    print(f"  Batch size: {batch_size}")
                    
                    try:
                        prompts = generate_test_prompts(seq_len, batch_size)
                        
                        for output_len in OUTPUT_LENGTHS:
                            print(f"    Output length: {output_len}")
                            
                            test_results = run_latency_test(llm, prompts, output_len)
                            
                            key = f"seq{seq_len}_batch{batch_size}_out{output_len}"
                            config_results[key] = {
                                "sequence_length": seq_len,
                                "batch_size": batch_size,
                                "output_length": output_len,
                                **test_results
                            }
                            
                            # 实时保存结果
                            results["configurations"][config["name"]] = config_results
                            
                            with open(os.path.join(results_dir, "latency_results_partial.json"), "w") as f:
                                json.dump(results, f, indent=2)
                    
                    except Exception as e:
                        print(f"    Error with seq_len={seq_len}, batch_size={batch_size}: {e}")
                        continue
            
            results["configurations"][config["name"]] = config_results
            
            # 清理模型
            del llm
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with configuration {config['name']}: {e}")
            continue
    
    return results

def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """分析测试结果"""
    analysis = {
        "summary": {},
        "scaling_analysis": {},
        "bottleneck_analysis": {}
    }
    
    # 计算不同配置的效率对比
    configs = list(results["configurations"].keys())
    if len(configs) >= 2:
        single_gpu_results = results["configurations"].get("single_gpu", {})
        
        for config_name in configs:
            if config_name == "single_gpu":
                continue
                
            config_results = results["configurations"][config_name]
            scaling_data = {}
            
            for key in single_gpu_results.keys():
                if key in config_results:
                    single_throughput = single_gpu_results[key]["throughput_tokens_per_sec"]
                    config_throughput = config_results[key]["throughput_tokens_per_sec"]
                    
                    speedup = config_throughput / single_throughput
                    scaling_data[key] = {
                        "speedup": speedup,
                        "efficiency": speedup / 2.0 * 100,  # 假设2个GPU
                        "single_gpu_throughput": single_throughput,
                        "parallel_throughput": config_throughput
                    }
            
            analysis["scaling_analysis"][config_name] = scaling_data
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="A100 Sequence Length Latency Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model to benchmark")
    parser.add_argument("--results-dir", default="experiments/results", help="Results directory")
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("Starting A100 Sequence Length Latency Benchmark")
    print(f"Model: {args.model}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # 运行基准测试
    results = benchmark_sequence_lengths(args.model, args.results_dir)
    
    # 分析结果
    analysis = analyze_results(results)
    results["analysis"] = analysis
    
    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.results_dir, f"latency_analysis_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # 打印简要总结
    print("\n=== Summary ===")
    for config_name, config_results in results["configurations"].items():
        if config_results:
            sample_key = next(iter(config_results.keys()))
            sample_result = config_results[sample_key]
            print(f"{config_name}: {sample_result['throughput_tokens_per_sec']:.2f} tokens/sec (avg)")

if __name__ == "__main__":
    main()
