#!/usr/bin/env python3
"""
并行开销测试脚本

测试Tensor Parallelism (TP)和Pipeline Parallelism (PP)的通信开销。
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

def setup_model(model_name: str, tp_size: int, pp_size: int):
    """初始化vLLM模型"""
    print(f"Setting up {model_name} with TP={tp_size}, PP={pp_size}")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        disable_log_stats=True,
        enforce_eager=True
    )
    
    return llm

def run_inference_benchmark(llm: LLM, prompts: List[str], max_tokens: int, runs: int = 5):
    """运行推理基准测试"""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True
    )
    
    # 预热
    for _ in range(3):
        _ = llm.generate(prompts[:1], sampling_params)
    
    # 实际测试
    latencies = []
    memory_usage = []
    
    for _ in range(runs):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
        
        # 记录内存使用
        max_memory = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        memory_usage.append(max_memory / 1024**3)
    
    avg_latency = np.mean(latencies)
    total_tokens = len(prompts) * max_tokens
    throughput = total_tokens / avg_latency
    
    return {
        "avg_latency": avg_latency,
        "std_latency": np.std(latencies),
        "throughput_tokens_per_sec": throughput,
        "avg_memory_gb": np.mean(memory_usage),
        "all_latencies": latencies
    }

def test_tensor_parallelism_overhead(model_name: str):
    """测试Tensor Parallelism开销"""
    print("Testing Tensor Parallelism overhead...")
    
    results = {}
    test_configs = [
        {"tp": 1, "pp": 1, "name": "single_gpu"},
        {"tp": 2, "pp": 1, "name": "tensor_parallel"}
    ]
    
    # 测试参数
    test_cases = [
        {"seq_len": 512, "batch_size": 1, "output_len": 100},
        {"seq_len": 1024, "batch_size": 2, "output_len": 200},
        {"seq_len": 2048, "batch_size": 4, "output_len": 100},
        {"seq_len": 4096, "batch_size": 1, "output_len": 200},
    ]
    
    for config in test_configs:
        if config["tp"] > torch.cuda.device_count():
            print(f"Skipping {config['name']}: need {config['tp']} GPUs")
            continue
        
        print(f"\nTesting configuration: {config['name']}")
        
        try:
            llm = setup_model(model_name, config["tp"], config["pp"])
            config_results = {}
            
            for test_case in test_cases:
                print(f"  Testing: seq_len={test_case['seq_len']}, batch={test_case['batch_size']}")
                
                # 生成测试数据
                base_prompt = "Analyze the following data and provide insights: " + "A" * test_case["seq_len"]
                prompts = [base_prompt] * test_case["batch_size"]
                
                # 运行测试
                result = run_inference_benchmark(llm, prompts, test_case["output_len"])
                
                key = f"seq{test_case['seq_len']}_batch{test_case['batch_size']}_out{test_case['output_len']}"
                config_results[key] = {
                    **test_case,
                    **result
                }
            
            results[config["name"]] = config_results
            
            del llm
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing {config['name']}: {e}")
            results[config["name"]] = {}
    
    return results

def test_pipeline_parallelism_overhead(model_name: str):
    """测试Pipeline Parallelism开销"""
    print("Testing Pipeline Parallelism overhead...")
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for Pipeline Parallelism")
        return {}
    
    results = {}
    test_configs = [
        {"tp": 1, "pp": 1, "name": "single_gpu"},
        {"tp": 1, "pp": 2, "name": "pipeline_parallel"}
    ]
    
    # PP测试用例（通常对较大batch更有效）
    test_cases = [
        {"seq_len": 1024, "batch_size": 1, "output_len": 200},
        {"seq_len": 1024, "batch_size": 4, "output_len": 200},
        {"seq_len": 2048, "batch_size": 2, "output_len": 100},
        {"seq_len": 2048, "batch_size": 8, "output_len": 100},
    ]
    
    for config in test_configs:
        print(f"\nTesting configuration: {config['name']}")
        
        try:
            llm = setup_model(model_name, config["tp"], config["pp"])
            config_results = {}
            
            for test_case in test_cases:
                print(f"  Testing: seq_len={test_case['seq_len']}, batch={test_case['batch_size']}")
                
                # 生成测试数据
                base_prompt = "Explain the concept in detail: " + "B" * test_case["seq_len"]
                prompts = [base_prompt] * test_case["batch_size"]
                
                # 运行测试
                result = run_inference_benchmark(llm, prompts, test_case["output_len"])
                
                key = f"seq{test_case['seq_len']}_batch{test_case['batch_size']}_out{test_case['output_len']}"
                config_results[key] = {
                    **test_case,
                    **result
                }
            
            results[config["name"]] = config_results
            
            del llm
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing {config['name']}: {e}")
            results[config["name"]] = {}
    
    return results

def analyze_parallelism_results(tp_results: Dict, pp_results: Dict) -> Dict[str, Any]:
    """分析并行化结果"""
    analysis = {
        "tensor_parallelism": {},
        "pipeline_parallelism": {},
        "recommendations": []
    }
    
    # 分析TP结果
    if "single_gpu" in tp_results and "tensor_parallel" in tp_results:
        tp_analysis = {}
        
        for key in tp_results["single_gpu"].keys():
            if key in tp_results["tensor_parallel"]:
                single_throughput = tp_results["single_gpu"][key]["throughput_tokens_per_sec"]
                tp_throughput = tp_results["tensor_parallel"][key]["throughput_tokens_per_sec"]
                
                speedup = tp_throughput / single_throughput
                efficiency = speedup / 2.0 * 100  # 假设2个GPU
                
                single_latency = tp_results["single_gpu"][key]["avg_latency"]
                tp_latency = tp_results["tensor_parallel"][key]["avg_latency"]
                latency_reduction = (1 - tp_latency / single_latency) * 100
                
                tp_analysis[key] = {
                    "speedup": speedup,
                    "efficiency_percent": efficiency,
                    "latency_reduction_percent": latency_reduction,
                    "single_gpu_throughput": single_throughput,
                    "tp_throughput": tp_throughput
                }
        
        analysis["tensor_parallelism"] = tp_analysis
        
        # TP推荐
        avg_efficiency = np.mean([v["efficiency_percent"] for v in tp_analysis.values()])
        if avg_efficiency > 75:
            analysis["recommendations"].append("TP scaling is efficient (>75% efficiency)")
        elif avg_efficiency > 50:
            analysis["recommendations"].append("TP scaling is moderate (50-75% efficiency)")
        else:
            analysis["recommendations"].append("TP scaling is poor (<50% efficiency)")
    
    # 分析PP结果
    if "single_gpu" in pp_results and "pipeline_parallel" in pp_results:
        pp_analysis = {}
        
        for key in pp_results["single_gpu"].keys():
            if key in pp_results["pipeline_parallel"]:
                single_throughput = pp_results["single_gpu"][key]["throughput_tokens_per_sec"]
                pp_throughput = pp_results["pipeline_parallel"][key]["throughput_tokens_per_sec"]
                
                speedup = pp_throughput / single_throughput
                efficiency = speedup / 2.0 * 100
                
                pp_analysis[key] = {
                    "speedup": speedup,
                    "efficiency_percent": efficiency,
                    "single_gpu_throughput": single_throughput,
                    "pp_throughput": pp_throughput
                }
        
        analysis["pipeline_parallelism"] = pp_analysis
        
        # PP推荐
        if pp_analysis:
            avg_efficiency = np.mean([v["efficiency_percent"] for v in pp_analysis.values()])
            if avg_efficiency > 60:
                analysis["recommendations"].append("PP scaling is good for this workload")
            else:
                analysis["recommendations"].append("PP scaling shows overhead, consider TP instead")
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Parallelism Overhead Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", help="Model to benchmark")
    parser.add_argument("--test-tp", action="store_true", help="Test Tensor Parallelism")
    parser.add_argument("--test-pp", action="store_true", help="Test Pipeline Parallelism")
    parser.add_argument("--test-all", action="store_true", help="Test both TP and PP")
    
    args = parser.parse_args()
    
    if not any([args.test_tp, args.test_pp, args.test_all]):
        args.test_all = True
    
    print("Starting Parallelism Overhead Benchmark")
    print(f"Model: {args.model}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    
    tp_results = {}
    pp_results = {}
    
    if args.test_tp or args.test_all:
        tp_results = test_tensor_parallelism_overhead(args.model)
        results["tensor_parallelism_results"] = tp_results
    
    if args.test_pp or args.test_all:
        pp_results = test_pipeline_parallelism_overhead(args.model)
        results["pipeline_parallelism_results"] = pp_results
    
    # 分析结果
    if tp_results or pp_results:
        analysis = analyze_parallelism_results(tp_results, pp_results)
        results["analysis"] = analysis
    
    # 保存结果
    os.makedirs("experiments/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/results/parallelism_overhead_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # 打印总结
    print_parallelism_summary(results)

def print_parallelism_summary(results: Dict[str, Any]):
    """打印并行化测试总结"""
    print("\n=== Parallelism Overhead Summary ===")
    
    analysis = results.get("analysis", {})
    
    if "tensor_parallelism" in analysis and analysis["tensor_parallelism"]:
        tp_data = list(analysis["tensor_parallelism"].values())
        avg_tp_efficiency = np.mean([d["efficiency_percent"] for d in tp_data])
        avg_tp_speedup = np.mean([d["speedup"] for d in tp_data])
        
        print(f"Tensor Parallelism:")
        print(f"  Average Speedup: {avg_tp_speedup:.2f}x")
        print(f"  Average Efficiency: {avg_tp_efficiency:.1f}%")
    
    if "pipeline_parallelism" in analysis and analysis["pipeline_parallelism"]:
        pp_data = list(analysis["pipeline_parallelism"].values())
        avg_pp_efficiency = np.mean([d["efficiency_percent"] for d in pp_data])
        avg_pp_speedup = np.mean([d["speedup"] for d in pp_data])
        
        print(f"Pipeline Parallelism:")
        print(f"  Average Speedup: {avg_pp_speedup:.2f}x")
        print(f"  Average Efficiency: {avg_pp_efficiency:.1f}%")
    
    if "recommendations" in analysis:
        print(f"\nRecommendations:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()
