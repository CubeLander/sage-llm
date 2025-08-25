#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM GPU性能监控命令行工具

这个工具提供了一个简单的命令行界面来监控vLLM的GPU使用率和性能指标。

使用方法：
    # 基础监控
    python -m vllm.profiler.cli --model facebook/opt-125m --monitor-basic
    
    # 详细监控
    python -m vllm.profiler.cli --model facebook/opt-125m --monitor-detailed
    
    # 实时监控模式
    python -m vllm.profiler.cli --model facebook/opt-125m --monitor-realtime
    
    # 自定义提示词监控
    python -m vllm.profiler.cli --model facebook/opt-125m --prompts-file prompts.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.profiler.integration import VLLMProfilingContext, get_monitoring_summary
from vllm.profiler.gpu_monitor import GPUHardwareMonitor

logger = init_logger(__name__)


class VLLMMonitorCLI:
    """vLLM监控命令行工具主类"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = None
        self.gpu_monitor = None
        
    def setup_llm(self):
        """设置vLLM实例"""
        logger.info(f"Loading model: {self.args.model}")
        
        llm_kwargs = {
            'model': self.args.model,
            'tensor_parallel_size': self.args.tensor_parallel_size,
            'max_num_seqs': self.args.max_num_seqs,
            'max_model_len': self.args.max_model_len,
            'gpu_memory_utilization': self.args.gpu_memory_utilization,
        }
        
        # 过滤掉None值
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
        
        self.llm = LLM(**llm_kwargs)
        logger.info("Model loaded successfully")
        
    def load_prompts(self) -> List[str]:
        """加载提示词"""
        if self.args.prompts_file:
            # 从文件加载提示词
            prompts_file = Path(self.args.prompts_file)
            if not prompts_file.exists():
                logger.error(f"Prompts file not found: {prompts_file}")
                sys.exit(1)
                
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
                
            logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
            return prompts
            
        elif self.args.custom_prompts:
            # 使用命令行提供的提示词
            return self.args.custom_prompts
            
        else:
            # 使用默认测试提示词
            default_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly,",
                "The most important lesson I've learned is",
                "Climate change is a pressing issue that requires",
                "The development of renewable energy sources",
                "Machine learning algorithms can help us",
                "The impact of social media on society",
                "Space exploration has always fascinated",
                "The importance of education in modern society",
                "Healthcare systems around the world are"
            ]
            
            # 根据需要的数量重复和截取
            num_prompts = self.args.num_prompts or 50
            prompts = []
            for i in range(num_prompts):
                prompt = default_prompts[i % len(default_prompts)]
                prompts.append(f"{prompt} (sample {i+1})")
                
            logger.info(f"Using {len(prompts)} default test prompts")
            return prompts
            
    def create_sampling_params(self) -> SamplingParams:
        """创建采样参数"""
        return SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens_per_prompt,
            seed=self.args.seed
        )
        
    def run_basic_monitoring(self):
        """运行基础监控"""
        logger.info("Starting basic monitoring...")
        
        profiling_config = {
            'enable_gpu_monitoring': True,
            'enable_layer_profiling': False,
            'output_dir': str(self.output_dir),
            'sample_interval': 0.1
        }
        
        with VLLMProfilingContext(**profiling_config):
            prompts = self.load_prompts()
            sampling_params = self.create_sampling_params()
            
            logger.info(f"Processing {len(prompts)} prompts...")
            start_time = time.time()
            
            # 批量处理或单个处理
            if self.args.batch_processing:
                outputs = self.llm.generate(prompts, sampling_params)
            else:
                outputs = []
                for i, prompt in enumerate(prompts):
                    logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                    batch_outputs = self.llm.generate([prompt], sampling_params)
                    outputs.extend(batch_outputs)
                    
                    # 每10个提示词输出进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i+1}/{len(prompts)} prompts")
                        
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Basic monitoring completed in {total_time:.2f} seconds")
            logger.info(f"Average time per prompt: {total_time/len(prompts):.3f} seconds")
            
            return outputs
            
    def run_detailed_monitoring(self):
        """运行详细监控"""
        logger.info("Starting detailed monitoring...")
        
        profiling_config = {
            'enable_gpu_monitoring': True,
            'enable_layer_profiling': True,
            'output_dir': str(self.output_dir),
            'sample_interval': 0.05  # 更频繁的采样
        }
        
        with VLLMProfilingContext(**profiling_config):
            # 详细监控使用较少的提示词
            prompts = self.load_prompts()[:min(10, len(self.load_prompts()))]
            sampling_params = self.create_sampling_params()
            
            logger.info(f"Processing {len(prompts)} prompts with detailed profiling...")
            
            outputs = []
            for i, prompt in enumerate(prompts):
                logger.info(f"Detailed profiling prompt {i+1}/{len(prompts)}")
                batch_outputs = self.llm.generate([prompt], sampling_params)
                outputs.extend(batch_outputs)
                
                # 输出中间结果
                summary = get_monitoring_summary()
                if 'monitors' in summary:
                    for monitor_name, monitor_data in summary['monitors'].items():
                        if 'avg_duration_ms' in monitor_data:
                            logger.info(f"Current avg duration: {monitor_data['avg_duration_ms']:.1f}ms")
                            
            logger.info("Detailed monitoring completed")
            return outputs
            
    def run_realtime_monitoring(self):
        """运行实时监控模式"""
        logger.info("Starting real-time monitoring mode...")
        
        # 启动GPU监控器
        self.gpu_monitor = GPUHardwareMonitor(sample_interval=0.1)
        self.gpu_monitor.start()
        
        try:
            prompts = self.load_prompts()
            sampling_params = self.create_sampling_params()
            
            # 实时显示GPU状态
            logger.info("Real-time GPU monitoring started. Press Ctrl+C to stop.")
            
            for round_num in range(self.args.monitoring_rounds):
                logger.info(f"\n=== Monitoring Round {round_num + 1}/{self.args.monitoring_rounds} ===")
                
                # 处理一批提示词
                batch_prompts = prompts[:self.args.batch_size] if self.args.batch_size else prompts
                
                for i, prompt in enumerate(batch_prompts):
                    # 显示当前GPU状态
                    gpu_metrics = self.gpu_monitor.get_current_metrics()
                    if gpu_metrics:
                        logger.info(
                            f"Prompt {i+1}: GPU={gpu_metrics.gpu_utilization:.1f}%, "
                            f"Memory={gpu_metrics.memory_utilization:.1f}%, "
                            f"Temp={gpu_metrics.temperature}°C"
                        )
                        
                    # 执行推理
                    outputs = self.llm.generate([prompt], sampling_params)
                    
                    # 短暂暂停以便观察
                    time.sleep(0.5)
                    
                # 输出这轮的统计
                recent_metrics = self.gpu_monitor.get_metrics_history(60.0)  # 最近1分钟
                if recent_metrics:
                    avg_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
                    max_util = max(m.gpu_utilization for m in recent_metrics)
                    logger.info(f"Round {round_num + 1} completed. Avg GPU: {avg_util:.1f}%, Max GPU: {max_util:.1f}%")
                    
                time.sleep(2)  # 轮间休息
                
        except KeyboardInterrupt:
            logger.info("Real-time monitoring stopped by user")
            
        finally:
            if self.gpu_monitor:
                self.gpu_monitor.stop()
                
                # 导出GPU监控历史
                gpu_history = self.gpu_monitor.get_metrics_history()
                if gpu_history:
                    history_file = self.output_dir / "gpu_monitoring_history.json"
                    history_data = [
                        {
                            'timestamp': m.timestamp,
                            'gpu_utilization': m.gpu_utilization,
                            'memory_utilization': m.memory_utilization,
                            'memory_used': m.memory_used,
                            'temperature': m.temperature,
                            'power_draw': m.power_draw
                        } for m in gpu_history
                    ]
                    
                    with open(history_file, 'w') as f:
                        json.dump(history_data, f, indent=2)
                        
                    logger.info(f"GPU monitoring history saved to {history_file}")
                    
    def run_custom_benchmark(self):
        """运行自定义基准测试"""
        logger.info("Starting custom benchmark...")
        
        # 测试不同的batch size
        batch_sizes = self.args.benchmark_batch_sizes or [1, 2, 4, 8]
        max_tokens_list = self.args.benchmark_max_tokens or [50, 100, 200]
        
        results = {}
        
        for batch_size in batch_sizes:
            for max_tokens in max_tokens_list:
                logger.info(f"Benchmarking: batch_size={batch_size}, max_tokens={max_tokens}")
                
                # 创建测试提示词
                test_prompts = self.load_prompts()[:batch_size]
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                
                # 使用监控上下文
                profiling_config = {
                    'enable_gpu_monitoring': True,
                    'enable_layer_profiling': False,
                    'output_dir': str(self.output_dir / f"benchmark_b{batch_size}_t{max_tokens}")
                }
                
                with VLLMProfilingContext(**profiling_config):
                    start_time = time.time()
                    outputs = self.llm.generate(test_prompts, sampling_params)
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
                    tokens_per_second = total_tokens / duration
                    
                    results[f"batch_{batch_size}_tokens_{max_tokens}"] = {
                        'batch_size': batch_size,
                        'max_tokens': max_tokens,
                        'duration_sec': duration,
                        'total_tokens': total_tokens,
                        'tokens_per_second': tokens_per_second,
                        'latency_per_sample': duration / batch_size
                    }
                    
                    logger.info(f"Result: {tokens_per_second:.1f} tokens/sec, {duration:.2f}s total")
                    
        # 保存基准测试结果
        benchmark_file = self.output_dir / "benchmark_results.json"
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Benchmark results saved to {benchmark_file}")
        
        # 输出摘要
        logger.info("\n=== Benchmark Summary ===")
        for config, result in results.items():
            logger.info(
                f"{config}: {result['tokens_per_second']:.1f} tok/s, "
                f"{result['latency_per_sample']:.3f}s/sample"
            )
            
    def run(self):
        """运行主程序"""
        try:
            # 设置vLLM
            self.setup_llm()
            
            # 根据模式运行不同的监控
            if self.args.mode == 'basic':
                self.run_basic_monitoring()
            elif self.args.mode == 'detailed':
                self.run_detailed_monitoring()
            elif self.args.mode == 'realtime':
                self.run_realtime_monitoring()
            elif self.args.mode == 'benchmark':
                self.run_custom_benchmark()
            else:
                logger.error(f"Unknown mode: {self.args.mode}")
                sys.exit(1)
                
            logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="vLLM GPU Performance Monitoring CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic monitoring with default prompts
    python -m vllm.profiler.cli --model facebook/opt-125m --mode basic
    
    # Detailed monitoring with custom prompts file
    python -m vllm.profiler.cli --model facebook/opt-125m --mode detailed --prompts-file my_prompts.txt
    
    # Real-time monitoring
    python -m vllm.profiler.cli --model facebook/opt-125m --mode realtime --monitoring-rounds 5
    
    # Custom benchmark
    python -m vllm.profiler.cli --model facebook/opt-125m --mode benchmark
        """
    )
    
    # vLLM model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                       help="Maximum number of sequences")
    parser.add_argument("--max-model-len", type=int,
                       help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    
    # Monitoring mode
    parser.add_argument("--mode", choices=['basic', 'detailed', 'realtime', 'benchmark'],
                       default='basic', help="Monitoring mode")
    
    # Input prompts
    parser.add_argument("--prompts-file", type=str,
                       help="File containing prompts (one per line)")
    parser.add_argument("--custom-prompts", nargs='+',
                       help="Custom prompts provided directly")
    parser.add_argument("--num-prompts", type=int, default=50,
                       help="Number of default prompts to generate")
    
    # Sampling parameters
    parser.add_argument("--max-tokens-per-prompt", type=int, default=100,
                       help="Maximum tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    
    # Processing options
    parser.add_argument("--batch-processing", action='store_true',
                       help="Process all prompts in a single batch")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size for processing")
    
    # Real-time monitoring options
    parser.add_argument("--monitoring-rounds", type=int, default=3,
                       help="Number of monitoring rounds (for realtime mode)")
    
    # Benchmark options
    parser.add_argument("--benchmark-batch-sizes", type=int, nargs='+',
                       default=[1, 2, 4, 8], help="Batch sizes for benchmark")
    parser.add_argument("--benchmark-max-tokens", type=int, nargs='+',
                       default=[50, 100, 200], help="Max tokens for benchmark")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./vllm_monitoring_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # 运行监控工具
    cli = VLLMMonitorCLI(args)
    cli.run()


if __name__ == "__main__":
    main()
