#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM实时GPU利用率和性能监控示例

这个脚本展示了如何在vLLM推理过程中实时监控：
1. GPU硬件利用率（使用率、显存、温度、功耗）
2. 每个torch操作的延迟和资源使用情况
3. 模型每一层的性能指标
4. 推理步骤的整体性能统计

运行方法：
python examples/gpu_profiling_example.py --model facebook/opt-125m --max-num-seqs 8
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.profiler.vllm_monitor import VLLMPerformanceMonitor, monitor_vllm_model_runner

logger = init_logger(__name__)


def create_sample_prompts(num_prompts: int = 10) -> list[str]:
    """创建测试用的提示词"""
    prompts = [
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
    
    # 重复和扩展到所需数量
    extended_prompts = []
    for i in range(num_prompts):
        base_prompt = prompts[i % len(prompts)]
        extended_prompts.append(f"{base_prompt} (sample {i+1})")
    
    return extended_prompts


class VLLMProfilingExample:
    """vLLM性能分析示例类"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化vLLM引擎
        self.engine_args = EngineArgs(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        
        self.llm = None
        self.monitor = None
        
    def setup_engine(self):
        """设置vLLM引擎和性能监控"""
        logger.info("Initializing vLLM engine...")
        
        # 创建LLM实例
        engine_config = self.engine_args.create_engine_config()
        self.llm = LLM(**engine_config.to_dict())
        
        # 获取ModelRunner实例
        if hasattr(self.llm.llm_engine, 'workers') and self.llm.llm_engine.workers:
            worker = self.llm.llm_engine.workers[0]
            if hasattr(worker, 'model_runner'):
                model_runner = worker.model_runner
                
                # 添加性能监控
                self.monitor = VLLMPerformanceMonitor(
                    model_runner=model_runner,
                    enable_realtime_monitoring=True,
                    enable_detailed_profiling=self.args.enable_detailed_profiling,
                    sample_interval=0.1
                )
                
                # 启动监控
                self.monitor.start_monitoring()
                logger.info("Performance monitoring started")
            else:
                logger.warning("Could not find model_runner in worker")
        else:
            logger.warning("Could not find workers in LLM engine")
            
    def run_basic_profiling(self):
        """运行基础性能分析"""
        logger.info("Starting basic profiling...")
        
        # 创建测试提示词
        prompts = create_sample_prompts(self.args.num_prompts)
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.args.max_tokens_per_prompt
        )
        
        logger.info(f"Running inference on {len(prompts)} prompts...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行推理
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            # 使用监控上下文
            if self.monitor:
                with self.monitor.profile_inference_step(
                    batch_size=1,
                    sequence_length=len(prompt.split()),
                    num_tokens=self.args.max_tokens_per_prompt
                ):
                    outputs = self.llm.generate([prompt], sampling_params)
            else:
                outputs = self.llm.generate([prompt], sampling_params)
                
            results.extend(outputs)
            
            # 每10个提示词输出一次中间统计
            if (i + 1) % 10 == 0 and self.monitor:
                summary = self.monitor.get_performance_summary(last_n_steps=10)
                logger.info(f"Last 10 steps average duration: {summary.get('avg_duration_ms', 0):.1f}ms")
                
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info(f"Completed inference in {total_duration:.2f} seconds")
        
        # 输出性能摘要
        if self.monitor:
            self._output_performance_summary(total_duration)
            
        return results
        
    def run_detailed_profiling(self):
        """运行详细的性能分析"""
        logger.info("Starting detailed profiling with torch profiler...")
        
        # 创建较少的提示词进行详细分析
        prompts = create_sample_prompts(min(5, self.args.num_prompts))
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.args.max_tokens_per_prompt
        )
        
        # 启用详细profiling
        if self.monitor:
            self.monitor.enable_detailed_profiling = True
            
        # 执行推理
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Detailed profiling prompt {i+1}/{len(prompts)}")
            
            if self.monitor:
                with self.monitor.profile_inference_step():
                    outputs = self.llm.generate([prompt], sampling_params)
            else:
                outputs = self.llm.generate([prompt], sampling_params)
                
            results.extend(outputs)
            
        return results
        
    def run_continuous_monitoring(self):
        """运行连续监控模式"""
        logger.info("Starting continuous monitoring...")
        
        prompts = create_sample_prompts(self.args.num_prompts)
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.args.max_tokens_per_prompt
        )
        
        # 连续运行多轮推理
        for round_num in range(self.args.monitoring_rounds):
            logger.info(f"Monitoring round {round_num + 1}/{self.args.monitoring_rounds}")
            
            for i, prompt in enumerate(prompts):
                if self.monitor:
                    with self.monitor.profile_inference_step():
                        outputs = self.llm.generate([prompt], sampling_params)
                else:
                    outputs = self.llm.generate([prompt], sampling_params)
                    
            # 每轮结束后输出统计
            if self.monitor:
                summary = self.monitor.get_performance_summary()
                logger.info(
                    f"Round {round_num + 1} completed. "
                    f"Total steps: {summary.get('total_steps', 0)}, "
                    f"Avg duration: {summary.get('avg_duration_ms', 0):.1f}ms"
                )
                
            # 短暂休息
            time.sleep(1)
            
    def _output_performance_summary(self, total_duration: float):
        """输出性能摘要"""
        if not self.monitor:
            return
            
        summary = self.monitor.get_performance_summary()
        
        logger.info("=== Performance Summary ===")
        logger.info(f"Total inference time: {total_duration:.2f}s")
        logger.info(f"Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"Average step duration: {summary.get('avg_duration_ms', 0):.1f}ms")
        logger.info(f"Min step duration: {summary.get('min_duration_ms', 0):.1f}ms")
        logger.info(f"Max step duration: {summary.get('max_duration_ms', 0):.1f}ms")
        logger.info(f"Average memory usage: {summary.get('avg_memory_mb', 0):.1f}MB")
        logger.info(f"Peak memory usage: {summary.get('max_memory_mb', 0):.1f}MB")
        
        # 输出层级性能统计
        layer_perf = summary.get('layer_performance', {})
        if layer_perf:
            logger.info("=== Top 10 Slowest Layers ===")
            sorted_layers = sorted(
                layer_perf.items(),
                key=lambda x: x[1]['avg_duration_ms'],
                reverse=True
            )[:10]
            
            for layer_name, stats in sorted_layers:
                logger.info(
                    f"{layer_name}: {stats['avg_duration_ms']:.2f}ms "
                    f"(calls: {stats['call_count']})"
                )
                
    def export_results(self):
        """导出监控结果"""
        if not self.monitor:
            logger.warning("No monitor available for export")
            return
            
        # 导出详细报告
        timestamp = int(time.time())
        report_file = self.output_dir / f"vllm_performance_report_{timestamp}.json"
        
        self.monitor.export_performance_report(report_file)
        
        # 导出摘要统计
        summary = self.monitor.get_performance_summary()
        summary_file = self.output_dir / f"performance_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Results exported to {self.output_dir}")
        
    def cleanup(self):
        """清理资源"""
        if self.monitor:
            self.monitor.stop_monitoring()
        logger.info("Cleanup completed")
        
    def run(self):
        """运行主要的分析流程"""
        try:
            # 设置引擎
            self.setup_engine()
            
            # 根据模式运行不同的分析
            if self.args.mode == 'basic':
                self.run_basic_profiling()
            elif self.args.mode == 'detailed':
                self.run_detailed_profiling()
            elif self.args.mode == 'continuous':
                self.run_continuous_monitoring()
            else:
                logger.error(f"Unknown mode: {self.args.mode}")
                return
                
            # 导出结果
            self.export_results()
            
        except Exception as e:
            logger.error(f"Error during profiling: {e}")
            raise
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="vLLM GPU Performance Profiling Example")
    
    # vLLM引擎参数
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                       help="Model name or path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                       help="Maximum number of sequences")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    
    # 测试参数
    parser.add_argument("--num-prompts", type=int, default=50,
                       help="Number of prompts to process")
    parser.add_argument("--max-tokens-per-prompt", type=int, default=128,
                       help="Maximum tokens per prompt")
    parser.add_argument("--monitoring-rounds", type=int, default=3,
                       help="Number of monitoring rounds (for continuous mode)")
    
    # 监控参数
    parser.add_argument("--mode", choices=['basic', 'detailed', 'continuous'],
                       default='basic', help="Profiling mode")
    parser.add_argument("--enable-detailed-profiling", action='store_true',
                       help="Enable detailed torch profiling")
    parser.add_argument("--output-dir", type=str, default="./profiling_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    logger.info("Starting vLLM GPU Performance Profiling")
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Number of prompts: {args.num_prompts}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # 运行分析
    profiler = VLLMProfilingExample(args)
    profiler.run()
    
    logger.info("Profiling completed successfully!")


if __name__ == "__main__":
    main()
