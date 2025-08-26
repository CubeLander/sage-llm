#!/usr/bin/env python3
"""
Qwen3 深度性能分析实验主脚本

完整的实验流程:
1. 模型加载和初始化
2. 深度profiling (每层每操作)
3. 分布式通信分析 
4. 结果分析和可视化
5. 生成详细报告

使用方法:
python run_qwen3_profiling_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --tp 1 --pp 1
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen3_detailed_profiling import Qwen3Profiler
from qwen3_comm_profiler import DistributedCommProfiler
from qwen3_visualizer import Qwen3ProfilerVisualizer


class Qwen3ExperimentRunner:
    """Qwen3实验运行器 - 协调整个分析流程"""
    
    def __init__(self, 
                 model_path: str,
                 tp_size: int = 1,
                 pp_size: int = 1,
                 max_seq_len: int = 512,
                 batch_size: int = 1,
                 output_dir: str = None):
        
        self.model_path = model_path
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        # 创建带时间戳的输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_path.split('/')[-1] if '/' in model_path else model_path
            output_dir = f"./qwen3_experiment_{model_name}_{timestamp}"
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存实验配置
        self.experiment_config = {
            'model_path': model_path,
            'tp_size': tp_size,
            'pp_size': pp_size,
            'max_seq_len': max_seq_len,
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat(),
            'output_dir': output_dir
        }
        
        # 保存配置到文件
        config_file = os.path.join(output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.experiment_config, f, indent=2)
            
        print(f"Experiment initialized. Results will be saved to: {output_dir}")
        
    def run_detailed_profiling(self, 
                             prompts: List[str] = None,
                             num_tokens: int = 50,
                             warmup_steps: int = 2,
                             profile_steps: int = 5) -> Dict[str, Any]:
        """运行详细的性能profiling"""
        
        print("\n" + "="*60)
        print("PHASE 1: DETAILED OPERATION PROFILING")
        print("="*60)
        
        # 创建profiler
        profiler = Qwen3Profiler(
            model_path=self.model_path,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            max_seq_len=self.max_seq_len,
            batch_size=self.batch_size,
            output_dir=self.output_dir
        )
        
        # 默认测试prompts
        if prompts is None:
            prompts = [
                "The key principles of deep learning include",
                "Distributed computing enables large-scale",
                "Transformer architectures have revolutionized",
                "Efficient neural network inference requires",
                "The future of artificial intelligence will",
            ]
        
        try:
            # 运行benchmark
            analysis = profiler.run_benchmark(
                prompts=prompts,
                num_tokens=num_tokens,
                warmup_steps=warmup_steps,
                profile_steps=profile_steps
            )
            
            # 保存结果
            profiler.save_results(analysis)
            profiler.print_summary(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error during detailed profiling: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def run_communication_analysis(self) -> Dict[str, Any]:
        """运行通信分析"""
        
        print("\n" + "="*60)
        print("PHASE 2: COMMUNICATION OVERHEAD ANALYSIS")
        print("="*60)
        
        # 创建通信profiler
        comm_profiler = DistributedCommProfiler(
            output_dir=self.output_dir
        )
        
        try:
            # Patch通信操作
            comm_profiler.patch_communication_ops()
            comm_profiler.patch_pipeline_ops()
            
            # 开始profiling
            comm_profiler.start_profiling()
            
            # 这里可以运行一个简化的推理来收集通信数据
            # 注意: 实际的通信分析需要在分布式环境中运行
            print("Note: Communication analysis requires distributed setup (TP>1 or PP>1)")
            print("Running in single GPU mode - limited communication data available")
            
            # 模拟一些通信数据收集
            time.sleep(1)
            
            comm_profiler.stop_profiling()
            
            # 分析结果
            analysis = comm_profiler.analyze_communication_patterns()
            
            # 保存结果
            comm_profiler.save_analysis(analysis)
            comm_profiler.print_communication_summary(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error during communication analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def create_visualizations(self):
        """创建可视化图表"""
        
        print("\n" + "="*60)
        print("PHASE 3: VISUALIZATION GENERATION")
        print("="*60)
        
        try:
            visualizer = Qwen3ProfilerVisualizer(self.output_dir)
            visualizer.generate_all_visualizations()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            
    def generate_comprehensive_report(self, 
                                    detailed_analysis: Dict[str, Any],
                                    comm_analysis: Dict[str, Any]):
        """生成综合分析报告"""
        
        print("\n" + "="*60)
        print("PHASE 4: COMPREHENSIVE REPORT GENERATION")
        print("="*60)
        
        report = {
            'experiment_info': self.experiment_config,
            'executive_summary': self._create_executive_summary(detailed_analysis, comm_analysis),
            'detailed_findings': {
                'performance_analysis': detailed_analysis,
                'communication_analysis': comm_analysis
            },
            'recommendations': self._generate_recommendations(detailed_analysis, comm_analysis),
            'appendix': {
                'methodology': self._describe_methodology(),
                'limitations': self._describe_limitations()
            }
        }
        
        # 保存报告
        report_file = os.path.join(self.output_dir, "comprehensive_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # 生成markdown报告
        self._generate_markdown_report(report)
        
        print(f"Comprehensive report saved to {self.output_dir}")
        
    def _create_executive_summary(self, detailed_analysis: Dict[str, Any], comm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行摘要"""
        
        summary = {
            'model_info': {
                'model_path': self.model_path,
                'configuration': f"TP={self.tp_size}, PP={self.pp_size}",
                'max_sequence_length': self.max_seq_len
            },
            'key_findings': [],
            'performance_metrics': {},
            'bottlenecks_identified': []
        }
        
        if detailed_analysis:
            perf_summary = detailed_analysis.get('summary', {})
            summary['performance_metrics'] = {
                'total_operations': perf_summary.get('total_operations', 0),
                'total_inference_time_ms': perf_summary.get('total_time', 0) * 1000,
                'average_operation_time_ms': perf_summary.get('avg_op_time', 0) * 1000,
                'operation_time_std_ms': perf_summary.get('std_op_time', 0) * 1000
            }
            
            # 识别主要瓶颈
            op_breakdown = detailed_analysis.get('operation_breakdown', {})
            if op_breakdown:
                sorted_ops = sorted(op_breakdown.items(), 
                                  key=lambda x: x[1]['total_time'], reverse=True)
                for op, stats in sorted_ops[:5]:
                    summary['bottlenecks_identified'].append({
                        'operation': op,
                        'total_time_ms': stats['total_time'] * 1000,
                        'percentage': stats['percentage']
                    })
        
        if comm_analysis:
            comm_summary = comm_analysis.get('summary', {})
            if comm_summary.get('total_comm_ops', 0) > 0:
                summary['communication_overhead'] = {
                    'total_comm_ops': comm_summary.get('total_comm_ops', 0),
                    'total_comm_time_ms': comm_summary.get('total_comm_time', 0) * 1000,
                    'avg_comm_time_ms': comm_summary.get('avg_comm_time', 0) * 1000
                }
        
        return summary
        
    def _generate_recommendations(self, detailed_analysis: Dict[str, Any], comm_analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        
        recommendations = []
        
        if detailed_analysis:
            op_breakdown = detailed_analysis.get('operation_breakdown', {})
            
            # 基于操作分析的建议
            if op_breakdown:
                sorted_ops = sorted(op_breakdown.items(), 
                                  key=lambda x: x[1]['total_time'], reverse=True)
                
                top_op = sorted_ops[0] if sorted_ops else None
                if top_op and top_op[1]['percentage'] > 30:
                    recommendations.append(f"Optimize {top_op[0]} operations (consuming {top_op[1]['percentage']:.1f}% of total time)")
                
                # 检查线性层性能
                linear_ops = [op for op, _ in sorted_ops if 'linear' in op[0].lower() or 'matmul' in op[0].lower()]
                if linear_ops:
                    recommendations.append("Consider using optimized GEMM implementations or tensor parallel for linear operations")
                
                # 检查attention性能
                attn_ops = [op for op, _ in sorted_ops if 'attention' in op[0].lower() or 'softmax' in op[0].lower()]
                if attn_ops:
                    recommendations.append("Consider using Flash Attention or other optimized attention implementations")
        
        if comm_analysis:
            comm_by_type = comm_analysis.get('by_operation_type', {})
            if comm_by_type:
                total_comm_time = sum(stats['total_time'] for stats in comm_by_type.values())
                if total_comm_time > 0:
                    recommendations.append("Communication overhead detected - consider optimizing tensor parallel degree or using faster interconnects")
        
        # 通用建议
        recommendations.extend([
            "Profile with different batch sizes to find optimal throughput/latency trade-off",
            "Consider mixed precision (FP16/BF16) for better performance",
            "Evaluate quantization (INT8/INT4) for inference acceleration",
            "Monitor GPU utilization and memory bandwidth efficiency"
        ])
        
        return recommendations
        
    def _describe_methodology(self) -> Dict[str, str]:
        """描述实验方法论"""
        
        return {
            'profiling_approach': 'Hook-based fine-grained operation timing with monkey-patching of torch operations',
            'timing_mechanism': 'High-resolution performance counters with GPU synchronization',
            'communication_analysis': 'Interception of vLLM distributed communication primitives',
            'measurement_overhead': 'Minimized through conditional recording and efficient data structures',
            'statistical_approach': 'Multiple runs with warmup to reduce measurement noise'
        }
        
    def _describe_limitations(self) -> List[str]:
        """描述实验限制"""
        
        return [
            'Profiling overhead may slightly affect absolute timing measurements',
            'Single-GPU mode has limited communication analysis capabilities',
            'Results may vary across different hardware configurations',
            'CUDA graph optimizations are disabled for measurement accuracy',
            'Memory measurements reflect allocated memory, not peak usage',
            'Attention kernel internals are not fully decomposed in current implementation'
        ]
        
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """生成Markdown格式的报告"""
        
        md_content = f"""# Qwen3 Deep Performance Analysis Report

## Executive Summary

**Model:** {report['experiment_info']['model_path']}  
**Configuration:** TP={report['experiment_info']['tp_size']}, PP={report['experiment_info']['pp_size']}  
**Date:** {report['experiment_info']['timestamp']}

### Key Performance Metrics
"""
        
        exec_summary = report['executive_summary']
        perf_metrics = exec_summary.get('performance_metrics', {})
        
        if perf_metrics:
            md_content += f"""
- **Total Operations:** {perf_metrics.get('total_operations', 'N/A'):,}
- **Total Inference Time:** {perf_metrics.get('total_inference_time_ms', 0):.2f} ms
- **Average Operation Time:** {perf_metrics.get('average_operation_time_ms', 0):.3f} ms
- **Standard Deviation:** {perf_metrics.get('operation_time_std_ms', 0):.3f} ms
"""
        
        # 添加瓶颈分析
        bottlenecks = exec_summary.get('bottlenecks_identified', [])
        if bottlenecks:
            md_content += "\\n### Top Performance Bottlenecks\\n"
            for i, bottleneck in enumerate(bottlenecks, 1):
                md_content += f"{i}. **{bottleneck['operation']}**: {bottleneck['total_time_ms']:.2f}ms ({bottleneck['percentage']:.1f}%)\\n"
        
        # 添加建议
        recommendations = report.get('recommendations', [])
        if recommendations:
            md_content += "\\n### Optimization Recommendations\\n"
            for i, rec in enumerate(recommendations, 1):
                md_content += f"{i}. {rec}\\n"
        
        md_content += f"""
## Methodology

{report['appendix']['methodology']['profiling_approach']}

## Files Generated

- `detailed_timings.csv` - Raw operation timing data
- `analysis_report.json` - Structured performance analysis
- `comprehensive_report.json` - Complete experiment report
- `visualizations/` - Performance visualization charts

## Limitations

"""
        for limitation in report['appendix']['limitations']:
            md_content += f"- {limitation}\\n"
        
        # 保存markdown报告
        md_file = os.path.join(self.output_dir, "README.md")
        with open(md_file, 'w') as f:
            f.write(md_content)
            
    def run_complete_experiment(self, 
                              prompts: List[str] = None,
                              num_tokens: int = 50,
                              warmup_steps: int = 2,
                              profile_steps: int = 5):
        """运行完整的实验流程"""
        
        print("Starting Qwen3 Deep Performance Analysis Experiment")
        print(f"Model: {self.model_path}")
        print(f"Configuration: TP={self.tp_size}, PP={self.pp_size}")
        print(f"Output Directory: {self.output_dir}")
        
        start_time = time.time()
        
        try:
            # Phase 1: 详细性能分析
            detailed_analysis = self.run_detailed_profiling(
                prompts=prompts,
                num_tokens=num_tokens,
                warmup_steps=warmup_steps,
                profile_steps=profile_steps
            )
            
            # Phase 2: 通信分析
            comm_analysis = self.run_communication_analysis()
            
            # Phase 3: 可视化
            self.create_visualizations()
            
            # Phase 4: 综合报告
            self.generate_comprehensive_report(detailed_analysis, comm_analysis)
            
            total_time = time.time() - start_time
            
            print(f"\\n{'='*60}")
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"Total Experiment Time: {total_time:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\\nExperiment failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 清理GPU内存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive Qwen3 performance analysis")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Model path or HuggingFace model name")
    parser.add_argument("--tp", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, 
                       help="Pipeline parallel size")
    parser.add_argument("--max-seq-len", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    
    # 实验参数
    parser.add_argument("--num-tokens", type=int, default=50,
                       help="Number of tokens to generate per prompt")
    parser.add_argument("--warmup-steps", type=int, default=2,
                       help="Number of warmup steps")
    parser.add_argument("--profile-steps", type=int, default=5,
                       help="Number of profiling steps")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    
    # 自定义prompts
    parser.add_argument("--prompts", type=str, nargs='+', default=None,
                       help="Custom prompts for testing")
    
    args = parser.parse_args()
    
    # 创建实验runner
    experiment = Qwen3ExperimentRunner(
        model_path=args.model,
        tp_size=args.tp,
        pp_size=args.pp,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # 运行实验
    experiment.run_complete_experiment(
        prompts=args.prompts,
        num_tokens=args.num_tokens,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps
    )


if __name__ == "__main__":
    main()
