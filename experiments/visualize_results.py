#!/usr/bin/env python3
"""
A100性能测试结果可视化脚本

生成性能测试的可视化图表和分析报告。
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from datetime import datetime

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_latest_results(results_dir: str) -> Dict[str, Any]:
    """加载最新的测试结果"""
    results = {}
    
    # 加载带宽测试结果
    bandwidth_files = glob.glob(os.path.join(results_dir, "bandwidth_test_*.json"))
    if bandwidth_files:
        with open(sorted(bandwidth_files)[-1], 'r') as f:
            results['bandwidth'] = json.load(f)
    
    # 加载延迟分析结果
    latency_files = glob.glob(os.path.join(results_dir, "latency_analysis_*.json"))
    if latency_files:
        with open(sorted(latency_files)[-1], 'r') as f:
            results['latency'] = json.load(f)
    
    # 加载并行化开销结果
    parallel_files = glob.glob(os.path.join(results_dir, "parallelism_overhead_*.json"))
    if parallel_files:
        with open(sorted(parallel_files)[-1], 'r') as f:
            results['parallelism'] = json.load(f)
    
    return results

def plot_bandwidth_results(bandwidth_data: Dict, save_dir: str):
    """绘制带宽测试结果"""
    if not bandwidth_data:
        print("No bandwidth data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GPU Bandwidth Analysis', fontsize=16, fontweight='bold')
    
    # P2P带宽 vs 数据大小
    if 'p2p_bandwidth' in bandwidth_data and bandwidth_data['p2p_bandwidth']:
        p2p_data = bandwidth_data['p2p_bandwidth']
        sizes = []
        bandwidths_0_to_1 = []
        bandwidths_1_to_0 = []
        
        for key, value in p2p_data.items():
            sizes.append(value['size_mb'])
            bandwidths_0_to_1.append(value['gpu0_to_gpu1_bandwidth_gbps'])
            bandwidths_1_to_0.append(value['gpu1_to_gpu0_bandwidth_gbps'])
        
        axes[0, 0].plot(sizes, bandwidths_0_to_1, 'o-', label='GPU0 → GPU1', linewidth=2, markersize=6)
        axes[0, 0].plot(sizes, bandwidths_1_to_0, 's-', label='GPU1 → GPU0', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Data Size (MB)')
        axes[0, 0].set_ylabel('Bandwidth (GB/s)')
        axes[0, 0].set_title('P2P Bandwidth vs Data Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log')
    
    # 内存带宽
    if 'single_gpu_memory' in bandwidth_data and bandwidth_data['single_gpu_memory']:
        mem_data = bandwidth_data['single_gpu_memory']
        sizes = []
        mem_bandwidths = []
        
        for key, value in mem_data.items():
            sizes.append(value['size_mb'])
            mem_bandwidths.append(value['memory_bandwidth_gbps'])
        
        axes[0, 1].plot(sizes, mem_bandwidths, 'o-', color='green', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Data Size (MB)')
        axes[0, 1].set_ylabel('Memory Bandwidth (GB/s)')
        axes[0, 1].set_title('GPU Memory Bandwidth')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log')
    
    # All-Reduce性能
    if 'all_reduce_bandwidth' in bandwidth_data and bandwidth_data['all_reduce_bandwidth']:
        allreduce_data = bandwidth_data['all_reduce_bandwidth']
        sizes = []
        allreduce_bandwidths = []
        
        for key, value in allreduce_data.items():
            sizes.append(value['size_mb'])
            allreduce_bandwidths.append(value['effective_bandwidth_gbps'])
        
        axes[1, 0].plot(sizes, allreduce_bandwidths, 'o-', color='red', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Data Size (MB)')
        axes[1, 0].set_ylabel('Effective Bandwidth (GB/s)')
        axes[1, 0].set_title('All-Reduce Bandwidth')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')
    
    # 带宽对比柱状图
    if 'analysis' in bandwidth_data and bandwidth_data['analysis']:
        analysis = bandwidth_data['analysis']
        categories = []
        values = []
        
        if 'p2p_analysis' in analysis:
            categories.append('Peak P2P')
            values.append(analysis['p2p_analysis']['peak_bandwidth_gbps'])
        
        if 'memory_analysis' in analysis:
            categories.append('Memory')
            values.append(analysis['memory_analysis']['peak_memory_bandwidth_gbps'])
        
        if categories and values:
            bars = axes[1, 1].bar(categories, values, color=['skyblue', 'lightgreen'])
            axes[1, 1].set_ylabel('Bandwidth (GB/s)')
            axes[1, 1].set_title('Peak Bandwidth Comparison')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bandwidth_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bandwidth analysis plot saved to {save_dir}/bandwidth_analysis.png")

def plot_latency_results(latency_data: Dict, save_dir: str):
    """绘制延迟分析结果"""
    if not latency_data or 'configurations' not in latency_data:
        print("No latency data available")
        return
    
    configurations = latency_data['configurations']
    
    # 提取序列长度vs延迟的数据
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Inference Latency Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    config_names = list(configurations.keys())
    
    # 序列长度 vs 延迟
    for i, (config_name, config_data) in enumerate(configurations.items()):
        if not config_data:
            continue
            
        seq_lengths = []
        latencies = []
        
        for key, value in config_data.items():
            if 'batch1_out100' in key:  # 固定batch=1, output=100的情况
                seq_lengths.append(value['sequence_length'])
                latencies.append(value['avg_latency'])
        
        if seq_lengths:
            sorted_data = sorted(zip(seq_lengths, latencies))
            seq_lengths, latencies = zip(*sorted_data)
            
            axes[0, 0].plot(seq_lengths, latencies, 'o-', 
                          color=colors[i % len(colors)], 
                          label=config_name, linewidth=2, markersize=6)
    
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Latency (seconds)')
    axes[0, 0].set_title('Latency vs Sequence Length (Batch=1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # 吞吐量对比
    for i, (config_name, config_data) in enumerate(configurations.items()):
        if not config_data:
            continue
            
        seq_lengths = []
        throughputs = []
        
        for key, value in config_data.items():
            if 'batch4_out100' in key:  # 固定batch=4, output=100的情况
                seq_lengths.append(value['sequence_length'])
                throughputs.append(value['throughput_tokens_per_sec'])
        
        if seq_lengths:
            sorted_data = sorted(zip(seq_lengths, throughputs))
            seq_lengths, throughputs = zip(*sorted_data)
            
            axes[0, 1].plot(seq_lengths, throughputs, 'o-', 
                          color=colors[i % len(colors)], 
                          label=config_name, linewidth=2, markersize=6)
    
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Throughput (tokens/sec)')
    axes[0, 1].set_title('Throughput vs Sequence Length (Batch=4)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # 批处理大小的影响
    fixed_seq_len = 1024
    for i, (config_name, config_data) in enumerate(configurations.items()):
        if not config_data:
            continue
            
        batch_sizes = []
        throughputs = []
        
        for key, value in config_data.items():
            if f'seq{fixed_seq_len}_' in key and 'out100' in key:
                batch_sizes.append(value['batch_size'])
                throughputs.append(value['throughput_tokens_per_sec'])
        
        if batch_sizes:
            sorted_data = sorted(zip(batch_sizes, throughputs))
            batch_sizes, throughputs = zip(*sorted_data)
            
            axes[1, 0].plot(batch_sizes, throughputs, 'o-', 
                          color=colors[i % len(colors)], 
                          label=config_name, linewidth=2, markersize=6)
    
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Throughput (tokens/sec)')
    axes[1, 0].set_title(f'Throughput vs Batch Size (Seq={fixed_seq_len})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 内存使用情况
    for i, (config_name, config_data) in enumerate(configurations.items()):
        if not config_data:
            continue
            
        seq_lengths = []
        memory_usage = []
        
        for key, value in config_data.items():
            if 'batch1_out100' in key:
                seq_lengths.append(value['sequence_length'])
                memory_usage.append(value['avg_memory_gb'])
        
        if seq_lengths:
            sorted_data = sorted(zip(seq_lengths, memory_usage))
            seq_lengths, memory_usage = zip(*sorted_data)
            
            axes[1, 1].plot(seq_lengths, memory_usage, 'o-', 
                          color=colors[i % len(colors)], 
                          label=config_name, linewidth=2, markersize=6)
    
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('Memory Usage (GB)')
    axes[1, 1].set_title('Memory Usage vs Sequence Length')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latency analysis plot saved to {save_dir}/latency_analysis.png")

def plot_parallelism_results(parallel_data: Dict, save_dir: str):
    """绘制并行化结果"""
    if not parallel_data:
        print("No parallelism data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parallelism Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # TP效率分析
    if 'analysis' in parallel_data and 'tensor_parallelism' in parallel_data['analysis']:
        tp_analysis = parallel_data['analysis']['tensor_parallelism']
        
        test_cases = list(tp_analysis.keys())
        speedups = [tp_analysis[case]['speedup'] for case in test_cases]
        efficiencies = [tp_analysis[case]['efficiency_percent'] for case in test_cases]
        
        x_pos = np.arange(len(test_cases))
        
        bars1 = axes[0, 0].bar(x_pos - 0.2, speedups, 0.4, label='Speedup', color='skyblue')
        axes[0, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Ideal (2x)')
        axes[0, 0].set_xlabel('Test Cases')
        axes[0, 0].set_ylabel('Speedup')
        axes[0, 0].set_title('Tensor Parallelism Speedup')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([case.replace('_', '\n') for case in test_cases], rotation=0, fontsize=8)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, speedups):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 效率图
        bars2 = axes[0, 1].bar(x_pos, efficiencies, color='lightgreen')
        axes[0, 1].axhline(y=100.0, color='red', linestyle='--', alpha=0.7, label='Ideal (100%)')
        axes[0, 1].set_xlabel('Test Cases')
        axes[0, 1].set_ylabel('Efficiency (%)')
        axes[0, 1].set_title('Tensor Parallelism Efficiency')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([case.replace('_', '\n') for case in test_cases], rotation=0, fontsize=8)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, efficiencies):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # PP效率分析
    if 'analysis' in parallel_data and 'pipeline_parallelism' in parallel_data['analysis']:
        pp_analysis = parallel_data['analysis']['pipeline_parallelism']
        
        if pp_analysis:
            test_cases = list(pp_analysis.keys())
            speedups = [pp_analysis[case]['speedup'] for case in test_cases]
            efficiencies = [pp_analysis[case]['efficiency_percent'] for case in test_cases]
            
            x_pos = np.arange(len(test_cases))
            
            bars3 = axes[1, 0].bar(x_pos, speedups, color='orange')
            axes[1, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Ideal (2x)')
            axes[1, 0].set_xlabel('Test Cases')
            axes[1, 0].set_ylabel('Speedup')
            axes[1, 0].set_title('Pipeline Parallelism Speedup')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([case.replace('_', '\n') for case in test_cases], rotation=0, fontsize=8)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, speedups):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 对比不同并行策略的效率
    if 'analysis' in parallel_data:
        analysis = parallel_data['analysis']
        strategies = []
        avg_efficiencies = []
        
        if 'tensor_parallelism' in analysis and analysis['tensor_parallelism']:
            strategies.append('Tensor\nParallel')
            tp_effs = [v['efficiency_percent'] for v in analysis['tensor_parallelism'].values()]
            avg_efficiencies.append(np.mean(tp_effs))
        
        if 'pipeline_parallelism' in analysis and analysis['pipeline_parallelism']:
            strategies.append('Pipeline\nParallel')
            pp_effs = [v['efficiency_percent'] for v in analysis['pipeline_parallelism'].values()]
            avg_efficiencies.append(np.mean(pp_effs))
        
        if strategies:
            bars4 = axes[1, 1].bar(strategies, avg_efficiencies, 
                                  color=['lightcoral', 'lightblue'][:len(strategies)])
            axes[1, 1].axhline(y=100.0, color='red', linestyle='--', alpha=0.7, label='Ideal (100%)')
            axes[1, 1].set_ylabel('Average Efficiency (%)')
            axes[1, 1].set_title('Parallelism Strategy Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, avg_efficiencies):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parallelism_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parallelism analysis plot saved to {save_dir}/parallelism_analysis.png")

def generate_summary_report(results: Dict[str, Any], save_dir: str):
    """生成综合分析报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# A100 双卡性能测试综合报告

**生成时间**: {timestamp}

## 测试环境信息
"""
    
    # 添加GPU信息
    if 'latency' in results:
        latency_data = results['latency']
        report += f"""
- **GPU数量**: {latency_data.get('gpu_count', 'N/A')}
- **GPU型号**: {', '.join(latency_data.get('gpu_names', ['N/A']))}
- **测试模型**: {latency_data.get('model_name', 'N/A')}
"""

    # 带宽分析
    if 'bandwidth' in results:
        bandwidth_data = results['bandwidth']
        report += "\n## GPU带宽性能分析\n"
        
        if 'analysis' in bandwidth_data:
            analysis = bandwidth_data['analysis']
            
            if 'p2p_analysis' in analysis:
                p2p = analysis['p2p_analysis']
                report += f"""
### P2P通信带宽
- **峰值带宽**: {p2p['peak_bandwidth_gbps']:.2f} GB/s
- **平均带宽**: {p2p['avg_bandwidth_gbps']:.2f} GB/s
- **带宽范围**: {p2p['bandwidth_range']['min']:.2f} - {p2p['bandwidth_range']['max']:.2f} GB/s
"""
            
            if 'memory_analysis' in analysis:
                mem = analysis['memory_analysis']
                report += f"""
### GPU内存带宽
- **峰值内存带宽**: {mem['peak_memory_bandwidth_gbps']:.2f} GB/s
- **平均内存带宽**: {mem['avg_memory_bandwidth_gbps']:.2f} GB/s
"""
            
            if 'efficiency_analysis' in analysis:
                eff = analysis['efficiency_analysis']
                report += f"""
### 带宽效率分析
- **P2P vs 内存带宽比**: {eff['p2p_vs_memory_ratio']:.2f}
- **NVLink理论利用率**: {eff['theoretical_nvlink_utilization']*100:.1f}%
"""

    # 延迟分析
    if 'latency' in results:
        latency_data = results['latency']
        report += "\n## 推理延迟性能分析\n"
        
        if 'configurations' in latency_data:
            configs = latency_data['configurations']
            
            report += "### 不同配置性能对比\n"
            for config_name, config_data in configs.items():
                if config_data:
                    # 取一个代表性的测试用例
                    sample_key = 'seq1024_batch4_out100'
                    if sample_key in config_data:
                        sample = config_data[sample_key]
                        report += f"""
**{config_name}**:
- 吞吐量: {sample['throughput_tokens_per_sec']:.2f} tokens/sec
- 平均延迟: {sample['avg_latency']:.3f} seconds
- 内存使用: {sample['avg_memory_gb']:.2f} GB
"""

    # 并行化分析
    if 'parallelism' in results:
        parallel_data = results['parallelism']
        report += "\n## 并行化效率分析\n"
        
        if 'analysis' in parallel_data:
            analysis = parallel_data['analysis']
            
            if 'tensor_parallelism' in analysis and analysis['tensor_parallelism']:
                tp_data = list(analysis['tensor_parallelism'].values())
                avg_speedup = np.mean([d['speedup'] for d in tp_data])
                avg_efficiency = np.mean([d['efficiency_percent'] for d in tp_data])
                
                report += f"""
### Tensor Parallelism (TP)
- **平均加速比**: {avg_speedup:.2f}x
- **平均效率**: {avg_efficiency:.1f}%
- **扩展性评价**: {'优秀' if avg_efficiency > 75 else '良好' if avg_efficiency > 50 else '待改进'}
"""
            
            if 'pipeline_parallelism' in analysis and analysis['pipeline_parallelism']:
                pp_data = list(analysis['pipeline_parallelism'].values())
                avg_speedup = np.mean([d['speedup'] for d in pp_data])
                avg_efficiency = np.mean([d['efficiency_percent'] for d in pp_data])
                
                report += f"""
### Pipeline Parallelism (PP)
- **平均加速比**: {avg_speedup:.2f}x
- **平均效率**: {avg_efficiency:.1f}%
- **扩展性评价**: {'优秀' if avg_efficiency > 60 else '良好' if avg_efficiency > 40 else '待改进'}
"""
            
            if 'recommendations' in analysis:
                report += "\n### 优化建议\n"
                for i, rec in enumerate(analysis['recommendations'], 1):
                    report += f"{i}. {rec}\n"

    # 总结和建议
    report += "\n## 总结与建议\n"
    
    # 基于结果给出具体建议
    recommendations = []
    
    if 'bandwidth' in results and 'analysis' in results['bandwidth']:
        bandwidth_analysis = results['bandwidth']['analysis']
        if 'efficiency_analysis' in bandwidth_analysis:
            nvlink_util = bandwidth_analysis['efficiency_analysis']['theoretical_nvlink_utilization']
            if nvlink_util < 0.5:
                recommendations.append("NVLink利用率较低，考虑增加数据并行度或优化通信模式")
            else:
                recommendations.append("NVLink带宽利用良好，硬件配置合理")
    
    if 'parallelism' in results and 'analysis' in results['parallelism']:
        parallel_analysis = results['parallelism']['analysis']
        
        # TP建议
        if 'tensor_parallelism' in parallel_analysis and parallel_analysis['tensor_parallelism']:
            tp_data = list(parallel_analysis['tensor_parallelism'].values())
            avg_tp_eff = np.mean([d['efficiency_percent'] for d in tp_data])
            
            if avg_tp_eff > 70:
                recommendations.append("TP扩展效率良好，适合计算密集型任务")
            else:
                recommendations.append("TP扩展效率有提升空间，可能存在通信瓶颈")
        
        # PP建议  
        if 'pipeline_parallelism' in parallel_analysis and parallel_analysis['pipeline_parallelism']:
            pp_data = list(parallel_analysis['pipeline_parallelism'].values())
            avg_pp_eff = np.mean([d['efficiency_percent'] for d in pp_data])
            
            if avg_pp_eff > 50:
                recommendations.append("PP适合大批次推理场景")
            else:
                recommendations.append("PP开销较大，小批次场景建议使用TP")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
    else:
        report += "基于测试结果，当前配置运行良好。\n"
    
    # 保存报告
    report_path = os.path.join(save_dir, 'performance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Comprehensive report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Visualize A100 performance test results")
    parser.add_argument("--results-dir", default="experiments/results", 
                       help="Directory containing test results")
    parser.add_argument("--output-dir", default="experiments/results/plots",
                       help="Directory to save plots and reports")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading test results...")
    results = load_latest_results(args.results_dir)
    
    if not results:
        print("No test results found!")
        return
    
    print("Generating visualizations...")
    
    # 生成各种图表
    if 'bandwidth' in results:
        plot_bandwidth_results(results['bandwidth'], args.output_dir)
    
    if 'latency' in results:
        plot_latency_results(results['latency'], args.output_dir)
    
    if 'parallelism' in results:
        plot_parallelism_results(results['parallelism'], args.output_dir)
    
    # 生成综合报告
    print("Generating comprehensive report...")
    report_path = generate_summary_report(results, args.output_dir)
    
    print(f"""
=== Visualization Complete ===
Plots and reports saved to: {args.output_dir}

Generated files:
- bandwidth_analysis.png: GPU bandwidth analysis
- latency_analysis.png: Inference latency analysis  
- parallelism_analysis.png: Parallelism efficiency analysis
- performance_report.md: Comprehensive performance report

View the report with: cat {report_path}
""")

if __name__ == "__main__":
    main()
