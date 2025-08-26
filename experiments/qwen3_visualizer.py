#!/usr/bin/env python3
"""
Qwen3 性能分析结果可视化

生成各种可视化图表来分析Qwen3模型的性能特征:
1. 层级性能热力图
2. 操作时间分布直方图  
3. 通信vs计算时间对比
4. 内存使用模式
5. 性能瓶颈识别
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any
import argparse

class Qwen3ProfilerVisualizer:
    """Qwen3 profiler结果可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """加载profiling数据"""
        
        # 加载详细timing数据
        timing_file = os.path.join(self.results_dir, "detailed_timings.csv")
        if os.path.exists(timing_file):
            self.timing_df = pd.read_csv(timing_file)
        else:
            self.timing_df = None
            
        # 加载分析报告
        report_file = os.path.join(self.results_dir, "analysis_report.json")
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                self.analysis_report = json.load(f)
        else:
            self.analysis_report = None
            
        # 加载通信分析数据
        comm_file = os.path.join(self.results_dir, "comm_analysis.json")
        if os.path.exists(comm_file):
            with open(comm_file, 'r') as f:
                self.comm_analysis = json.load(f)
        else:
            self.comm_analysis = None
            
    def create_layer_performance_heatmap(self):
        """创建层级性能热力图"""
        if self.timing_df is None:
            return
            
        # 按层和操作聚合数据
        layer_op_times = self.timing_df.groupby(['layer_name', 'op_name'])['duration'].sum().unstack(fill_value=0)
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(layer_op_times, annot=False, cmap='YlOrRd', 
                   cbar_kws={'label': 'Total Time (seconds)'})
        plt.title('Qwen3 Layer Performance Heatmap\n(Time by Layer and Operation)', fontsize=14, fontweight='bold')
        plt.xlabel('Operation Type')
        plt.ylabel('Layer Name')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "layer_performance_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_operation_time_distribution(self):
        """创建操作时间分布图"""
        if self.analysis_report is None:
            return
            
        op_breakdown = self.analysis_report.get('operation_breakdown', {})
        if not op_breakdown:
            return
            
        # 准备数据
        ops = list(op_breakdown.keys())
        times = [op_breakdown[op]['total_time'] * 1000 for op in ops]  # 转换为毫秒
        percentages = [op_breakdown[op]['percentage'] for op in ops]
        
        # 排序
        sorted_data = sorted(zip(ops, times, percentages), key=lambda x: x[1], reverse=True)
        ops, times, percentages = zip(*sorted_data)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 操作时间条形图
        bars = ax1.barh(range(len(ops[:15])), times[:15])  # 只显示前15个
        ax1.set_yticks(range(len(ops[:15])))
        ax1.set_yticklabels([op[:30] for op in ops[:15]])  # 截断长名称
        ax1.set_xlabel('Total Time (ms)')
        ax1.set_title('Top 15 Operations by Total Time')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, time) in enumerate(zip(bars, times[:15])):
            ax1.text(bar.get_width() + max(times[:15]) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{time:.2f}ms', va='center', fontsize=9)
        
        # 百分比饼图
        top_10_ops = ops[:10]
        top_10_percentages = percentages[:10]
        others_percentage = sum(percentages[10:])
        
        if others_percentage > 0:
            pie_labels = list(top_10_ops) + ['Others']
            pie_sizes = list(top_10_percentages) + [others_percentage]
        else:
            pie_labels = top_10_ops
            pie_sizes = top_10_percentages
            
        wedges, texts, autotexts = ax2.pie(pie_sizes, labels=[label[:20] for label in pie_labels], 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Operation Time Distribution (Top 10 + Others)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "operation_time_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_compute_vs_communication_chart(self):
        """创建计算vs通信时间对比图"""
        if not self.comm_analysis or not self.analysis_report:
            return
            
        # 提取数据
        summary = self.analysis_report.get('summary', {})
        total_time = summary.get('total_time', 1)
        
        comm_summary = self.comm_analysis.get('summary', {})
        comm_time = comm_summary.get('total_comm_time', 0)
        compute_time = total_time - comm_time
        
        # 按操作类型分析通信
        comm_by_type = self.comm_analysis.get('by_operation_type', {})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 计算vs通信总体对比
        categories = ['Computation', 'Communication']
        times = [compute_time * 1000, comm_time * 1000]  # 转换为毫秒
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(categories, times, color=colors)
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Computation vs Communication Time')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值和百分比标签
        for bar, time in zip(bars, times):
            percentage = (time / (sum(times))) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times) * 0.01,
                    f'{time:.2f}ms\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # 通信操作类型分解
        if comm_by_type:
            comm_ops = list(comm_by_type.keys())
            comm_times = [comm_by_type[op]['total_time'] * 1000 for op in comm_ops]
            
            ax2.pie(comm_times, labels=comm_ops, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Communication Time Breakdown')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "compute_vs_communication.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_memory_usage_analysis(self):
        """创建内存使用分析图"""
        if self.timing_df is None:
            return
            
        # 计算内存变化
        self.timing_df['memory_delta'] = self.timing_df['memory_after'] - self.timing_df['memory_before']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 内存使用时间序列
        ax1.plot(range(len(self.timing_df)), self.timing_df['memory_after'] / (1024**3), marker='o', markersize=2)
        ax1.set_xlabel('Operation Index')
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title('GPU Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 内存增量分布
        memory_deltas_mb = self.timing_df['memory_delta'] / (1024**2)
        ax2.hist(memory_deltas_mb, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Memory Delta (MB)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Memory Changes per Operation')
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Mean: {memory_deltas_mb.mean():.2f} MB\nStd: {memory_deltas_mb.std():.2f} MB\nMax: {memory_deltas_mb.max():.2f} MB'
        ax2.text(0.7, 0.8, stats_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "memory_usage_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_performance_bottleneck_analysis(self):
        """创建性能瓶颈分析图"""
        if not self.analysis_report:
            return
            
        layer_breakdown = self.analysis_report.get('layer_breakdown', {})
        if not layer_breakdown:
            return
            
        # 准备数据
        layers = list(layer_breakdown.keys())
        avg_times = [layer_breakdown[layer]['avg_time'] * 1000 for layer in layers]
        total_times = [layer_breakdown[layer]['total_time'] * 1000 for layer in layers]
        counts = [layer_breakdown[layer]['count'] for layer in layers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 层平均时间散点图
        scatter = ax1.scatter(counts, avg_times, s=[t/10 for t in total_times], alpha=0.6, c=total_times, cmap='viridis')
        ax1.set_xlabel('Operation Count')
        ax1.set_ylabel('Average Time per Operation (ms)')
        ax1.set_title('Layer Performance: Average Time vs Operation Count\n(Bubble size = Total Time)')
        ax1.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax1, label='Total Time (ms)')
        
        # 瓶颈层识别 - 按总时间排序
        sorted_layers = sorted(zip(layers, total_times), key=lambda x: x[1], reverse=True)
        top_layers, top_times = zip(*sorted_layers[:15])
        
        bars = ax2.barh(range(len(top_layers)), top_times)
        ax2.set_yticks(range(len(top_layers)))
        ax2.set_yticklabels([layer[:40] for layer in top_layers])
        ax2.set_xlabel('Total Time (ms)')
        ax2.set_title('Top 15 Performance Bottlenecks (by Total Time)')
        ax2.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, time) in enumerate(zip(bars, top_times)):
            ax2.text(bar.get_width() + max(top_times) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{time:.2f}ms', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_bottlenecks.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_bandwidth_analysis_chart(self):
        """创建带宽分析图表"""
        if not self.comm_analysis:
            return
            
        bandwidth_analysis = self.comm_analysis.get('bandwidth_analysis', {})
        if not bandwidth_analysis:
            return
            
        by_size_range = bandwidth_analysis.get('by_size_range', {})
        if not by_size_range:
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        size_ranges = list(by_size_range.keys())
        avg_bandwidths = [by_size_range[sr]['avg_bandwidth'] for sr in size_ranges]
        max_bandwidths = [by_size_range[sr]['max_bandwidth'] for sr in size_ranges]
        
        x_pos = np.arange(len(size_ranges))
        width = 0.35
        
        ax.bar(x_pos - width/2, avg_bandwidths, width, label='Average Bandwidth', alpha=0.7)
        ax.bar(x_pos + width/2, max_bandwidths, width, label='Peak Bandwidth', alpha=0.7)
        
        ax.set_xlabel('Tensor Size Range')
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.set_title('Communication Bandwidth by Tensor Size Range')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(size_ranges, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "bandwidth_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_timeline_visualization(self):
        """创建操作时间线可视化"""
        if self.timing_df is None:
            return
            
        # 选择前100个操作进行可视化
        df_subset = self.timing_df.head(100).copy()
        df_subset['start_relative'] = df_subset['start_time'] - df_subset['start_time'].min()
        df_subset['end_relative'] = df_subset['start_relative'] + df_subset['duration']
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # 为不同操作类型分配颜色
        unique_ops = df_subset['op_name'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ops)))
        color_map = dict(zip(unique_ops, colors))
        
        for idx, row in df_subset.iterrows():
            ax.barh(idx, row['duration'] * 1000, left=row['start_relative'] * 1000, 
                   color=color_map[row['op_name']], alpha=0.7, height=0.8)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Operation Index')
        ax.set_title('Operation Timeline (First 100 Operations)')
        ax.grid(axis='x', alpha=0.3)
        
        # 创建图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[op], alpha=0.7, label=op[:20]) 
                          for op in list(unique_ops)[:10]]  # 只显示前10个操作类型
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "operation_timeline.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_summary_dashboard(self):
        """创建总结仪表板"""
        if not self.analysis_report:
            return
            
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        summary = self.analysis_report.get('summary', {})
        
        # 总体统计
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        stats_text = f"""
        QWEN3 PROFILING SUMMARY DASHBOARD
        
        Total Operations: {summary.get('total_operations', 0):,}
        Total Time: {summary.get('total_time', 0)*1000:.2f} ms
        Average Operation Time: {summary.get('avg_op_time', 0)*1000:.3f} ms
        Standard Deviation: {summary.get('std_op_time', 0)*1000:.3f} ms
        """
        
        if self.comm_analysis:
            comm_summary = self.comm_analysis.get('summary', {})
            comm_percentage = (comm_summary.get('total_comm_time', 0) / summary.get('total_time', 1)) * 100
            stats_text += f"\nCommunication Overhead: {comm_percentage:.1f}%"
            
        ax1.text(0.5, 0.5, stats_text, transform=ax1.transAxes, fontsize=14, 
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 其他小图表可以添加到其余的子图中...
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary_dashboard.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print(f"Loading data from {self.results_dir}")
        self.load_data()
        
        print("Generating visualizations...")
        
        visualizations = [
            ("Layer Performance Heatmap", self.create_layer_performance_heatmap),
            ("Operation Time Distribution", self.create_operation_time_distribution),
            ("Compute vs Communication", self.create_compute_vs_communication_chart),
            ("Memory Usage Analysis", self.create_memory_usage_analysis),
            ("Performance Bottlenecks", self.create_performance_bottleneck_analysis),
            ("Bandwidth Analysis", self.create_bandwidth_analysis_chart),
            ("Operation Timeline", self.create_timeline_visualization),
            ("Summary Dashboard", self.create_summary_dashboard)
        ]
        
        for name, func in visualizations:
            try:
                print(f"  Creating {name}...")
                func()
            except Exception as e:
                print(f"  Warning: Could not create {name}: {e}")
        
        print(f"All visualizations saved to {self.output_dir}")
        
    def create_interactive_dashboard(self):
        """创建交互式仪表板 (使用plotly)"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            # 这里可以创建交互式图表
            print("Interactive dashboard creation not implemented yet")
            
        except ImportError:
            print("Plotly not available for interactive dashboard")


def main():
    parser = argparse.ArgumentParser(description="Visualize Qwen3 profiling results")
    parser.add_argument("results_dir", help="Directory containing profiling results")
    parser.add_argument("--interactive", action="store_true", help="Create interactive dashboard")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return
        
    visualizer = Qwen3ProfilerVisualizer(args.results_dir)
    
    if args.interactive:
        visualizer.create_interactive_dashboard()
    else:
        visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
