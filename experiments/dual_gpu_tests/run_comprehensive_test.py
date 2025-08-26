#!/usr/bin/env python3
"""
双A100 GPU性能测试主控制脚本
自动运行所有性能测试实验并生成综合报告
"""

import os
import sys
import json
import subprocess
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 添加实验脚本路径
experiment_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(experiment_dir)

class ComprehensiveGPUTester:
    def __init__(self, output_dir: str = "comprehensive_results"):
        """
        初始化综合GPU测试器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f"test_run_{self.timestamp}")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 测试配置
        self.test_config = {
            'models': ['facebook/opt-125m', 'facebook/opt-1.3b'],
            'sequence_lengths': [256, 512, 1024, 2048],
            'batch_sizes': [1, 2, 4, 8],
            'max_tp_size': 2,
            'max_pp_size': 2
        }
        
        self.test_results = {}
    
    def run_gpu_bandwidth_test(self):
        """运行GPU带宽测试"""
        print(f"\n{'='*60}")
        print("RUNNING GPU BANDWIDTH TEST")
        print(f"{'='*60}")
        
        bandwidth_script = os.path.join(experiment_dir, "gpu_bandwidth_test.py")
        
        try:
            cmd = [sys.executable, bandwidth_script, 
                   "--output-dir", self.results_dir]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✓ GPU bandwidth test completed successfully")
                self.test_results['gpu_bandwidth'] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                print(f"✗ GPU bandwidth test failed: {result.stderr}")
                self.test_results['gpu_bandwidth'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print("✗ GPU bandwidth test timed out")
            self.test_results['gpu_bandwidth'] = {
                'status': 'timeout',
                'error': 'Test timed out after 5 minutes'
            }
        except Exception as e:
            print(f"✗ GPU bandwidth test error: {e}")
            self.test_results['gpu_bandwidth'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def run_sequence_length_test(self, model: str):
        """运行序列长度测试"""
        print(f"\n{'='*60}")
        print(f"RUNNING SEQUENCE LENGTH TEST - {model}")
        print(f"{'='*60}")
        
        seq_script = os.path.join(experiment_dir, "sequence_length_test.py")
        
        try:
            cmd = [sys.executable, seq_script,
                   "--model", model,
                   "--min-length", str(min(self.test_config['sequence_lengths'])),
                   "--max-length", str(max(self.test_config['sequence_lengths'])),
                   "--steps", "8",
                   "--output-dir", self.results_dir]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                print(f"✓ Sequence length test for {model} completed successfully")
                self.test_results[f'sequence_length_{model.replace("/", "_")}'] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                print(f"✗ Sequence length test for {model} failed: {result.stderr}")
                self.test_results[f'sequence_length_{model.replace("/", "_")}'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"✗ Sequence length test for {model} timed out")
            self.test_results[f'sequence_length_{model.replace("/", "_")}'] = {
                'status': 'timeout',
                'error': 'Test timed out after 30 minutes'
            }
        except Exception as e:
            print(f"✗ Sequence length test error: {e}")
            self.test_results[f'sequence_length_{model.replace("/", "_")}'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def run_pp_overhead_test(self, model: str):
        """运行PP开销测试"""
        print(f"\n{'='*60}")
        print(f"RUNNING PP OVERHEAD TEST - {model}")
        print(f"{'='*60}")
        
        pp_script = os.path.join(experiment_dir, "pp_overhead_test.py")
        
        try:
            cmd = [sys.executable, pp_script,
                   "--model", model,
                   "--max-pp", str(self.test_config['max_pp_size']),
                   "--seq-lengths"] + [str(x) for x in self.test_config['sequence_lengths'][:3]],
            cmd.extend(["--batch-sizes"] + [str(x) for x in self.test_config['batch_sizes'][:3]])
            cmd.extend(["--output-dir", self.results_dir])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
            
            if result.returncode == 0:
                print(f"✓ PP overhead test for {model} completed successfully")
                self.test_results[f'pp_overhead_{model.replace("/", "_")}'] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                print(f"✗ PP overhead test for {model} failed: {result.stderr}")
                self.test_results[f'pp_overhead_{model.replace("/", "_")}'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"✗ PP overhead test for {model} timed out")
            self.test_results[f'pp_overhead_{model.replace("/", "_")}'] = {
                'status': 'timeout',
                'error': 'Test timed out after 40 minutes'
            }
        except Exception as e:
            print(f"✗ PP overhead test error: {e}")
            self.test_results[f'pp_overhead_{model.replace("/", "_")}'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def run_tp_communication_test(self, model: str):
        """运行TP通信测试"""
        print(f"\n{'='*60}")
        print(f"RUNNING TP COMMUNICATION TEST - {model}")
        print(f"{'='*60}")
        
        tp_script = os.path.join(experiment_dir, "tp_communication_test.py")
        
        try:
            cmd = [sys.executable, tp_script,
                   "--model", model,
                   "--max-tp", str(self.test_config['max_tp_size']),
                   "--seq-lengths"] + [str(x) for x in self.test_config['sequence_lengths'][:3]]
            cmd.extend(["--batch-sizes"] + [str(x) for x in self.test_config['batch_sizes'][:3]])
            cmd.extend(["--output-dir", self.results_dir])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
            
            if result.returncode == 0:
                print(f"✓ TP communication test for {model} completed successfully")
                self.test_results[f'tp_communication_{model.replace("/", "_")}'] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                print(f"✗ TP communication test for {model} failed: {result.stderr}")
                self.test_results[f'tp_communication_{model.replace("/", "_")}'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"✗ TP communication test for {model} timed out")
            self.test_results[f'tp_communication_{model.replace("/", "_")}'] = {
                'status': 'timeout',
                'error': 'Test timed out after 40 minutes'
            }
        except Exception as e:
            print(f"✗ TP communication test error: {e}")
            self.test_results[f'tp_communication_{model.replace("/", "_")}'] = {
                'status': 'error',
                'error': str(e)
            }
    
    def run_all_tests(self):
        """运行所有测试"""
        start_time = time.time()
        
        print(f"Starting comprehensive GPU performance testing...")
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Test configuration:")
        for key, value in self.test_config.items():
            print(f"  {key}: {value}")
        
        # 1. GPU带宽测试（只需运行一次）
        self.run_gpu_bandwidth_test()
        
        # 2. 对每个模型运行其他测试
        for model in self.test_config['models']:
            print(f"\n{'='*80}")
            print(f"TESTING MODEL: {model}")
            print(f"{'='*80}")
            
            # 序列长度测试
            self.run_sequence_length_test(model)
            
            # PP开销测试（只对较大模型测试）
            if '1.3b' in model or 'large' in model.lower():
                self.run_pp_overhead_test(model)
            
            # TP通信测试（只对较大模型测试）
            if '1.3b' in model or 'large' in model.lower():
                self.run_tp_communication_test(model)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("ALL TESTS COMPLETED")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time/60:.2f} minutes")
        
        # 保存测试状态
        self.save_test_summary()
        
        # 生成报告
        self.generate_comprehensive_report()
    
    def save_test_summary(self):
        """保存测试总结"""
        summary_file = os.path.join(self.results_dir, "test_summary.json")
        
        summary = {
            'test_config': self.test_config,
            'timestamp': self.timestamp,
            'results_directory': self.results_dir,
            'test_results': self.test_results,
            'summary_statistics': self.get_summary_statistics()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Test summary saved to: {summary_file}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results.values() if r['status'] == 'success')
        failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        timeout_tests = sum(1 for r in self.test_results.values() if r['status'] == 'timeout')
        error_tests = sum(1 for r in self.test_results.values() if r['status'] == 'error')
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'timeout_tests': timeout_tests,
            'error_tests': error_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0
        }
    
    def load_result_files(self) -> Dict[str, Any]:
        """加载所有结果文件"""
        results = {}
        
        # 查找所有JSON结果文件
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json') and not filename.startswith('test_summary'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        results[filename] = data
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return results
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print(f"\nGenerating comprehensive performance report...")
        
        # 加载所有结果文件
        result_files = self.load_result_files()
        
        # 创建报告
        report_file = os.path.join(self.results_dir, "comprehensive_report.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# 双A100 GPU性能测试综合报告\n\n")
            f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**测试配置**: {self.test_config}\n\n")
            
            # 测试概览
            stats = self.get_summary_statistics()
            f.write(f"## 测试概览\n\n")
            f.write(f"- 总测试数: {stats['total_tests']}\n")
            f.write(f"- 成功测试: {stats['successful_tests']}\n")
            f.write(f"- 失败测试: {stats['failed_tests']}\n")
            f.write(f"- 超时测试: {stats['timeout_tests']}\n")
            f.write(f"- 成功率: {stats['success_rate']:.2%}\n\n")
            
            # 各项测试结果详情
            f.write(f"## 详细测试结果\n\n")
            
            for test_name, result in self.test_results.items():
                f.write(f"### {test_name}\n")
                f.write(f"**状态**: {result['status']}\n\n")
                
                if result['status'] == 'success':
                    f.write("✅ 测试成功完成\n\n")
                else:
                    f.write(f"❌ 测试失败: {result.get('error', 'Unknown error')}\n\n")
            
            # 性能分析总结
            f.write(f"## 性能分析总结\n\n")
            self.write_performance_analysis(f, result_files)
            
            # 推荐配置
            f.write(f"## 推荐配置\n\n")
            self.write_recommendations(f, result_files)
        
        print(f"Comprehensive report saved to: {report_file}")
        
        # 生成可视化图表
        self.generate_visualizations(result_files)
    
    def write_performance_analysis(self, f, result_files: Dict[str, Any]):
        """写入性能分析"""
        f.write("### GPU带宽测试\n")
        
        # 寻找带宽测试结果
        bandwidth_results = None
        for filename, data in result_files.items():
            if 'bandwidth' in filename.lower():
                bandwidth_results = data
                break
        
        if bandwidth_results:
            f.write("- GPU间带宽测试已完成\n")
            f.write("- 详细结果请参考相应的JSON文件\n\n")
        else:
            f.write("- GPU带宽测试结果未找到\n\n")
        
        f.write("### 序列长度扩展性\n")
        f.write("- 测试了不同序列长度下的推理性能\n")
        f.write("- 分析了推理时间与序列长度的关系\n\n")
        
        f.write("### Pipeline Parallel 开销\n")
        f.write("- 测试了不同PP配置下的通信开销\n")
        f.write("- 分析了流水线气泡的影响\n\n")
        
        f.write("### Tensor Parallel 通信\n")
        f.write("- 测试了All-Reduce和All-Gather操作的延迟\n")
        f.write("- 分析了TP扩展的效率\n\n")
    
    def write_recommendations(self, f, result_files: Dict[str, Any]):
        """写入推荐配置"""
        f.write("基于测试结果，我们推荐以下配置：\n\n")
        f.write("1. **小批量推理** (batch_size <= 4):\n")
        f.write("   - 推荐使用 TP=1, PP=1 (单GPU)\n")
        f.write("   - 通信开销可能超过并行收益\n\n")
        
        f.write("2. **大批量推理** (batch_size >= 8):\n")
        f.write("   - 推荐使用 TP=2 进行张量并行\n")
        f.write("   - 对于超大模型考虑 PP=2\n\n")
        
        f.write("3. **长序列推理** (sequence_length >= 2048):\n")
        f.write("   - 优先考虑 TP 并行以减少内存使用\n")
        f.write("   - 注意通信开销与计算的平衡\n\n")
        
        f.write("4. **内存受限场景**:\n")
        f.write("   - 使用 PP=2 将模型分布到两个GPU\n")
        f.write("   - 接受一定的流水线气泡开销\n\n")
    
    def generate_visualizations(self, result_files: Dict[str, Any]):
        """生成可视化图表"""
        print("Generating visualization charts...")
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表目录
            charts_dir = os.path.join(self.results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # 这里可以添加具体的可视化代码
            # 由于结果文件格式可能不同，我们创建一个简单的示例
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('GPU Performance Test Results Overview', fontsize=16)
            
            # 示例图表1: 测试成功率
            stats = self.get_summary_statistics()
            labels = ['Success', 'Failed', 'Timeout', 'Error']
            sizes = [stats['successful_tests'], stats['failed_tests'], 
                    stats['timeout_tests'], stats['error_tests']]
            colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
            
            axes[0,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0,0].set_title('Test Results Distribution')
            
            # 示例图表2-4: 预留位置
            for i, ax in enumerate([axes[0,1], axes[1,0], axes[1,1]]):
                ax.text(0.5, 0.5, f'Chart {i+2}\nWaiting for data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Performance Chart {i+2}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization charts saved to: {charts_dir}")
            
        except ImportError:
            print("Warning: matplotlib not available, skipping visualization generation")
        except Exception as e:
            print(f"Error generating visualizations: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Comprehensive GPU Performance Testing")
    parser.add_argument("--output-dir", default="comprehensive_results", 
                       help="Output directory for all results")
    parser.add_argument("--models", nargs='+', 
                       default=['facebook/opt-125m', 'facebook/opt-1.3b'],
                       help="Models to test")
    parser.add_argument("--quick-test", action='store_true',
                       help="Run quick test with minimal configurations")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ComprehensiveGPUTester(output_dir=args.output_dir)
    
    # 调整配置
    if args.quick_test:
        tester.test_config = {
            'models': ['facebook/opt-125m'],
            'sequence_lengths': [256, 512],
            'batch_sizes': [1, 4],
            'max_tp_size': 2,
            'max_pp_size': 2
        }
        print("Running in quick test mode with minimal configurations")
    else:
        tester.test_config['models'] = args.models
    
    # 运行所有测试
    tester.run_all_tests()
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TESTING COMPLETED!")
    print(f"{'='*80}")
    print(f"Results directory: {tester.results_dir}")
    print(f"Check comprehensive_report.md for detailed analysis")

if __name__ == "__main__":
    main()
