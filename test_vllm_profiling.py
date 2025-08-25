#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
vLLM GPU性能监控快速测试脚本

这个脚本提供了一个简单的测试环境来验证监控系统是否正常工作。
运行这个脚本可以快速测试各种监控功能。
"""

import argparse
import json
import time
from pathlib import Path

# 导入监控模块
from vllm.profiler import (
    enable_vllm_profiling, disable_vllm_profiling, 
    VLLMProfilingContext, profile_vllm_function,
    GPUHardwareMonitor
)

def test_gpu_hardware_monitor():
    """测试GPU硬件监控器"""
    print("=== 测试GPU硬件监控器 ===")
    
    monitor = GPUHardwareMonitor(device_id=0, sample_interval=0.1)
    monitor.start()
    
    print("GPU监控已启动，收集5秒数据...")
    time.sleep(5)
    
    # 获取当前指标
    current = monitor.get_current_metrics()
    if current:
        print(f"当前GPU利用率: {current.gpu_utilization:.1f}%")
        print(f"当前显存利用率: {current.memory_utilization:.1f}%")
        print(f"显存使用量: {current.memory_used:.0f}MB / {current.memory_total:.0f}MB")
        if current.temperature:
            print(f"GPU温度: {current.temperature:.1f}°C")
        if current.power_draw:
            print(f"功耗: {current.power_draw:.1f}W")
    
    # 获取历史数据
    history = monitor.get_metrics_history()
    print(f"收集了 {len(history)} 个监控样本")
    
    if history:
        avg_util = sum(m.gpu_utilization for m in history) / len(history)
        print(f"平均GPU利用率: {avg_util:.1f}%")
    
    monitor.stop()
    print("GPU硬件监控测试完成\n")


def test_basic_integration():
    """测试基础集成功能"""
    print("=== 测试基础集成功能 ===")
    
    # 使用环境变量启用监控
    import os
    os.environ['VLLM_ENABLE_PROFILING'] = '1'
    os.environ['VLLM_ENABLE_GPU_MONITORING'] = '1'
    os.environ['VLLM_PROFILING_OUTPUT_DIR'] = './test_results'
    
    try:
        from vllm import LLM, SamplingParams
        
        # 创建LLM实例（这会自动启用监控）
        llm = LLM("facebook/opt-125m", max_num_seqs=2)
        
        # 运行简单推理
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        outputs = llm.generate(["Hello world, tell me about"], sampling_params)
        
        print(f"生成了 {len(outputs)} 个输出")
        print(f"输出示例: {outputs[0].outputs[0].text[:100]}...")
        
        # 禁用监控并导出结果
        disable_vllm_profiling()
        
        print("基础集成测试完成\n")
        
    except ImportError:
        print("vLLM未安装，跳过基础集成测试\n")
    except Exception as e:
        print(f"基础集成测试失败: {e}\n")


def test_context_manager():
    """测试上下文管理器"""
    print("=== 测试上下文管理器 ===")
    
    try:
        from vllm import LLM, SamplingParams
        
        with VLLMProfilingContext(
            enable_gpu_monitoring=True,
            enable_layer_profiling=False,
            output_dir="./test_context_results"
        ):
            llm = LLM("facebook/opt-125m", max_num_seqs=1)
            sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
            
            # 运行多次推理测试
            prompts = [
                "The future of AI is",
                "Technology will help us",
                "The most important thing is"
            ]
            
            for i, prompt in enumerate(prompts):
                print(f"处理提示词 {i+1}: {prompt}")
                outputs = llm.generate([prompt], sampling_params)
                print(f"生成完成: {len(outputs[0].outputs[0].text)} 字符")
        
        print("上下文管理器测试完成\n")
        
    except ImportError:
        print("vLLM未安装，跳过上下文管理器测试\n")
    except Exception as e:
        print(f"上下文管理器测试失败: {e}\n")


@profile_vllm_function(
    enable_gpu_monitoring=True,
    output_dir="./test_decorator_results"
)
def test_decorator_function():
    """测试装饰器功能"""
    print("=== 测试装饰器功能 ===")
    
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM("facebook/opt-125m", max_num_seqs=1)
        sampling_params = SamplingParams(temperature=0.5, max_tokens=40)
        
        prompts = ["Once upon a time", "In the beginning"]
        
        for prompt in prompts:
            print(f"装饰器监控中 - 处理: {prompt}")
            outputs = llm.generate([prompt], sampling_params)
            print(f"装饰器监控中 - 完成: {len(outputs)} 个输出")
        
        print("装饰器测试完成\n")
        return True
        
    except ImportError:
        print("vLLM未安装，跳过装饰器测试\n")
        return False
    except Exception as e:
        print(f"装饰器测试失败: {e}\n")
        return False


def test_manual_profiling():
    """测试手动启用/禁用监控"""
    print("=== 测试手动监控控制 ===")
    
    # 手动启用监控
    enable_vllm_profiling(
        enable_gpu_monitoring=True,
        enable_layer_profiling=False,
        output_dir="./test_manual_results"
    )
    
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM("facebook/opt-125m", max_num_seqs=1)
        sampling_params = SamplingParams(temperature=0.8, max_tokens=25)
        
        # 运行测试
        outputs = llm.generate(["Artificial intelligence"], sampling_params)
        print(f"手动监控 - 生成完成: {outputs[0].outputs[0].text[:50]}...")
        
        # 获取监控摘要
        from vllm.profiler.integration import get_monitoring_summary
        summary = get_monitoring_summary()
        print(f"监控摘要: {len(summary.get('monitors', {}))} 个监控器活跃")
        
    except ImportError:
        print("vLLM未安装，跳过手动监控测试")
    except Exception as e:
        print(f"手动监控测试失败: {e}")
    
    # 手动禁用监控
    disable_vllm_profiling()
    print("手动监控测试完成\n")


def generate_test_report():
    """生成测试报告"""
    print("=== 生成测试报告 ===")
    
    report = {
        "test_timestamp": time.time(),
        "test_results": {
            "gpu_hardware_monitor": "已测试",
            "basic_integration": "已测试", 
            "context_manager": "已测试",
            "decorator_function": "已测试",
            "manual_profiling": "已测试"
        },
        "output_directories": [
            "./test_results",
            "./test_context_results", 
            "./test_decorator_results",
            "./test_manual_results"
        ],
        "notes": "监控系统功能测试完成"
    }
    
    # 保存报告
    report_file = Path("./vllm_profiling_test_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"测试报告已保存到: {report_file}")
    

def main():
    parser = argparse.ArgumentParser(description="vLLM性能监控系统测试脚本")
    parser.add_argument("--test", choices=[
        'gpu', 'basic', 'context', 'decorator', 'manual', 'all'
    ], default='all', help="要运行的测试类型")
    parser.add_argument("--skip-vllm", action='store_true', 
                       help="跳过需要vLLM的测试（仅测试GPU监控）")
    
    args = parser.parse_args()
    
    print("vLLM GPU性能监控系统测试")
    print("=" * 40)
    
    if args.test in ['gpu', 'all']:
        test_gpu_hardware_monitor()
    
    if not args.skip_vllm:
        if args.test in ['basic', 'all']:
            test_basic_integration()
        
        if args.test in ['context', 'all']:
            test_context_manager()
        
        if args.test in ['decorator', 'all']:
            test_decorator_function()
        
        if args.test in ['manual', 'all']:
            test_manual_profiling()
    else:
        print("跳过vLLM相关测试")
    
    if args.test == 'all':
        generate_test_report()
    
    print("测试完成！")
    print("\n检查以下目录的监控结果:")
    print("- ./test_results")
    print("- ./test_context_results")
    print("- ./test_decorator_results")
    print("- ./test_manual_results")


if __name__ == "__main__":
    main()
