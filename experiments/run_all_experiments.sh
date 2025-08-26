#!/bin/bash

# A100 性能测试自动化执行脚本
# 
# 这个脚本会依次执行所有的性能测试实验，并生成综合报告

set -e  # 遇到错误时停止执行

echo "========================================"
echo "A100 双卡性能测试实验开始"
echo "时间: $(date)"
echo "========================================"

# 检查GPU可用性
echo "检查GPU环境..."
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

if [ $? -ne 0 ]; then
    echo "错误: GPU环境检查失败"
    exit 1
fi

# 创建结果目录
mkdir -p experiments/results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiments/results/batch_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "结果将保存到: $RESULTS_DIR"

# 设置默认模型
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-2-7b-hf"}
echo "使用模型: $MODEL_NAME"

# 实验1: GPU带宽测试 (最基础的测试，先做)
echo ""
echo "========================================"
echo "实验1: GPU间带宽测试"
echo "========================================"
python experiments/gpu_bandwidth_test.py > "$RESULTS_DIR/bandwidth_test.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 带宽测试完成"
else
    echo "✗ 带宽测试失败，查看 $RESULTS_DIR/bandwidth_test.log"
fi

# 实验2: 推理延迟与序列长度关系
echo ""
echo "========================================"
echo "实验2: 推理延迟与序列长度关系"
echo "========================================"
python experiments/latency_vs_sequence_length.py --model "$MODEL_NAME" --results-dir "$RESULTS_DIR" > "$RESULTS_DIR/latency_test.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 延迟测试完成"
else
    echo "✗ 延迟测试失败，查看 $RESULTS_DIR/latency_test.log"
fi

# 实验3: 并行化开销测试
echo ""
echo "========================================"
echo "实验3: 并行化开销测试"
echo "========================================"
python experiments/parallelism_overhead.py --model "$MODEL_NAME" --test-all > "$RESULTS_DIR/parallelism_test.log" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 并行化开销测试完成"
else
    echo "✗ 并行化开销测试失败，查看 $RESULTS_DIR/parallelism_test.log"
fi

# 移动结果文件到批次目录
echo ""
echo "整理结果文件..."
mv experiments/results/*.json "$RESULTS_DIR/" 2>/dev/null || true

# 生成综合报告
echo ""
echo "========================================"
echo "生成综合测试报告"
echo "========================================"

cat > "$RESULTS_DIR/experiment_summary.md" << EOF
# A100 双卡性能测试报告

**测试时间**: $(date)
**测试模型**: $MODEL_NAME
**GPU配置**: $(python -c "import torch; print(f'{torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else \"N/A\"}')")

## 实验概述

本次测试包含以下4个实验：

1. **GPU间带宽测试**: 测量A100 GPU之间的通信带宽
2. **推理延迟分析**: 分析推理时间随序列长度的变化
3. **并行化开销**: 测量TP和PP的通信开销
4. **综合性能评估**: 多维度性能分析

## 测试结果文件

- \`bandwidth_test_*.json\`: GPU带宽测试结果
- \`latency_analysis_*.json\`: 延迟分析结果  
- \`parallelism_overhead_*.json\`: 并行化开销结果

## 日志文件

- \`bandwidth_test.log\`: 带宽测试日志
- \`latency_test.log\`: 延迟测试日志
- \`parallelism_test.log\`: 并行化测试日志

## 快速查看结果

### 带宽测试结果
\`\`\`bash
python -c "
import json
import glob
files = glob.glob('$RESULTS_DIR/bandwidth_test_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
    if 'analysis' in data:
        analysis = data['analysis']
        if 'p2p_analysis' in analysis:
            print(f'峰值P2P带宽: {analysis[\"p2p_analysis\"][\"peak_bandwidth_gbps\"]:.2f} GB/s')
        if 'memory_analysis' in analysis:
            print(f'内存带宽: {analysis[\"memory_analysis\"][\"peak_memory_bandwidth_gbps\"]:.2f} GB/s')
else:
    print('带宽测试结果未找到')
"
\`\`\`

### 延迟测试结果摘要
\`\`\`bash
python -c "
import json
import glob
files = glob.glob('$RESULTS_DIR/latency_analysis_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
    for config, results in data.get('configurations', {}).items():
        if results:
            sample_key = list(results.keys())[0]
            throughput = results[sample_key]['throughput_tokens_per_sec']
            print(f'{config}: {throughput:.2f} tokens/sec')
else:
    print('延迟测试结果未找到')
"
\`\`\`

### 并行化效率
\`\`\`bash
python -c "
import json
import glob
import numpy as np
files = glob.glob('$RESULTS_DIR/parallelism_overhead_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
    analysis = data.get('analysis', {})
    if 'tensor_parallelism' in analysis and analysis['tensor_parallelism']:
        efficiencies = [v['efficiency_percent'] for v in analysis['tensor_parallelism'].values()]
        print(f'TP平均效率: {np.mean(efficiencies):.1f}%')
    if 'pipeline_parallelism' in analysis and analysis['pipeline_parallelism']:
        efficiencies = [v['efficiency_percent'] for v in analysis['pipeline_parallelism'].values()]
        print(f'PP平均效率: {np.mean(efficiencies):.1f}%')
else:
    print('并行化测试结果未找到')
"
\`\`\`

EOF

echo "✓ 综合报告已生成: $RESULTS_DIR/experiment_summary.md"

# 显示快速结果摘要
echo ""
echo "========================================"
echo "测试结果快速摘要"
echo "========================================"

# 检查并显示带宽结果
BANDWIDTH_FILES=(experiments/results/bandwidth_test_*.json)
if [ -f "${BANDWIDTH_FILES[0]}" ]; then
    echo "GPU带宽测试结果:"
    python -c "
import json
import glob
files = glob.glob('experiments/results/bandwidth_test_*.json')
if files:
    with open(sorted(files)[-1]) as f:
        data = json.load(f)
    if 'analysis' in data and data['analysis']:
        analysis = data['analysis']
        if 'p2p_analysis' in analysis:
            print(f'  峰值P2P带宽: {analysis[\"p2p_analysis\"][\"peak_bandwidth_gbps\"]:.2f} GB/s')
        if 'memory_analysis' in analysis:
            print(f'  内存带宽: {analysis[\"memory_analysis\"][\"peak_memory_bandwidth_gbps\"]:.2f} GB/s')
"
else
    echo "  带宽测试结果文件未找到"
fi

# 检查并显示延迟结果
LATENCY_FILES=(experiments/results/latency_analysis_*.json)
if [ -f "${LATENCY_FILES[0]}" ]; then
    echo "推理性能结果:"
    python -c "
import json
import glob
files = glob.glob('experiments/results/latency_analysis_*.json')
if files:
    with open(sorted(files)[-1]) as f:
        data = json.load(f)
    for config, results in data.get('configurations', {}).items():
        if results:
            sample_key = list(results.keys())[0]
            throughput = results[sample_key]['throughput_tokens_per_sec']
            print(f'  {config}: {throughput:.2f} tokens/sec')
"
else
    echo "  延迟测试结果文件未找到"
fi

# 检查并显示并行化结果
PARALLEL_FILES=(experiments/results/parallelism_overhead_*.json)
if [ -f "${PARALLEL_FILES[0]}" ]; then
    echo "并行化效率结果:"
    python -c "
import json
import glob
import numpy as np
files = glob.glob('experiments/results/parallelism_overhead_*.json')
if files:
    with open(sorted(files)[-1]) as f:
        data = json.load(f)
    analysis = data.get('analysis', {})
    if 'tensor_parallelism' in analysis and analysis['tensor_parallelism']:
        efficiencies = [v['efficiency_percent'] for v in analysis['tensor_parallelism'].values()]
        print(f'  TP平均效率: {np.mean(efficiencies):.1f}%')
    if 'pipeline_parallelism' in analysis and analysis['pipeline_parallelism']:
        efficiencies = [v['efficiency_percent'] for v in analysis['pipeline_parallelism'].values()]
        print(f'  PP平均效率: {np.mean(efficiencies):.1f}%')
    
    if 'recommendations' in analysis:
        print('  推荐:')
        for rec in analysis['recommendations']:
            print(f'    - {rec}')
"
else
    echo "  并行化测试结果文件未找到"
fi

echo ""
echo "========================================"
echo "所有测试完成!"
echo "详细结果请查看: $RESULTS_DIR/"
echo "========================================"
