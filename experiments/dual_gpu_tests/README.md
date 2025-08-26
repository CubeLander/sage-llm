# 双A100 GPU性能测试实验套件

这是一个完整的双A100 GPU性能测试实验套件，专门设计用于测量和分析大语言模型推理性能。

## 📋 实验概述

本实验套件包含以下测试：

1. **GPU间带宽测试** - 测量两块A100之间的实际带宽性能
2. **序列长度扩展性测试** - 分析推理时间与序列长度的关系
3. **Pipeline Parallel开销测试** - 测量PP层间通信开销和流水线气泡效应
4. **Tensor Parallel通信测试** - 分析TP模式下All-Reduce和All-Gather操作的性能

## 🚀 快速开始

### 环境要求

- 2x NVIDIA A100 GPU (建议80GB版本)
- CUDA 11.8+
- Python 3.8+
- PyTorch 2.0+
- vLLM框架

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm
pip install matplotlib seaborn pandas numpy
```

### 运行测试

最简单的方式是使用我们的一键启动脚本：

```bash
# 进入实验目录
cd experiments/dual_gpu_tests

# 给启动脚本添加执行权限
chmod +x quick_start.sh

# 运行快速测试（推荐首次使用）
./quick_start.sh quick
```

### 可用的测试命令

```bash
# GPU带宽测试
./quick_start.sh bandwidth

# 序列长度扩展性测试
./quick_start.sh sequence --model facebook/opt-1.3b

# Pipeline Parallel开销测试
./quick_start.sh pp --model facebook/opt-1.3b

# Tensor Parallel通信测试
./quick_start.sh tp --model facebook/opt-1.3b

# 运行所有测试
./quick_start.sh comprehensive
```

## 📁 文件结构

```
dual_gpu_tests/
├── README.md                          # 本文件
├── config.ini                         # 配置文件
├── quick_start.sh                     # 一键启动脚本
├── run_comprehensive_test.py          # 综合测试控制器
├── gpu_bandwidth_test.py              # GPU带宽测试
├── sequence_length_test.py            # 序列长度测试
├── pp_overhead_test.py                # PP开销测试
├── tp_communication_test.py           # TP通信测试
├── visualization_generator.py         # 结果可视化
└── results/                           # 测试结果目录
    ├── test_YYYYMMDD_HHMMSS/         # 时间戳命名的结果目录
    │   ├── *.json                    # 原始数据
    │   ├── *.md                      # 分析报告
    │   └── charts/                   # 可视化图表
    └── ...
```

## 🔧 自定义配置

您可以通过修改 `config.ini` 文件来自定义测试参数：

```ini
[basic]
# 测试模型列表
models = [
    "facebook/opt-125m",
    "facebook/opt-1.3b",
]

[test_params]
# 序列长度范围
sequence_lengths = [128, 256, 512, 1024, 2048, 4096]

# 批次大小
batch_sizes = [1, 2, 4, 8, 16, 32]

# 测试重复次数
num_samples = 5
```

## 📊 结果解读

### GPU带宽测试结果

```json
{
  "peak_bandwidth_gb_s": 450.2,
  "latency_microseconds": 12.5,
  "efficiency_percent": 85.3
}
```

### 序列长度测试结果

测试会生成序列长度 vs 推理时间的关系图，帮助您：
- 识别性能拐点
- 分析内存瓶颈
- 优化批次大小

### PP开销测试结果

```json
{
  "pipeline_parallel_size": 2,
  "communication_overhead": 0.003,
  "pipeline_bubble_ratio": 0.15,
  "scaling_efficiency": 0.78
}
```

### TP通信测试结果

```json
{
  "all_reduce_bandwidth_gb_s": 320.5,
  "all_gather_bandwidth_gb_s": 280.1,
  "communication_efficiency": 0.82
}
```

## 📈 性能优化建议

根据测试结果，我们将提供以下优化建议：

### 小批量推理 (batch_size ≤ 4)
- 推荐使用单GPU (TP=1, PP=1)
- 通信开销可能超过并行收益

### 大批量推理 (batch_size ≥ 8)
- 推荐使用TP=2进行张量并行
- 对于超大模型考虑PP=2

### 长序列推理 (seq_len ≥ 2048)
- 优先考虑TP并行以减少内存使用
- 注意通信开销与计算的平衡

### 内存受限场景
- 使用PP=2将模型分布到两个GPU
- 接受一定的流水线气泡开销

## 🐛 故障排除

### 常见问题

**Q: 测试运行时出现CUDA out of memory错误**
A: 请降低`gpu_memory_utilization`参数或减小batch_size

**Q: 分布式初始化失败**
A: 请确保两块GPU之间有NVLink连接，并检查NCCL环境

**Q: 某些测试超时**
A: 可以在`config.ini`中增加`test_timeout`值，或使用`--quick-test`模式

### 获取帮助

```bash
# 显示帮助信息
./quick_start.sh --help

# 检查环境
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## 📄 许可证

本实验套件遵循与hotLLM项目相同的许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个实验套件！

---

**注意**: 这个实验套件是为双A100 GPU环境专门设计的。在其他硬件配置上运行可能需要调整参数。
