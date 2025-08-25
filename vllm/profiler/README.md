# vLLM GPU利用率和性能监控系统

## 概述

这是一个为vLLM设计的完整GPU利用率和性能监控系统，可以实时监控：

1. **GPU硬件层面**：GPU使用率、显存利用率、温度、功耗等
2. **CUDA内核层面**：每个CUDA内核的执行时间和资源占用
3. **PyTorch操作层面**：每个torch操作的延迟和内存使用
4. **模型层面**：每一层的性能指标和调用统计

## 快速开始

### 1. 环境变量方式（最简单）

```bash
# 启用基础监控
export VLLM_ENABLE_PROFILING=1
export VLLM_ENABLE_GPU_MONITORING=1
export VLLM_PROFILING_OUTPUT_DIR=./profiling_results

# 运行任何vLLM程序，监控会自动启用
python your_vllm_script.py
```

### 2. 代码集成方式

```python
from vllm import LLM, SamplingParams
from vllm.profiler.integration import enable_vllm_profiling, disable_vllm_profiling

# 启用监控
enable_vllm_profiling(
    enable_gpu_monitoring=True,      # 启用GPU硬件监控
    enable_layer_profiling=False,    # 启用详细层级监控（会影响性能）
    output_dir="./profiling_results" # 结果输出目录
)

# 正常使用vLLM
llm = LLM("facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["Hello world"], sampling_params)

# 禁用监控并导出结果
disable_vllm_profiling()
```

### 3. 上下文管理器方式

```python
from vllm import LLM, SamplingParams
from vllm.profiler.integration import VLLMProfilingContext

with VLLMProfilingContext(
    enable_gpu_monitoring=True,
    output_dir="./profiling_results"
):
    llm = LLM("facebook/opt-125m")
    outputs = llm.generate(["Hello world"], sampling_params)
    # 监控会在退出时自动停止和导出结果
```

### 4. 装饰器方式

```python
from vllm.profiler.integration import profile_vllm_function

@profile_vllm_function(
    enable_gpu_monitoring=True,
    output_dir="./profiling_results"
)
def run_inference():
    from vllm import LLM, SamplingParams
    
    llm = LLM("facebook/opt-125m")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    outputs = llm.generate(["Hello world"], sampling_params)
    return outputs

# 函数执行时会自动监控
results = run_inference()
```

## 高级用法

### 1. 专用监控脚本

```bash
# 使用提供的示例脚本进行全面监控
python examples/gpu_profiling_example.py \
    --model facebook/opt-125m \
    --mode detailed \
    --num-prompts 100 \
    --enable-detailed-profiling \
    --output-dir ./detailed_profiling
```

### 2. 直接使用监控器

```python
from vllm.profiler.vllm_monitor import VLLMPerformanceMonitor
from vllm import LLM

llm = LLM("facebook/opt-125m")

# 获取ModelRunner（需要根据vLLM版本调整）
model_runner = llm.llm_engine.workers[0].model_runner

# 创建监控器
monitor = VLLMPerformanceMonitor(
    model_runner=model_runner,
    enable_realtime_monitoring=True,
    enable_detailed_profiling=False
)

# 启动监控
monitor.start_monitoring()

# 监控推理过程
for i in range(100):
    with monitor.profile_inference_step(batch_size=1, num_tokens=100):
        outputs = llm.generate(["Hello world"], sampling_params)
    
    # 每10步输出一次摘要
    if i % 10 == 0:
        summary = monitor.get_performance_summary(last_n_steps=10)
        print(f"Last 10 steps avg duration: {summary.get('avg_duration_ms', 0):.1f}ms")

# 停止监控并导出结果
monitor.stop_monitoring()
monitor.export_performance_report("detailed_report.json")
```

### 3. GPU硬件监控

```python
from vllm.profiler.gpu_monitor import GPUHardwareMonitor

# 创建GPU硬件监控器
gpu_monitor = GPUHardwareMonitor(device_id=0, sample_interval=0.1)
gpu_monitor.start()

# 运行你的推理代码
# ...

# 获取监控结果
current_metrics = gpu_monitor.get_current_metrics()
print(f"GPU Utilization: {current_metrics.gpu_utilization:.1f}%")
print(f"Memory Utilization: {current_metrics.memory_utilization:.1f}%")
print(f"Temperature: {current_metrics.temperature}°C")

# 获取历史数据
history = gpu_monitor.get_metrics_history(duration_sec=60.0)  # 最近1分钟
gpu_monitor.stop()
```

## 监控结果分析

### 1. 性能摘要

监控系统会生成包含以下信息的性能摘要：

```json
{
  "total_steps": 1000,
  "avg_duration_ms": 45.2,
  "min_duration_ms": 32.1,
  "max_duration_ms": 78.5,
  "avg_memory_mb": 1024.5,
  "max_memory_mb": 1536.2,
  "layer_performance": {
    "transformer.layers.0.attention": {
      "avg_duration_ms": 12.3,
      "call_count": 1000
    }
  }
}
```

### 2. GPU监控历史

```json
{
  "gpu_history": [
    {
      "timestamp": 1703123456.789,
      "gpu_utilization": 85.2,
      "memory_utilization": 67.8,
      "temperature": 72.0,
      "power_draw": 180.5
    }
  ]
}
```

### 3. 详细步骤分析

每个推理步骤的详细信息：

```json
{
  "detailed_steps": [
    {
      "step_id": 0,
      "batch_size": 8,
      "duration_ms": 45.2,
      "memory_used_mb": 256.5,
      "layer_timings": {
        "transformer.layers.0": {
          "duration_ms": 12.3,
          "memory_delta_mb": 64.2,
          "input_shapes": [[8, 512, 768]],
          "output_shapes": [[8, 512, 768]]
        }
      },
      "gpu_metrics": {
        "start_utilization": 80.5,
        "end_utilization": 95.2
      }
    }
  ]
}
```

## 配置选项

### 监控配置

- `enable_gpu_monitoring`: 启用GPU硬件监控（推荐始终开启）
- `enable_layer_profiling`: 启用详细的层级profiling（会影响性能，仅用于深度分析）
- `sample_interval`: GPU监控采样间隔（秒），默认0.1秒
- `output_dir`: 结果输出目录
- `auto_export_interval`: 自动导出间隔（秒），None表示不自动导出

### 环境变量

- `VLLM_ENABLE_PROFILING`: 启用监控（1/true/yes）
- `VLLM_ENABLE_GPU_MONITORING`: 启用GPU硬件监控
- `VLLM_ENABLE_LAYER_PROFILING`: 启用详细层级监控
- `VLLM_PROFILING_OUTPUT_DIR`: 输出目录路径

## 性能影响

### GPU硬件监控
- **性能影响**: 极小（<1%）
- **推荐用途**: 生产环境实时监控

### 层级profiling
- **性能影响**: 较大（5-15%）
- **推荐用途**: 开发和调试阶段的性能分析

### 详细torch profiling
- **性能影响**: 很大（20-50%）
- **推荐用途**: 深度性能分析和优化

## 故障排除

### 1. nvidia-smi不可用
```python
# 确保CUDA环境正确安装
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True)
if result.returncode != 0:
    print("nvidia-smi not available")
```

### 2. 内存不足
```python
# 减少监控历史缓存大小
monitor = VLLMPerformanceMonitor(
    model_runner=model_runner,
    max_history_size=1000  # 减少到1000个样本
)
```

### 3. 权限问题
```bash
# 确保有足够权限访问GPU信息
sudo nvidia-smi
```

### 4. ROCm GPU支持
```python
# 对于AMD GPU，确保rocm-smi可用
import subprocess
result = subprocess.run(["rocm-smi"], capture_output=True)
```

## 最佳实践

### 1. 生产环境监控
- 仅启用GPU硬件监控
- 设置合理的采样间隔（0.1-1.0秒）
- 定期导出和清理历史数据

### 2. 性能调试
- 启用详细层级监控
- 使用小规模数据进行测试
- 关注内存和计算时间的热点层

### 3. 批量处理监控
- 监控不同batch size的性能表现
- 分析GPU利用率随负载变化的趋势
- 记录内存使用模式

### 4. 模型对比
- 使用相同监控配置对比不同模型
- 关注每token的平均延迟
- 分析GPU利用率的差异

## 扩展和定制

### 1. 自定义监控指标

```python
from vllm.profiler.gpu_monitor import GPUMetrics

class CustomGPUMetrics(GPUMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metric = self._calculate_custom_metric()
    
    def _calculate_custom_metric(self):
        # 自定义指标计算逻辑
        return 0.0
```

### 2. 集成到现有监控系统

```python
# 将监控结果发送到外部系统
def send_to_monitoring_system(metrics):
    import requests
    requests.post("http://your-monitoring-system/api/metrics", json=metrics)

# 在监控器中添加回调
monitor.on_step_completed = send_to_monitoring_system
```

通过这个系统，你可以全面了解vLLM在运行时的GPU利用率和每一层的性能表现，从而进行针对性的优化。
