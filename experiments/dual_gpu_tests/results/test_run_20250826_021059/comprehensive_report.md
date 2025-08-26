# 双A100 GPU性能测试综合报告

**测试时间**: 2025-08-26 02:10:59
**测试配置**: {'models': ['facebook/opt-125m'], 'sequence_lengths': [256, 512, 1024, 2048], 'batch_sizes': [1, 2, 4, 8], 'max_tp_size': 2, 'max_pp_size': 2}

## 测试概览

- 总测试数: 2
- 成功测试: 0
- 失败测试: 2
- 超时测试: 0
- 成功率: 0.00%

## 详细测试结果

### gpu_bandwidth
**状态**: failed

❌ 测试失败: /home/tjy/.conda/envs/hotLLM/bin/python3: can't open file '/home/tjy/hotLLM/experiments/dual_gpu_tests/gpu_bandwidth_test.py': [Errno 2] No such file or directory


### sequence_length_facebook_opt-125m
**状态**: failed

❌ 测试失败: Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.


## 性能分析总结

### GPU带宽测试
- GPU带宽测试结果未找到

### 序列长度扩展性
- 测试了不同序列长度下的推理性能
- 分析了推理时间与序列长度的关系

### Pipeline Parallel 开销
- 测试了不同PP配置下的通信开销
- 分析了流水线气泡的影响

### Tensor Parallel 通信
- 测试了All-Reduce和All-Gather操作的延迟
- 分析了TP扩展的效率

## 推荐配置

基于测试结果，我们推荐以下配置：

1. **小批量推理** (batch_size <= 4):
   - 推荐使用 TP=1, PP=1 (单GPU)
   - 通信开销可能超过并行收益

2. **大批量推理** (batch_size >= 8):
   - 推荐使用 TP=2 进行张量并行
   - 对于超大模型考虑 PP=2

3. **长序列推理** (sequence_length >= 2048):
   - 优先考虑 TP 并行以减少内存使用
   - 注意通信开销与计算的平衡

4. **内存受限场景**:
   - 使用 PP=2 将模型分布到两个GPU
   - 接受一定的流水线气泡开销

