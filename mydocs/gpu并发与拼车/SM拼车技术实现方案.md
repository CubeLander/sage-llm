# SM拼车技术实现方案详解

## 1. 技术背景与目标

### 1.1 SM拼车技术概念
SM拼车技术(SM Carpooling)是指在GPU中低负载时，让多个小模型请求并行执行，充分利用GPU的SM(Streaming Multiprocessor)资源，避免资源闲置。

### 1.2 核心挑战
```
传统单模型执行：
┌─────────────────────────────────────────────────────────┐
│ GPU (108 SMs)                                           │
│ ┌─────────────┐                                        │
│ │小模型A(20SM)│        大量SM闲置                       │
│ └─────────────┘                                        │
└─────────────────────────────────────────────────────────┘
利用率: ~18%

期望的拼车执行：
┌─────────────────────────────────────────────────────────┐
│ GPU (108 SMs)                                           │
│ ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐│
│ │模型A    ││模型B    ││模型C    ││模型D    ││模型E    ││
│ │(20SM)   ││(18SM)   ││(22SM)   ││(25SM)   ││(23SM)   ││
│ └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘│
└─────────────────────────────────────────────────────────┘
利用率: ~100%
```

## 2. 实现方案分类

### 2.1 方案总览
```
SM拼车实现方案层次图：
┌─────────────────────────────────────────────────────────┐
│ 应用层拼车 (Application Level)                           │
├─────────────────────────────────────────────────────────┤
│ 框架层拼车 (Framework Level)                            │
├─────────────────────────────────────────────────────────┤
│ 运行时拼车 (Runtime Level)                              │
├─────────────────────────────────────────────────────────┤
│ 驱动层拼车 (Driver Level)                               │
├─────────────────────────────────────────────────────────┤
│ 硬件层拼车 (Hardware Level)                             │
└─────────────────────────────────────────────────────────┘
```

## 3. 详细实现方案

### 3.1 方案一：PyTorch Patch方案

#### 3.1.1 核心原理：动态SM分配与合并

**关键理念：软分割 + 动态合并**
```python
# 核心思想：不是硬切分GPU，而是动态调度SM使用
class DynamicSMScheduler:
    def __init__(self):
        self.total_sms = 108  # A100总SM数
        self.sm_allocation_map = {}  # SM分配映射
        self.pending_large_tasks = []  # 大任务等待队列
        self.small_tasks_pool = []     # 小任务池
        
    def submit_request(self, model_request):
        required_sms = self.estimate_sm_requirement(model_request)
        
        if required_sms > 64:  # 大模型请求
            return self.handle_large_model(model_request, required_sms)
        else:  # 小模型请求，可以拼车
            return self.handle_small_model(model_request, required_sms)
    
    def handle_large_model(self, model_request, required_sms):
        # 大模型需要等待所有小任务完成，然后独占GPU
        if self.has_running_small_tasks():
            # 等待小任务完成
            self.pending_large_tasks.append((model_request, required_sms))
            return None
        else:
            # 可以立即使用全部SM
            return self.execute_exclusive(model_request, self.total_sms)
    
    def handle_small_model(self, model_request, required_sms):
        # 检查是否有大任务等待
        if self.pending_large_tasks:
            # 有大任务等待，不接受新的小任务
            return self.execute_sync_fallback(model_request)
        
        # 可以进行拼车调度
        available_sms = self.get_available_sms()
        if available_sms >= required_sms:
            return self.execute_carpool(model_request, required_sms)
        else:
            # SM不够，加入小任务池等待
            self.small_tasks_pool.append((model_request, required_sms))
```

#### 3.1.2 PyTorch内部机制深入分析

**A. CUDA上下文与SM分配的关系**
```cpp
// PyTorch C++层面的实现原理
class DynamicSMAllocator {
private:
    // 关键：不是真正"分割"SM，而是控制kernel launch的参数
    struct SMPartition {
        int start_block_idx;    // 起始block索引  
        int max_blocks;         // 最大block数量
        cudaStream_t stream;    // 专用流
        float occupancy_limit;  // 占用率限制 (关键参数!)
    };
    
    std::vector<SMPartition> active_partitions;
    
public:
    // 核心：通过限制occupancy来"软分割"SM
    void launch_kernel_with_sm_limit(
        const void* kernel_func,
        dim3 grid_dim,
        dim3 block_dim,
        int target_sm_count,
        cudaStream_t stream
    ) {
        // 计算目标occupancy
        float target_occupancy = (float)target_sm_count / total_sms;
        
        // 关键技术：动态调整grid大小以限制SM使用
        int max_blocks_per_sm = get_max_blocks_per_sm(kernel_func, block_dim);
        int target_total_blocks = target_sm_count * max_blocks_per_sm;
        
        // 如果原始grid太大，进行分批处理
        if (grid_dim.x * grid_dim.y * grid_dim.z > target_total_blocks) {
            launch_in_batches(kernel_func, grid_dim, block_dim, 
                            target_total_blocks, stream);
        } else {
            // 直接launch，但设置occupancy限制
            cudaLaunchKernel(kernel_func, grid_dim, block_dim, 
                           args, shared_mem, stream);
        }
    }
    
    // 分批执行大kernel以控制SM使用
    void launch_in_batches(const void* kernel_func, dim3 original_grid,
                          dim3 block_dim, int max_blocks_per_batch,
                          cudaStream_t stream) {
        int total_blocks = original_grid.x * original_grid.y * original_grid.z;
        int num_batches = (total_blocks + max_blocks_per_batch - 1) / max_blocks_per_batch;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * max_blocks_per_batch;
            int batch_size = std::min(max_blocks_per_batch, 
                                    total_blocks - batch_start);
            
            // 计算当前批次的grid维度
            dim3 batch_grid = calculate_batch_grid(batch_size, original_grid);
            
            // Launch当前批次
            cudaLaunchKernel(kernel_func, batch_grid, block_dim,
                           adjusted_args(batch_start), shared_mem, stream);
        }
    }
};
```

**B. PyTorch张量操作的SM控制机制**
```python
# PyTorch层面的实现
class SMControlledTensor(torch.Tensor):
    def __new__(cls, data, sm_limit=None):
        tensor = torch.as_tensor(data)
        tensor.__class__ = cls
        tensor.sm_limit = sm_limit
        return tensor
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        
        # 在每个torch函数调用时注入SM控制
        if self.sm_limit is not None:
            # 设置当前操作的SM限制
            with sm_context(self.sm_limit):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

@contextmanager
def sm_context(sm_limit):
    # 设置当前线程的SM使用限制
    old_limit = get_current_sm_limit()
    set_current_sm_limit(sm_limit)
    try:
        yield
    finally:
        set_current_sm_limit(old_limit)

# 使用示例：模型的每一层都可以有不同的SM限制
class CarpoolModel(nn.Module):
    def __init__(self, base_model, sm_budget=20):
        super().__init__()
        self.base_model = base_model
        self.sm_budget = sm_budget
        
    def forward(self, x):
        # 关键：x变成了SM受限的张量
        x = SMControlledTensor(x, sm_limit=self.sm_budget)
        
        # 后续所有操作都会受到SM限制
        return self.base_model(x)
```

#### 3.1.2 具体实现细节

**A. PyTorch C++扩展方案**
```cpp
// pytorch_sm_carpool.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cudnn.h>

class SMResourceManager {
private:
    struct SMAllocation {
        int start_sm;
        int end_sm;
        cudaStream_t stream;
        cublasHandle_t cublas_handle;
        cudnnHandle_t cudnn_handle;
    };
    
    std::vector<SMAllocation> allocations;
    std::mutex allocation_mutex;
    
public:
    // 分配指定数量的SM
    SMAllocation* allocate_sms(int required_sms) {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        
        // 找到可用的SM范围
        int start_sm = find_available_sms(required_sms);
        if (start_sm == -1) return nullptr;
        
        // 创建专用的CUDA流和句柄
        SMAllocation* alloc = new SMAllocation;
        alloc->start_sm = start_sm;
        alloc->end_sm = start_sm + required_sms - 1;
        
        cudaStreamCreate(&alloc->stream);
        cublasCreate(&alloc->cublas_handle);
        cublasSetStream(alloc->cublas_handle, alloc->stream);
        cudnnCreate(&alloc->cudnn_handle);
        cudnnSetStream(alloc->cudnn_handle, alloc->stream);
        
        // 设置CUDA上下文的SM使用范围
        set_sm_mask(alloc->stream, start_sm, start_sm + required_sms - 1);
        
        allocations.push_back(*alloc);
        return alloc;
    }
    
    // 释放SM资源
    void deallocate_sms(SMAllocation* alloc) {
        cudaStreamDestroy(alloc->stream);
        cublasDestroy(alloc->cublas_handle);
        cudnnDestroy(alloc->cudnn_handle);
        
        // 从分配表中移除
        auto it = std::find(allocations.begin(), allocations.end(), *alloc);
        if (it != allocations.end()) {
            allocations.erase(it);
        }
        delete alloc;
    }
};

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SMResourceManager>(m, "SMResourceManager")
        .def(py::init<>())
        .def("allocate_sms", &SMResourceManager::allocate_sms)
        .def("deallocate_sms", &SMResourceManager::deallocate_sms);
}
```

**B. PyTorch Python层封装**
```python
# sm_carpool.py
import torch
import torch_sm_carpool  # 我们的C++扩展

class CarpoolModel(torch.nn.Module):
    def __init__(self, base_model, required_sms=None):
        super().__init__()
        self.base_model = base_model
        self.required_sms = required_sms or self._estimate_sm_requirement()
        self.sm_manager = torch_sm_carpool.SMResourceManager()
        
    def forward(self, x):
        # 申请SM资源
        allocation = self.sm_manager.allocate_sms(self.required_sms)
        if allocation is None:
            # 回退到正常执行
            return self.base_model(x)
            
        try:
            # 在分配的SM上执行
            with torch.cuda.stream(allocation.stream):
                result = self.base_model(x)
                # 确保计算完成
                torch.cuda.current_stream().wait_stream(allocation.stream)
            return result
        finally:
            # 释放资源
            self.sm_manager.deallocate_sms(allocation)
    
    def _estimate_sm_requirement(self):
        # 基于模型大小和复杂度估计SM需求
        param_count = sum(p.numel() for p in self.base_model.parameters())
        
        if param_count < 1e6:      # <1M参数
            return 8
        elif param_count < 10e6:   # <10M参数
            return 16
        elif param_count < 100e6:  # <100M参数
            return 32
        else:                      # >100M参数
            return 64
```

#### 3.1.3 优缺点分析

**优点：**
- 与PyTorch深度集成，对用户相对透明
- 可以利用PyTorch的自动求导和优化器
- 开发和调试相对容易

**缺点：**
- 需要修改PyTorch核心代码或依赖私有API
- SM级别的精确控制可能受限于CUDA驱动
- 跨模型的内存隔离可能不够彻底

### 3.2 方案二：CUDA库劫持方案

#### 3.2.1 核心思路

**A. LD_PRELOAD方案**
```c
// cuda_carpool_intercept.c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <pthread.h>

// 原始CUDA函数指针
static cudaError_t (*real_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;
static cudaError_t (*real_cudaMemcpy)(void*, const void*, size_t, enum cudaMemcpyKind) = NULL;

// 拼车调度器
typedef struct {
    int available_sms;
    pthread_mutex_t mutex;
    // SM分配位图
    unsigned long sm_bitmap;  // 每位代表一个SM的可用状态
} SMCarPoolManager;

static SMCarPoolManager g_carpool_manager = {
    .available_sms = 108,  // A100默认
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .sm_bitmap = ~0UL  // 初始时所有SM可用
};

// 劫持cudaLaunchKernel
cudaError_t cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream)
{
    // 延迟加载真实函数
    if (!real_cudaLaunchKernel) {
        real_cudaLaunchKernel = dlsym(RTLD_NEXT, "cudaLaunchKernel");
    }
    
    // 估算所需SM数量
    int required_sms = estimate_kernel_sm_requirement(func, gridDim, blockDim);
    
    // 尝试分配SM
    int allocated_sms = allocate_sms(&g_carpool_manager, required_sms);
    if (allocated_sms == -1) {
        // 无法分配，加入等待队列或降级处理
        return handle_allocation_failure(func, gridDim, blockDim, args, sharedMem, stream);
    }
    
    // 设置kernel在指定SM范围内执行
    cudaError_t result = launch_kernel_on_sms(
        real_cudaLaunchKernel,
        func, gridDim, blockDim, args, sharedMem, stream,
        allocated_sms
    );
    
    // 执行完成后释放SM
    deallocate_sms(&g_carpool_manager, allocated_sms);
    
    return result;
}

// SM分配函数
int allocate_sms(SMCarPoolManager* manager, int required_sms) {
    pthread_mutex_lock(&manager->mutex);
    
    // 在位图中寻找连续的可用SM
    int start_sm = find_consecutive_sms(manager->sm_bitmap, required_sms);
    
    if (start_sm != -1) {
        // 标记这些SM为已使用
        unsigned long mask = ((1UL << required_sms) - 1) << start_sm;
        manager->sm_bitmap &= ~mask;
        manager->available_sms -= required_sms;
    }
    
    pthread_mutex_unlock(&manager->mutex);
    return start_sm;
}
```

**B. CUDA驱动级劫持**
```c
// nvidia_carpool.c - 更底层的劫持
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>

// 劫持CUDA驱动的ioctl调用
static struct kprobe kp_nvidia_ioctl;

static int nvidia_ioctl_pre(struct kprobe *p, struct pt_regs *regs) {
    // 检查是否是kernel launch相关的ioctl
    unsigned int cmd = regs->si;  // ioctl命令
    
    if (is_kernel_launch_ioctl(cmd)) {
        // 在这里实现SM调度逻辑
        return handle_kernel_launch_with_carpool(regs);
    }
    
    return 0;  // 继续正常处理
}

static int __init carpool_init(void) {
    kp_nvidia_ioctl.symbol_name = "nvidia_ioctl";
    kp_nvidia_ioctl.pre_handler = nvidia_ioctl_pre;
    
    return register_kprobe(&kp_nvidia_ioctl);
}

static void __exit carpool_exit(void) {
    unregister_kprobe(&kp_nvidia_ioctl);
}

module_init(carpool_init);
module_exit(carpool_exit);
```

#### 3.2.2 优缺点分析

**优点：**
- 对应用程序完全透明
- 可以处理任何CUDA应用，不局限于PyTorch
- 更底层的控制能力

**缺点：**
- 实现复杂度高，需要深入理解CUDA驱动
- 稳定性风险大，可能导致系统崩溃
- 调试困难，错误定位复杂
- 可能违反CUDA的使用协议

### 3.3 方案三：CUDA MPS改进方案

#### 3.3.1 基于CUDA MPS的拼车实现
```bash
# 启动MPS服务
nvidia-cuda-mps-control -d

# 设置每个客户端的资源限制
echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control
echo "set_default_device_memory_limit_percent 25" | nvidia-cuda-mps-control

# 创建特定的MPS客户端配置
cat > mps_carpool.conf << EOF
client_1 50 25  # 50%计算资源，25%内存
client_2 30 25
client_3 20 25
EOF
```

```python
# mps_carpool_manager.py
import os
import subprocess
import threading
from queue import Queue

class MPSCarpoolManager:
    def __init__(self):
        self.active_clients = {}
        self.resource_queue = Queue()
        self.total_compute_percentage = 100
        self.available_compute = 100
        
    def submit_model_request(self, model_func, required_compute_pct=25):
        if self.available_compute >= required_compute_pct:
            # 立即分配资源
            client_id = self._create_mps_client(required_compute_pct)
            thread = threading.Thread(
                target=self._execute_with_mps,
                args=(client_id, model_func, required_compute_pct)
            )
            thread.start()
        else:
            # 加入等待队列
            self.resource_queue.put((model_func, required_compute_pct))
    
    def _create_mps_client(self, compute_pct):
        client_id = f"client_{len(self.active_clients)}"
        
        # 设置MPS客户端限制
        cmd = f"echo 'set_active_thread_percentage {client_id} {compute_pct}' | nvidia-cuda-mps-control"
        subprocess.run(cmd, shell=True)
        
        self.active_clients[client_id] = compute_pct
        self.available_compute -= compute_pct
        
        return client_id
    
    def _execute_with_mps(self, client_id, model_func, compute_pct):
        # 设置环境变量，让CUDA使用特定的MPS客户端
        env = os.environ.copy()
        env['CUDA_MPS_CLIENT_ID'] = client_id
        
        try:
            # 在MPS环境中执行模型
            result = model_func()
        finally:
            # 释放资源
            self._release_mps_client(client_id, compute_pct)
    
    def _release_mps_client(self, client_id, compute_pct):
        del self.active_clients[client_id]
        self.available_compute += compute_pct
        
        # 检查等待队列
        if not self.resource_queue.empty():
            model_func, required_compute = self.resource_queue.get()
            self.submit_model_request(model_func, required_compute)
```

### 3.4 方案四：CUDA Streams + 动态并行

#### 3.4.1 CUDA Streams的内部工作原理

**核心概念：并发执行 + 资源协调**
```
GPU硬件层面的Stream工作机制：
┌─────────────────────────────────────────────────────────────────┐
│ GPU硬件 (108 SMs)                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Stream Scheduler (硬件调度器)                                │ │
│ │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │ │
│ │ │Stream 0 │ │Stream 1 │ │Stream 2 │ │Stream 3 │             │ │
│ │ │Queue    │ │Queue    │ │Queue    │ │Queue    │             │ │
│ │ └─────────┘ └─────────┘ └─────────┘ └─────────┘             │ │
│ └─────────────────────────────────────────────────────────────┐ │
│                                  ↓                            │ │
│ ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │ │
│ │ SM 0-26     │ SM 27-53    │ SM 54-80    │ SM 81-107       │ │ │
│ │ (Model A)   │ (Model B)   │ (Model C)   │ (可用/待分配)    │ │ │
│ └─────────────┴─────────────┴─────────────┴─────────────────┘ │ │
└─────────────────────────────────────────────────────────────────┘
```

**关键技术：Stream并不直接分配SM，而是通过竞争获得SM资源**

```python
class IntelligentStreamCarpool:
    def __init__(self):
        self.total_sms = 108
        # 关键：多个Stream共享同一个GPU，通过调度协调使用SM
        self.streams = [torch.cuda.Stream() for _ in range(8)]
        self.stream_models = {}  # 记录每个stream正在运行的模型
        self.sm_usage_tracker = SMUsageTracker()
        
    def submit_model_batch(self, models_and_inputs, allow_merge=True):
        """
        核心方法：智能决定是拼车还是独占
        """
        total_estimated_sms = sum(
            self._estimate_model_sms(model) 
            for model, _ in models_and_inputs
        )
        
        if total_estimated_sms <= self.total_sms * 0.8:
            # 可以拼车：所有模型预计使用不超过80%的SM
            return self._execute_carpool_mode(models_and_inputs)
        else:
            # 需要独占或分批：避免过度竞争
            return self._execute_exclusive_mode(models_and_inputs)
    
    def _execute_carpool_mode(self, models_and_inputs):
        """拼车模式：多个小模型并发执行"""
        futures = []
        
        for i, (model, input_data) in enumerate(models_and_inputs):
            stream_id = i % len(self.streams)
            stream = self.streams[stream_id]
            
            # 关键：在不同stream上并发launch kernel
            future = self._launch_on_stream(model, input_data, stream, stream_id)
            futures.append(future)
        
        # 等待所有stream完成
        results = []
        for future in futures:
            results.append(future.result())
        
        return results
    
    def _execute_exclusive_mode(self, models_and_inputs):
        """独占模式：大模型或过多小模型时，顺序执行以获得最大SM利用"""
        results = []
        
        # 使用主stream顺序执行，获得全部SM资源
        main_stream = torch.cuda.current_stream()
        
        for model, input_data in models_and_inputs:
            with torch.cuda.stream(main_stream):
                result = self._execute_model_full_gpu(model, input_data)
                results.append(result)
                
        return results
    
    def _launch_on_stream(self, model, input_data, stream, stream_id):
        """在指定stream上启动模型，关键是理解这里的并发机制"""
        return asyncio.create_task(
            self._async_model_execution(model, input_data, stream, stream_id)
        )
    
    async def _async_model_execution(self, model, input_data, stream, stream_id):
        """
        异步模型执行：这里是Stream拼车的核心
        """
        # 1. 将数据和模型移到GPU (如果需要)
        with torch.cuda.stream(stream):
            if not input_data.is_cuda:
                input_data = input_data.cuda(non_blocking=True)
            
            # 2. 模型forward - 关键时刻！
            # 这里每个kernel launch都会竞争SM资源
            # GPU硬件调度器会自动分配可用的SM给当前stream的kernel
            start_time = time.time()
            
            with torch.no_grad():
                result = model(input_data)
            
            # 3. 等待当前stream的所有操作完成
            stream.synchronize()
            
            execution_time = time.time() - start_time
            
            # 4. 记录性能数据，用于后续调度优化
            self.sm_usage_tracker.record_execution(
                stream_id, self._get_model_signature(model), 
                execution_time, self._estimate_model_sms(model)
            )
            
        return result.cpu()  # 返回到CPU避免显存竞争
```

#### 3.4.2 CUDA Streams的关键技术细节

**A. GPU硬件级的Stream调度机制**
```cpp
// GPU硬件层面的调度逻辑 (概念性代码)
class GPUStreamScheduler {
private:
    struct StreamContext {
        cudaStream_t stream;
        std::queue<KernelLaunch> pending_kernels;
        int estimated_sm_usage;
        bool is_active;
    };
    
    std::vector<StreamContext> streams;
    int available_sms = 108;
    
public:
    void schedule_kernel_launch(cudaStream_t stream, KernelLaunch kernel) {
        auto& ctx = get_stream_context(stream);
        
        // 关键：检查是否有足够的SM来启动这个kernel
        int required_sms = estimate_kernel_sm_usage(kernel);
        
        if (available_sms >= required_sms) {
            // 立即启动
            launch_kernel_on_sms(kernel, allocate_sms(required_sms));
            available_sms -= required_sms;
        } else {
            // 加入等待队列
            ctx.pending_kernels.push(kernel);
        }
    }
    
    void on_kernel_completion(KernelLaunch completed_kernel) {
        // kernel完成，释放SM资源
        available_sms += completed_kernel.used_sms;
        
        // 检查是否可以启动等待中的kernel
        try_launch_pending_kernels();
    }
    
    void try_launch_pending_kernels() {
        // 遍历所有stream的等待队列
        for (auto& ctx : streams) {
            while (!ctx.pending_kernels.empty() && 
                   available_sms >= ctx.pending_kernels.front().required_sms) {
                
                auto kernel = ctx.pending_kernels.front();
                ctx.pending_kernels.pop();
                
                launch_kernel_on_sms(kernel, allocate_sms(kernel.required_sms));
                available_sms -= kernel.required_sms;
            }
        }
    }
};
```

**B. 智能拼车决策算法**
```python
class AdaptiveCarpoolDecisionEngine:
    def __init__(self):
        self.execution_history = {}
        self.sm_utilization_threshold = 0.8
        
    def should_carpool(self, models_and_inputs):
        """
        决策引擎：基于历史数据和当前负载决定是否拼车
        """
        # 1. 估算总SM需求
        total_sm_demand = 0
        execution_time_estimates = []
        
        for model, input_data in models_and_inputs:
            model_sig = self._get_model_signature(model, input_data)
            
            # 从历史数据预测SM使用和执行时间
            if model_sig in self.execution_history:
                hist = self.execution_history[model_sig]
                avg_sms = hist['avg_sm_usage']
                avg_time = hist['avg_execution_time']
            else:
                # 新模型，使用启发式估算
                avg_sms = self._heuristic_sm_estimate(model, input_data)
                avg_time = self._heuristic_time_estimate(model, input_data)
            
            total_sm_demand += avg_sms
            execution_time_estimates.append(avg_time)
        
        # 2. 拼车可行性分析
        if total_sm_demand <= 108 * self.sm_utilization_threshold:
            # SM资源足够，检查并行效益
            sequential_time = sum(execution_time_estimates)
            parallel_time = max(execution_time_estimates) * 1.2  # 考虑竞争开销
            
            if parallel_time < sequential_time * 0.8:  # 并行有20%以上收益
                return "CARPOOL"
            else:
                return "SEQUENTIAL" 
        else:
            # SM资源不够，但可以考虑分批拼车
            return self._plan_batch_carpool(models_and_inputs, execution_time_estimates)
    
    def _plan_batch_carpool(self, models_and_inputs, time_estimates):
        """
        批次拼车规划：将请求分组，每组内部拼车，组间顺序执行
        """
        # 按执行时间和SM需求对模型分组
        models_with_estimates = list(zip(models_and_inputs, time_estimates))
        models_with_estimates.sort(key=lambda x: x[1])  # 按执行时间排序
        
        batches = []
        current_batch = []
        current_sm_usage = 0
        
        for (model, input_data), time_est in models_with_estimates:
            model_sms = self._estimate_model_sms(model)
            
            if current_sm_usage + model_sms <= 108 * 0.8:
                # 可以加入当前批次
                current_batch.append((model, input_data))
                current_sm_usage += model_sms
            else:
                # 开始新批次
                if current_batch:
                    batches.append(current_batch)
                current_batch = [(model, input_data)]
                current_sm_usage = model_sms
        
        if current_batch:
            batches.append(current_batch)
        
        return f"BATCH_CARPOOL:{len(batches)}"
```

#### 3.4.3 关键优势：动态性与灵活性

**为什么CUDA Streams特别适合"临时拼车"的需求：**

1. **无硬分割**: Stream不会硬性分割GPU资源，而是通过竞争机制动态获取
2. **自动负载均衡**: GPU硬件调度器会自动平衡不同Stream间的资源分配
3. **弹性调整**: 可以根据实际负载在拼车和独占间自由切换
4. **最小开销**: Stream创建和切换的开销很小

```python
# 使用示例：展示动态切换能力
class FlexibleInferenceEngine:
    def __init__(self):
        self.carpool_manager = IntelligentStreamCarpool()
        
    def process_request_batch(self, requests):
        # 动态决策：根据当前请求特点选择执行模式
        decision = self.carpool_manager.decide_execution_mode(requests)
        
        if decision == "SINGLE_LARGE":
            # 大模型独占全GPU
            return self._execute_exclusive(requests[0])
        elif decision == "MULTI_SMALL":
            # 多个小模型拼车
            return self._execute_carpool(requests)
        elif decision == "MIXED_BATCH":
            # 混合：先处理小模型拼车，再处理大模型
            small_requests = [r for r in requests if r.is_small_model()]
            large_requests = [r for r in requests if not r.is_small_model()]
            
            results = []
            if small_requests:
                results.extend(self._execute_carpool(small_requests))
            if large_requests:
                results.extend(self._execute_exclusive_batch(large_requests))
            
            return results
    
    def _execute_exclusive(self, request):
        """独占模式：使用全部108个SM"""
        with torch.cuda.stream(torch.cuda.current_stream()):
            return request.model(request.input_data)
    
    def _execute_carpool(self, requests):
        """拼车模式：多stream并发，SM自动分配"""
        return self.carpool_manager.submit_model_batch(
            [(r.model, r.input_data) for r in requests],
            allow_merge=True
        )
```

### 3.5 方案五：基于NVIDIA MIG的硬件拼车

#### 3.5.1 MIG分区管理
```python
# mig_carpool.py
import subprocess
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MIGPartition:
    instance_id: int
    compute_units: int  # SM数量
    memory_gb: int
    in_use: bool = False

class MIGCarpoolManager:
    def __init__(self):
        self.partitions = self._discover_mig_partitions()
        
    def _discover_mig_partitions(self) -> List[MIGPartition]:
        # 查询MIG分区信息
        cmd = "nvidia-smi mig -lgi"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        partitions = []
        for line in result.stdout.strip().split('\n')[1:]:  # 跳过header
            parts = line.split()
            if len(parts) >= 4:
                instance_id = int(parts[0])
                compute_units = int(parts[2])  # 简化解析
                memory_gb = int(parts[3].rstrip('GB'))
                partitions.append(MIGPartition(instance_id, compute_units, memory_gb))
        
        return partitions
    
    def allocate_partition(self, required_compute_units: int, required_memory_gb: int) -> Optional[int]:
        # 寻找合适的分区
        for partition in self.partitions:
            if (not partition.in_use and 
                partition.compute_units >= required_compute_units and 
                partition.memory_gb >= required_memory_gb):
                
                partition.in_use = True
                return partition.instance_id
        
        return None
    
    def release_partition(self, instance_id: int):
        for partition in self.partitions:
            if partition.instance_id == instance_id:
                partition.in_use = False
                break
    
    def execute_on_partition(self, instance_id: int, model_func):
        # 设置CUDA设备到特定MIG分区
        old_device = torch.cuda.current_device()
        
        try:
            # MIG分区通常表示为设备的UUID
            torch.cuda.set_device(f"MIG-{instance_id}")
            result = model_func()
        finally:
            torch.cuda.set_device(old_device)
        
        return result

# 使用示例
mig_manager = MIGCarpoolManager()

def run_small_model():
    partition_id = mig_manager.allocate_partition(
        required_compute_units=14,  # 需要14个SM
        required_memory_gb=10       # 需要10GB内存
    )
    
    if partition_id is not None:
        try:
            result = mig_manager.execute_on_partition(partition_id, lambda: model(input_data))
        finally:
            mig_manager.release_partition(partition_id)
        return result
    else:
        print("No available MIG partition")
        return None
```

## 4. 综合比较与推荐

### 4.1 方案对比表

```
┌─────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ 方案            │ 实现难度 │ 性能    │ 稳定性   │ 兼容性   │ 推荐度  │
├─────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ PyTorch Patch   │ 中等    │ 高      │ 高      │ 中等    │ ★★★★☆ │
│ CUDA库劫持      │ 高      │ 很高    │ 低      │ 高      │ ★★☆☆☆ │
│ MPS改进         │ 低      │ 中等    │ 高      │ 高      │ ★★★☆☆ │
│ CUDA Streams    │ 低      │ 中等    │ 很高    │ 很高    │ ★★★★★ │
│ MIG硬件分区     │ 中等    │ 很高    │ 很高    │ 低      │ ★★★☆☆ │
└─────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

### 4.2 推荐实施路径

#### 阶段一：快速原型（推荐CUDA Streams方案）
```python
# 最小可行产品
class QuickCarpoolDemo:
    def __init__(self):
        self.streams = [torch.cuda.Stream() for _ in range(4)]
        self.executors = [ThreadPoolExecutor(max_workers=1) for _ in range(4)]
    
    def submit_models(self, models_and_inputs):
        futures = []
        for i, (model, input_data) in enumerate(models_and_inputs):
            stream_id = i % len(self.streams)
            future = self.executors[stream_id].submit(
                self._execute_model, model, input_data, stream_id
            )
            futures.append(future)
        return futures
```

#### 阶段二：生产优化（PyTorch扩展）
- 开发C++扩展实现更精细的SM控制
- 实现动态资源调度算法
- 添加性能监控和调优功能

#### 阶段三：深度集成（考虑驱动层方案）
- 如果前两阶段效果好，可以考虑更底层的优化
- 与NVIDIA合作开发官方SM调度API

### 4.3 关键技术挑战

1. **SM使用量估算**：如何准确预测模型的SM需求
2. **内存隔离**：确保多个模型间的内存安全
3. **性能监控**：实时监控SM利用率和性能指标
4. **动态调度**：根据实际负载动态调整资源分配

## 5. 总结

SM拼车技术是一个很有前景的GPU利用率优化方向。建议：

1. **先从CUDA Streams方案开始**，风险小，易于实现和调试
2. **逐步向PyTorch扩展方向发展**，获得更好的性能和控制能力
3. **谨慎考虑CUDA库劫持方案**，虽然性能潜力大但风险也高
4. **关注NVIDIA的官方发展**，如Multi-Instance GPU等新技术

最终目标是在保证稳定性的前提下，最大化GPU资源利用率，为用户提供更高的推理吞吐量和更低的成本。
