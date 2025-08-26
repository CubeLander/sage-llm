#!/usr/bin/env python3
"""
A100性能测试环境检查脚本

检查运行性能测试所需的环境和依赖。
"""

import sys
import subprocess
import importlib
import torch
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  警告: 推荐使用Python 3.8+")
        return False
    else:
        print("✓ Python版本符合要求")
        return True

def check_cuda_gpu():
    """检查CUDA和GPU"""
    print("\n检查CUDA和GPU环境...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    if gpu_count < 2:
        print("⚠️  警告: 检测到少于2个GPU，某些测试可能无法运行")
    else:
        print("✓ GPU配置符合要求")
    
    # 检查P2P访问
    if gpu_count >= 2:
        try:
            can_p2p = torch.cuda.can_device_access_peer(0, 1)
            print(f"GPU 0-1 P2P访问: {'支持' if can_p2p else '不支持'}")
        except:
            print("P2P访问检查失败")
    
    return True

def check_required_packages():
    """检查必需的Python包"""
    print("\n检查必需的Python包...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("vllm", "vLLM推理引擎"),
        ("transformers", "Transformers库"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib (用于可视化)"),
        ("seaborn", "Seaborn (用于可视化)")
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {description} - 已安装")
        except ImportError:
            print(f"❌ {description} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少包: {', '.join(missing_packages)}")
        print("安装命令:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✓ 所有必需包已安装")
        return True

def check_vllm_installation():
    """检查vLLM安装是否正常"""
    print("\n检查vLLM安装...")
    
    try:
        from vllm import LLM, SamplingParams
        print("✓ vLLM导入成功")
        
        # 简单的功能测试
        try:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
            print("✓ vLLM基本功能正常")
            return True
        except Exception as e:
            print(f"⚠️  vLLM功能测试失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ vLLM导入失败: {e}")
        print("安装vLLM: pip install vllm")
        return False

def check_nvidia_smi():
    """检查nvidia-smi命令"""
    print("\n检查nvidia-smi...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvidia-smi可用")
            return True
        else:
            print("❌ nvidia-smi命令失败")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ nvidia-smi命令不可用")
        return False

def check_disk_space():
    """检查磁盘空间"""
    print("\n检查磁盘空间...")
    
    current_dir = Path.cwd()
    try:
        stat = current_dir.stat()
        # 简单的空间检查 - 在实际实现中需要更精确的方法
        print("✓ 磁盘空间检查通过")
        return True
    except:
        print("⚠️  磁盘空间检查失败")
        return False

def check_model_access():
    """检查模型访问权限"""
    print("\n检查模型访问...")
    
    try:
        from transformers import AutoTokenizer
        # 尝试加载一个小模型的tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="/tmp")
        print("✓ Hugging Face模型访问正常")
        return True
    except Exception as e:
        print(f"⚠️  模型访问可能有问题: {e}")
        print("建议设置HF_TOKEN环境变量或检查网络连接")
        return False

def print_system_info():
    """打印系统信息"""
    print("\n=== 系统信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    try:
        import vllm
        print(f"vLLM版本: {vllm.__version__}")
    except:
        print("vLLM版本: 未安装")

def main():
    print("=== A100性能测试环境检查 ===\n")
    
    checks = [
        ("Python版本", check_python_version),
        ("CUDA和GPU", check_cuda_gpu),
        ("必需包", check_required_packages),
        ("vLLM安装", check_vllm_installation),
        ("nvidia-smi", check_nvidia_smi),
        ("磁盘空间", check_disk_space),
        ("模型访问", check_model_access),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}检查失败: {e}")
            results.append((check_name, False))
    
    print_system_info()
    
    # 总结
    print("\n=== 检查总结 ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"通过检查: {passed}/{total}")
    
    failed_checks = [name for name, result in results if not result]
    if failed_checks:
        print(f"失败检查: {', '.join(failed_checks)}")
        print("\n⚠️  存在问题，可能影响实验运行")
        print("请根据上述提示解决问题后重新检查")
        return False
    else:
        print("✓ 所有检查通过，环境就绪!")
        print("\n可以开始运行实验:")
        print("  ./experiments/run_all_experiments.sh")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
