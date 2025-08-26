#!/bin/bash

# 双A100 GPU性能测试快速启动脚本
# 使用方法: ./quick_start.sh [test_type] [options]

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
OUTPUT_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$OUTPUT_DIR/test_$TIMESTAMP"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助信息
show_help() {
    echo "双A100 GPU性能测试快速启动脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [test_type] [options]"
    echo ""
    echo "测试类型:"
    echo "  bandwidth     - GPU间带宽测试"
    echo "  sequence      - 序列长度扩展性测试"
    echo "  pp            - Pipeline Parallel开销测试"
    echo "  tp            - Tensor Parallel通信测试"
    echo "  comprehensive - 运行所有测试"
    echo "  quick         - 快速测试（最小配置）"
    echo ""
    echo "选项:"
    echo "  --model MODEL        指定模型 (默认: facebook/opt-125m)"
    echo "  --output-dir DIR     指定输出目录 (默认: results)"
    echo "  --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 bandwidth                    # 运行GPU带宽测试"
    echo "  $0 sequence --model opt-1.3b    # 运行序列长度测试"
    echo "  $0 comprehensive                # 运行所有测试"
    echo "  $0 quick                        # 快速测试"
}

# 检查环境
check_environment() {
    echo -e "${BLUE}检查运行环境...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: Python3 未安装${NC}"
        exit 1
    fi
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}错误: nvidia-smi 未找到${NC}"
        exit 1
    fi
    
    # 检查GPU数量
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo -e "${GREEN}发现 $GPU_COUNT 个GPU${NC}"
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo -e "${YELLOW}警告: 检测到少于2个GPU，某些测试可能无法运行${NC}"
    fi
    
    # 检查依赖
    echo "检查Python依赖..."
    python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
        echo -e "${RED}错误: PyTorch未安装${NC}"
        exit 1
    }
    
    python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    python3 -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
    
    echo -e "${GREEN}环境检查完成${NC}"
}

# 运行带宽测试
run_bandwidth_test() {
    echo -e "${BLUE}运行GPU带宽测试...${NC}"
    mkdir -p "$RESULTS_DIR"
    
    python3 gpu_bandwidth_test.py \
        --output-dir "$RESULTS_DIR" \
        "$@"
    
    echo -e "${GREEN}GPU带宽测试完成${NC}"
}

# 运行序列长度测试
run_sequence_test() {
    local model="${1:-facebook/opt-125m}"
    echo -e "${BLUE}运行序列长度测试 (模型: $model)...${NC}"
    mkdir -p "$RESULTS_DIR"
    
    python3 sequence_length_test.py \
        --model "$model" \
        --min-length 100 \
        --max-length 2048 \
        --steps 8 \
        --output-dir "$RESULTS_DIR"
    
    echo -e "${GREEN}序列长度测试完成${NC}"
}

# 运行PP测试
run_pp_test() {
    local model="${1:-facebook/opt-1.3b}"
    echo -e "${BLUE}运行Pipeline Parallel测试 (模型: $model)...${NC}"
    mkdir -p "$RESULTS_DIR"
    
    python3 pp_overhead_test.py \
        --model "$model" \
        --max-pp 2 \
        --seq-lengths 256 512 1024 \
        --batch-sizes 1 4 8 \
        --output-dir "$RESULTS_DIR"
    
    echo -e "${GREEN}Pipeline Parallel测试完成${NC}"
}

# 运行TP测试
run_tp_test() {
    local model="${1:-facebook/opt-1.3b}"
    echo -e "${BLUE}运行Tensor Parallel测试 (模型: $model)...${NC}"
    mkdir -p "$RESULTS_DIR"
    
    python3 tp_communication_test.py \
        --model "$model" \
        --max-tp 2 \
        --seq-lengths 256 512 1024 \
        --batch-sizes 1 4 8 \
        --output-dir "$RESULTS_DIR"
    
    echo -e "${GREEN}Tensor Parallel测试完成${NC}"
}

# 运行综合测试
run_comprehensive_test() {
    echo -e "${BLUE}运行综合性能测试...${NC}"
    mkdir -p "$OUTPUT_DIR"
    
    python3 run_comprehensive_test.py \
        --output-dir "$OUTPUT_DIR" \
        "$@"
    
    echo -e "${GREEN}综合测试完成${NC}"
}

# 运行快速测试
run_quick_test() {
    echo -e "${BLUE}运行快速测试...${NC}"
    mkdir -p "$OUTPUT_DIR"
    
    python3 run_comprehensive_test.py \
        --output-dir "$OUTPUT_DIR" \
        --quick-test \
        "$@"
    
    echo -e "${GREEN}快速测试完成${NC}"
}

# 主函数
main() {
    local test_type="$1"
    shift
    
    # 解析选项
    local model="facebook/opt-125m"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                RESULTS_DIR="$OUTPUT_DIR/test_$TIMESTAMP"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                # 传递给具体测试脚本的参数
                break
                ;;
        esac
    done
    
    # 检查环境
    check_environment
    
    # 根据测试类型执行相应操作
    case "$test_type" in
        bandwidth)
            run_bandwidth_test "$@"
            ;;
        sequence)
            run_sequence_test "$model" "$@"
            ;;
        pp)
            run_pp_test "$model" "$@"
            ;;
        tp)
            run_tp_test "$model" "$@"
            ;;
        comprehensive)
            run_comprehensive_test --models "$model" "$@"
            ;;
        quick)
            run_quick_test "$@"
            ;;
        "")
            echo -e "${YELLOW}请指定测试类型${NC}"
            show_help
            exit 1
            ;;
        *)
            echo -e "${RED}未知的测试类型: $test_type${NC}"
            show_help
            exit 1
            ;;
    esac
    
    # 显示结果位置
    echo -e "${GREEN}测试完成！${NC}"
    echo -e "结果保存在: ${BLUE}$RESULTS_DIR${NC}"
    
    # 如果有结果文件，显示快速统计
    if [ -d "$RESULTS_DIR" ]; then
        echo -e "\n${BLUE}生成的文件:${NC}"
        ls -la "$RESULTS_DIR" | grep -E '\.(json|md|png)$' | while read line; do
            echo "  $line"
        done
    fi
}

# 如果没有参数，显示帮助
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# 运行主函数
main "$@"
