#!/bin/bash
# Xcode AI Proxy Python 版本启动脚本

echo "🚀 启动 Xcode AI Proxy Python 版本"

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件，从示例文件复制..."
    if [ -f .env.python.example ]; then
        cp .env.python.example .env
        echo "📝 请编辑 .env 文件并填入真实的 API 密钥"
        exit 1
    else
        echo "❌ 未找到 .env.python.example 文件"
        exit 1
    fi
fi

# 安装依赖
echo "📦 安装 Python 依赖..."
pip3 install -r requirements.txt

# 启动服务
echo "🔧 启动服务器..."
python3 server_python.py
