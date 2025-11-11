# 使用PyTorch官方镜像（支持GPU）
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV CUDA_VISIBLE_DEVICES=0

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY *.py ./

# 创建必要的目录
RUN mkdir -p models results

# 设置环境变量
ENV PYTHONPATH=/app

# 默认命令
CMD ["python", "main.py"]

