@echo off
REM CV车辆识别 - 启动GPU训练

echo ==========================================
echo CV车辆识别 - 启动GPU训练
echo ==========================================
echo.

cd /d "%~dp0"

echo [步骤1] 检查Docker状态...
docker info >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker未运行，请先启动Docker Desktop
    pause
    exit /b 1
)
echo [OK] Docker运行正常
echo.

echo [步骤2] 检查GPU支持...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到NVIDIA GPU，将使用CPU模式
) else (
    echo [OK] 检测到NVIDIA GPU
    nvidia-smi --query-gpu=name --format=csv,noheader
)
echo.

echo [步骤3] 开始构建和启动Docker容器...
echo 注意: 首次构建需要下载基础镜像，可能需要10-30分钟
echo.

docker-compose -f docker-compose.gpu.yml up --build

pause

