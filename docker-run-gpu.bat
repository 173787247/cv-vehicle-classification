@echo off
REM CV车辆识别 - Docker GPU运行脚本 (Windows)

echo ==========================================
echo CV车辆识别 - Docker GPU运行脚本
echo ==========================================

REM 检查Docker是否运行
docker info >nul 2>&1
if errorlevel 1 (
    echo [错误] Docker未运行，请先启动Docker Desktop
    pause
    exit /b 1
)

echo [信息] Docker运行正常

REM 检查是否有GPU支持
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到NVIDIA GPU，将使用CPU版本
    echo [信息] 开始构建和启动容器（CPU模式）...
    docker-compose up --build
) else (
    echo [信息] 检测到NVIDIA GPU，使用GPU版本...
    echo [信息] 开始构建和启动容器（GPU模式）...
    docker-compose -f docker-compose.gpu.yml up --build
)

pause

