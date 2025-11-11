@echo off
chcp 65001 >nul
echo ========================================
echo CV车辆识别 - 车辆品牌分类模型
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo 检查依赖包...
python test_imports.py
if errorlevel 1 (
    echo.
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

echo.
echo 开始训练模型...
echo.
python main.py

echo.
echo 按任意键退出...
pause >nul

