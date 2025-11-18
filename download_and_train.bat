@echo off
REM CV车辆识别 - 下载数据并训练脚本

echo ==========================================
echo CV车辆识别 - 数据下载和训练流程
echo ==========================================
echo.

REM 检查数据目录
set DATA_DIR=C:\baidunetdiskdownload\CV-车辆检测
set TRAIN_FILE=%DATA_DIR%\re_id_1000_train.txt
set TEST_FILE=%DATA_DIR%\re_id_1000_test.txt
set IMAGE_DIR=%DATA_DIR%\image

echo [步骤1] 检查数据文件...
echo.

if not exist "%DATA_DIR%" (
    echo [警告] 数据目录不存在: %DATA_DIR%
    echo.
    echo 请按照以下步骤下载数据：
    echo 1. 打开百度网盘链接: https://pan.baidu.com/s/1GnQ0aUciBN1_x85Qn-swWg
    echo 2. 提取码: 3rms
    echo 3. 下载 "CV-车辆检测" 文件夹
    echo 4. 将下载的文件解压到: %DATA_DIR%
    echo.
    echo 数据目录结构应该是：
    echo %DATA_DIR%\
    echo   ├── re_id_1000_train.txt
    echo   ├── re_id_1000_test.txt
    echo   └── image\
    echo       ├── 1\
    echo       ├── 2\
    echo       ...
    echo       └── 10\
    echo.
    pause
    exit /b 1
)

if not exist "%TRAIN_FILE%" (
    echo [错误] 训练索引文件不存在: %TRAIN_FILE%
    echo 请检查数据文件是否已正确下载和解压
    pause
    exit /b 1
)

if not exist "%TEST_FILE%" (
    echo [错误] 测试索引文件不存在: %TEST_FILE%
    echo 请检查数据文件是否已正确下载和解压
    pause
    exit /b 1
)

if not exist "%IMAGE_DIR%" (
    echo [错误] 图像目录不存在: %IMAGE_DIR%
    echo 请检查数据文件是否已正确下载和解压
    pause
    exit /b 1
)

echo [✓] 数据文件检查通过
echo.

REM 统计文件数量
echo [步骤2] 统计数据文件...
python -c "import os; train_file=r'%TRAIN_FILE%'; test_file=r'%TEST_FILE%'; train_count=len([l for l in open(train_file, 'r', encoding='utf-8') if l.strip()]) if os.path.exists(train_file) else 0; test_count=len([l for l in open(test_file, 'r', encoding='utf-8') if l.strip()]) if os.path.exists(test_file) else 0; print(f'训练样本数: {train_count}'); print(f'测试样本数: {test_count}')" 2>nul
if errorlevel 1 (
    echo [警告] 无法统计文件数量，继续执行...
)
echo.

REM 检查Python环境
echo [步骤3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python未安装或未添加到PATH
    echo 请先安装Python 3.8+
    pause
    exit /b 1
)
python --version
echo.

REM 检查依赖
echo [步骤4] 检查依赖包...
python test_imports.py
if errorlevel 1 (
    echo.
    echo [警告] 依赖包未安装，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖包安装失败
        pause
        exit /b 1
    )
)
echo.

REM 开始训练
echo [步骤5] 开始训练模型...
echo ==========================================
echo.
python main.py

if errorlevel 1 (
    echo.
    echo [错误] 训练过程出现错误
    pause
    exit /b 1
)

echo.
echo ==========================================
echo 训练完成！
echo ==========================================
echo.
echo 模型文件保存在: models\
echo 结果文件保存在: results\
echo.
pause

