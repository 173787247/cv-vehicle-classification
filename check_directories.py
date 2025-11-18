"""
检查数据文件和源代码是否在正确的位置
"""
import os

# 源代码目录
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件目录（配置）
DATA_DIR = r"C:\baidunetdiskdownload\CV-车辆检测"

print("="*60)
print("目录结构检查")
print("="*60)
print()

print("1. 源代码目录（项目代码）:")
print(f"   {SOURCE_DIR}")
print()

print("2. 数据文件目录（下载的数据）:")
print(f"   {DATA_DIR}")
print()

print("="*60)
print("检查结果:")
print("="*60)
print()

# 检查项目目录中是否有数据文件
source_has_data = False
if os.path.exists(os.path.join(SOURCE_DIR, "re_id_1000_train.txt")):
    print("[警告] 项目目录中发现训练索引文件！")
    source_has_data = True

if os.path.exists(os.path.join(SOURCE_DIR, "re_id_1000_test.txt")):
    print("[警告] 项目目录中发现测试索引文件！")
    source_has_data = True

if os.path.exists(os.path.join(SOURCE_DIR, "image")):
    print("[警告] 项目目录中发现image文件夹！")
    source_has_data = True

if not source_has_data:
    print("[OK] 项目目录中没有数据文件，数据文件在正确的位置")
print()

# 检查数据目录
print("3. 数据文件目录检查:")
if os.path.exists(DATA_DIR):
    print(f"   [OK] 数据目录存在: {DATA_DIR}")
    
    train_file = os.path.join(DATA_DIR, "re_id_1000_train.txt")
    test_file = os.path.join(DATA_DIR, "re_id_1000_test.txt")
    image_dir = os.path.join(DATA_DIR, "image")
    
    if os.path.exists(train_file):
        print(f"   [OK] 训练索引文件存在")
    else:
        print(f"   [错误] 训练索引文件不存在")
    
    if os.path.exists(test_file):
        print(f"   [OK] 测试索引文件存在")
    else:
        print(f"   [错误] 测试索引文件不存在")
    
    if os.path.exists(image_dir):
        print(f"   [OK] 图像目录存在")
    else:
        print(f"   [错误] 图像目录不存在")
else:
    print(f"   [错误] 数据目录不存在: {DATA_DIR}")
print()

print("="*60)
print("总结:")
print("="*60)
print()
print("源代码目录和数据文件目录是分开的，不会冲突！")
print()
print("如果项目目录中有数据文件，可以安全删除（数据在正确位置）。")
print("如果数据目录中没有文件，请将下载的数据解压到：")
print(f"   {DATA_DIR}")

