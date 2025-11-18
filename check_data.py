"""
检查数据文件是否准备就绪
"""
import os
from config import TRAIN_INDEX_FILE, TEST_INDEX_FILE, IMAGE_DIR, DATA_DIR

def check_data_files():
    """检查数据文件"""
    print("="*60)
    print("CV车辆识别 - 数据文件检查")
    print("="*60)
    print()
    
    all_ok = True
    
    # 检查数据目录
    print(f"数据目录: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"  [ERROR] 数据目录不存在")
        all_ok = False
    else:
        print(f"  [OK] 数据目录存在")
    print()
    
    # 检查训练索引文件
    print(f"训练索引文件: {TRAIN_INDEX_FILE}")
    if not os.path.exists(TRAIN_INDEX_FILE):
        print(f"  [ERROR] 训练索引文件不存在")
        all_ok = False
    else:
        with open(TRAIN_INDEX_FILE, 'r', encoding='utf-8') as f:
            train_lines = [l.strip() for l in f if l.strip()]
        print(f"  [OK] 训练索引文件存在")
        print(f"  [OK] 训练样本数: {len(train_lines)}")
    print()
    
    # 检查测试索引文件
    print(f"测试索引文件: {TEST_INDEX_FILE}")
    if not os.path.exists(TEST_INDEX_FILE):
        print(f"  [ERROR] 测试索引文件不存在")
        all_ok = False
    else:
        with open(TEST_INDEX_FILE, 'r', encoding='utf-8') as f:
            test_lines = [l.strip() for l in f if l.strip()]
        print(f"  [OK] 测试索引文件存在")
        print(f"  [OK] 测试样本数: {len(test_lines)}")
    print()
    
    # 检查图像目录
    print(f"图像目录: {IMAGE_DIR}")
    if not os.path.exists(IMAGE_DIR):
        print(f"  [ERROR] 图像目录不存在")
        all_ok = False
    else:
        print(f"  [OK] 图像目录存在")
        # 检查子文件夹
        subdirs = [d for d in os.listdir(IMAGE_DIR) 
                   if os.path.isdir(os.path.join(IMAGE_DIR, d)) and d.isdigit()]
        subdirs.sort(key=int)
        print(f"  [OK] 找到 {len(subdirs)} 个编号文件夹: {', '.join(subdirs)}")
        
        # 检查每个文件夹
        for subdir in subdirs:
            subdir_path = os.path.join(IMAGE_DIR, subdir)
            files = [f for f in os.listdir(subdir_path) 
                    if os.path.isfile(os.path.join(subdir_path, f))]
            # 递归查找图片文件
            image_count = 0
            for root, dirs, files in os.walk(subdir_path):
                image_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    文件夹 {subdir}: {image_count} 张图片")
    print()
    
    # 总结
    print("="*60)
    if all_ok:
        print("[OK] 所有数据文件检查通过，可以开始训练！")
        print()
        print("运行以下命令开始训练：")
        print("  python main.py")
        print("  或")
        print("  run.bat")
        return True
    else:
        print("[ERROR] 数据文件不完整，请检查以下内容：")
        print()
        print("1. 从百度网盘下载数据：")
        print("   链接: https://pan.baidu.com/s/1GnQ0aUciBN1_x85Qn-swWg")
        print("   提取码: 3rms")
        print()
        print("2. 将下载的文件解压到：")
        print(f"   {DATA_DIR}")
        print()
        print("3. 确保目录结构如下：")
        print(f"   {DATA_DIR}\\")
        print("     ├── re_id_1000_train.txt")
        print("     ├── re_id_1000_test.txt")
        print("     └── image\\")
        print("         ├── 1\\")
        print("         ├── 2\\")
        print("         ...")
        print("         └── 10\\")
        return False

if __name__ == "__main__":
    check_data_files()

