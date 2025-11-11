"""
测试依赖包是否安装
"""
import sys

def test_imports():
    """测试所有必需的依赖包"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA 不可用，将使用CPU训练（速度较慢）")
    except:
        pass
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖包已安装！")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

