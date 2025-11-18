"""
CV车辆识别配置文件
"""
import os

# 数据路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 检查是否在Docker容器中运行
if os.path.exists("/app/data"):
    # Docker环境
    DATA_DIR = "/app/data"
else:
    # 本地环境（百度网盘下载目录）
    DATA_DIR = r"C:\baidunetdiskdownload\CV-车辆检测"

IMAGE_DIR = os.path.join(DATA_DIR, "image")

# 索引文件路径
TRAIN_INDEX_FILE = os.path.join(DATA_DIR, "re_id_1000_train.txt")
TEST_INDEX_FILE = os.path.join(DATA_DIR, "re_id_1000_test.txt")

# 模型保存路径（保存在项目目录）
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 模型参数
MIN_ACCURACY = 0.80  # 最低准确率要求
MAX_ITERATIONS = 10  # 最大调参迭代次数

# 车辆品牌类别（10类）
# 注意：顺序必须与文件夹编号对应（1-10对应索引0-9）
# 文件夹顺序：1=丰田_凯美瑞, 2=丰田_卡罗拉, 3=丰田_花冠, 4=别克_君越, 
#           5=大众_迈腾, 6=奥迪_A4, 7=日产_轩逸, 8=日产_骐达, 
#           9=本田_雅阁, 10=福特_福克斯
CAR_BRANDS = [
    "丰田_凯美瑞",      # 索引0，对应文件夹1 (Toyota_Camry)
    "丰田_卡罗拉",      # 索引1，对应文件夹2 (Toyota_Corolla)
    "丰田_花冠",        # 索引2，对应文件夹3 (Toyota_Corolla_EX)
    "别克_君越",        # 索引3，对应文件夹4 (Buick_LaCrosse)
    "大众_迈腾",        # 索引4，对应文件夹5 (Volkswagen_Magotan)
    "奥迪_A4",          # 索引5，对应文件夹6 (Audi_A4)
    "日产_轩逸",        # 索引6，对应文件夹7 (Nissan_Sylphy)
    "日产_骐达",        # 索引7，对应文件夹8 (Nissan_Tiida)
    "本田_雅阁",        # 索引8，对应文件夹9 (Honda_Accord)
    "福特_福克斯"       # 索引9，对应文件夹10 (Ford_Focus)
]

NUM_CLASSES = len(CAR_BRANDS)  # 10个类别

# 图像预处理参数
IMAGE_SIZE = (224, 224)  # 输入图像尺寸
BATCH_SIZE = 32  # 批次大小
NUM_WORKERS = 0  # 数据加载线程数（Docker中设为0避免共享内存问题）

# 训练参数
LEARNING_RATE = 0.001  # 初始学习率
EPOCHS = 50  # 训练轮数
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值

# 数据增强参数
USE_DATA_AUGMENTATION = True  # 是否使用数据增强

# 模型类型（可选：resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b3）
MODEL_TYPE = "resnet18"  # 默认使用ResNet18（速度快）

# GPU设置
USE_GPU = True  # 是否使用GPU
CUDA_VISIBLE_DEVICES = "0"  # GPU设备ID

