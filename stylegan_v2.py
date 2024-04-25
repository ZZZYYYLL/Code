# 导入os模块用于操作系统功能，改变当前工作目录
import os
# 将当前工作目录更改为'/content'
os.chdir('/content')
CODE_DIR = 'SAM'

!git clone https://github.com/yuval-alaluf/SAM.git $CODE_DIR
# 下载ninja构建系统的Linux版本压缩包
!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
!sudo unzip ninja-linux.zip -d /usr/local/bin/
!sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
=======
# 导入os模块用于操作系统功能，改变当前工作目录
import os
# 将当前工作目录更改为'/content'
# os.chdir('/content')
CODE_DIR = 'SAM'

# !git clone https://github.com/yuval-alaluf/SAM.git $CODE_DIR
# # 下载ninja构建系统的Linux版本压缩包
# !wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
# !sudo unzip ninja-linux.zip -d /usr/local/bin/
# !sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
>>>>>>> a531946 (done)

os.chdir(f'./{CODE_DIR}')
# 导入所需的Python模块和函数
from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
# 将当前目录及其父目录添加到sys.path中，以便可以导入模块和函数
sys.path.append(".")
sys.path.append("..")

# 从datasets.augmentations导入AgeTransformer用于年龄转换
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
# 从utils.common导入tensor2im用于将张量转换为图像
from models.psp import pSp
# 定义实验类型为ffhq_aging，用于老化转换
EXPERIMENT_TYPE = 'ffhq_aging'
# 定义实验数据参数
EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
<<<<<<< HEAD
        "model_path": "/content/drive/MyDrive/stylegan_aging/sam_ffhq_aging.pt",
        "image_path": "/content/微信图片_20240402200213.jpg",
=======
        "model_path": "/root/autodl-tmp/sam_ffhq_aging.pt",
        "image_path": "/root/autodl-tmp/images/male/048A331.JPG",
>>>>>>> a531946 (done)
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

# 从EXPERIMENT_DATA_ARGS中获取当前实验类型的参数
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
# 从EXPERIMENT_ARGS中获取模型的路径
model_path = EXPERIMENT_ARGS['model_path']
# 加载模型权重，设定为在CPU上进行
ckpt = torch.load(model_path, map_location='cpu')
# 提取训练选项
opts = ckpt['opts']
pprint.pprint(opts)

# 更新训练选项中的checkpoint路径
opts['checkpoint_path'] = model_path
# 将opts字典转换为Namespace对象，以便作为参数传递
opts = Namespace(**opts)
# 根据选项初始化pSp模型
net = pSp(opts)
# 将模型设置为评估模式
net.eval()
# 将模型转移到CUDA设备上
net.cuda()
print('Model successfully loaded!')
# 从EXPERIMENT_DATA_ARGS中获取原始图像的路径
image_path = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]["image_path"]
# 加载并将图像转换为RGB格式
original_image = Image.open(image_path).convert("RGB")
# 将原始图像的大小调整为256x256
original_image.resize((256, 256))
# 下载面部特征点检测模型
<<<<<<< HEAD
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
=======
# !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# !bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
>>>>>>> a531946 (done)

# 定义对图像进行对齐的函数
def run_alignment(image_path):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

# 对指定路径的图像进行对齐
aligned_image = run_alignment(image_path)
# 将对齐后的图像大小调整为256x256
aligned_image.resize((256, 256))
# 从EXPERIMENT_ARGS获取图像转换流程
img_transforms = EXPERIMENT_ARGS['transform']
# 对对齐后的图像应用转换流程，以便后续作为模型输入
input_image = img_transforms(aligned_image)

# 定义目标年龄，对于每个年龄点，生成一个转换后的图像
target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# 为每个目标年龄创建一个年龄转换器实例
age_transformers = [AgeTransformer(target_age=age) for age in target_ages]
# 定义一个函数，用于将输入图像通过网络进行转换
def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch

# 初始化结果图像为对齐后的图像，并调整其大小到1024x1024以便后续拼接
results = np.array(aligned_image.resize((1024, 1024)))
# 遍历所有的年龄转换器
for age_transformer in age_transformers:
    print(f"Running on target age: {age_transformer.target_age}")
    with torch.no_grad():
        input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
        input_image_age = torch.stack(input_image_age)
        result_tensor = run_on_batch(input_image_age, net)[0]
        result_image = tensor2im(result_tensor)
        results = np.concatenate([results, result_image], axis=1)

results = Image.fromarray(results)

results   # 注意：这是一个非常大的图像（11*1024 x 1024），显示可能需要一些时间

# 将转换后的图像保存为文件
results.save("age_transformed_image.jpg")
