import numpy as np
# 图像数据的均值和标准差，用于对图像进行标准化处理
# 图像的每个颜色通道（红色、绿色、蓝色）减去均值并除以标准差，使图像的像素值分布更为稳定
IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD = [58.395, 57.12, 57.375]

# get_colors函数的作用：返回一个颜色列表的变换结果
def get_colors():

    RGB_tuples = np.array(colors) / 255.# 将颜色值从[0, 255]范围缩放到[0, 1]范围
    RGB_tuples = RGB_tuples[:, [2, 1, 0]]# 将 RGB 顺序调整为 BGR（OpenCV默认）
    # 交换颜色通道，生成颜色库
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [2, 0, 1]]],
                                axis=0)
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [1, 2, 0]]],
                                axis=0)
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [0, 2, 1]]],
                                axis=0)
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [2, 1, 0]]],
                                axis=0)
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [1, 0, 2]]],
                                axis=0)

    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [2, 2, 1]]],
                                axis=0)
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [1, 2, 2]]],
                                axis=0)
    RGB_tuples = np.concatenate([RGB_tuples, RGB_tuples[1:8, [2, 0, 2]]],
                                axis=0)

    return RGB_tuples

# 定义一些基础的RGB颜色
# 在图片上画BBox或Mask时，需要为不同的类别分配不同的颜色
colors = [
    [
        255.0,
        255.0,
        255.0,
    ],
    [83, 154, 251],
    [91, 153, 251],
    [99, 152, 251],
    [107, 151, 251],
    [114, 150, 251],
    [122, 149, 251],
    [130, 148, 251],
    # [138, 147, 251],
    # [146, 146, 251],
    # [154, 145, 251],
    # [162, 144, 251],
    # [170, 143, 251],
    # [178, 142, 251],
    # [186, 141, 251],
    # [194, 140, 251],
    # [202, 139, 251],
    # [210, 138, 251],
    # [218, 137, 251],
]
