"""实验二共用的图像处理函数，彩色输入统一为 uint8 三通道。"""

from pathlib import Path

import cv2
import numpy as np


def read_color_image(path):
    image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    if min(image.shape[:2]) <= 100:
        raise ValueError("图像宽、高都应大于 100，才能读取 [100, 100] 像素")
    return image


def to_uint8(image):
    """将 random_noise 的 [0, 1] 浮点结果转为 OpenCV 可用格式。"""
    return np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def filter_three_ways(image, kernel_size=5):
    return (
        cv2.blur(image, (kernel_size, kernel_size)),
        cv2.medianBlur(image, kernel_size),
        cv2.GaussianBlur(image, (kernel_size, kernel_size), 0),
    )


def manual_median_filter_color(image, kernel_size=5):
    """逐通道、逐像素实现中值滤波；边界使用边缘复制。"""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("请输入 uint8 格式的 H×W×3 彩色图像")
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("滤波窗口必须为正奇数")

    pad = kernel_size // 2
    height, width, channels = image.shape
    filtered_img = np.zeros_like(image)
    for c in range(channels):
        channel = image[:, :, c]
        padded_channel = np.pad(channel, pad_width=pad, mode="edge")
        for i in range(height):
            for j in range(width):
                region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                filtered_img[i, j, c] = np.median(region)
    return filtered_img
