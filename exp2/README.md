# 实验二：图像增强

## 一、实验目的

1. 掌握图像读取及 BGR、RGB、灰度图之间的转换方法。
2. 了解椒盐噪声和高斯噪声的特点，比较不同滤波方法的效果。
3. 手动实现彩色图像的中值滤波，加深对邻域处理和分通道处理的理解。

---

## 二、实验环境

- 操作系统：macOS，Apple Silicon
- 实验工具：VS Code、Jupyter Notebook
- Python 环境：Conda `cv`，Python 3.12.14
- 主要依赖：OpenCV 5.0.0、NumPy 2.5.3、scikit-image 0.26.0、Matplotlib 3.11.2

---

## 三、实验内容

### 1. 导入依赖

导入图像处理、噪声生成和绘图所需的库，并输出版本信息。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
```

![实验环境及运行输出](assets/01_environment.png)

### 2. 图像读取与颜色空间转换

读取彩色图像 `p1.jpg`，获取 `[100, 100]` 处的像素值，并转换为 RGB 图和灰度图。图像宽度为 480 像素，高度为 360 像素。

```python
img = read_color_image(BASE_DIR / "p1.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b, g, r = map(int, img[100, 100])
```

运行结果：

```text
图像形状: (360, 480, 3) 数据类型: uint8
[100, 100] BGR: (182, 213, 234)
[100, 100] RGB: (234, 213, 182)
灰度值: 216
```

OpenCV 按 BGR 顺序读取图像，Matplotlib 按 RGB 顺序显示。直接显示 BGR 数组时，图中橙色部分变成了蓝色；转换为 RGB 后颜色正常。灰度图去掉了颜色信息，但仍能看出主体的轮廓和明暗变化。

![图像读取与颜色转换结果](assets/02_color_spaces.png)

### 3. 添加噪声

在 RGB 图像上分别添加椒盐噪声和高斯噪声。椒盐噪声比例参数为 0.4，高斯噪声均值为 0.2、方差为 0.03。

```python
sp_noise_img = random_noise(rgb_img, mode="s&p", amount=0.4, rng=42)
gus_noise_img = random_noise(
    rgb_img, mode="gaussian", mean=0.2, var=0.03, rng=43,
)
```

椒盐噪声使图像出现大量离散噪点。由于彩色图像的通道值分别受到影响，图中也出现了彩色噪点。高斯噪声产生较均匀的颗粒感，并使图像整体变亮，这是正均值噪声带来的亮度偏移。

![两类噪声的对比结果](assets/03_noise.png)

### 4. 图像滤波

将噪声图转换为 `uint8` 格式，分别使用均值、中值和高斯滤波。滤波窗口均为 5×5。

```python
cv2.blur(image, (kernel_size, kernel_size))
cv2.medianBlur(image, kernel_size)
cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

下图第一行为椒盐噪声的滤波结果，第二行为高斯噪声的滤波结果，三列依次为均值、中值、高斯滤波。

椒盐噪声经过中值滤波后，绿色背景和主体表面的杂点明显减少，轮廓仍比较清楚。均值和高斯滤波虽然减轻了噪声，但还保留较多杂色。

对于高斯噪声，均值和高斯滤波能够减轻颗粒感，中值滤波也有一定效果。滤波后图像变得平滑，但部分细节会模糊。三种滤波方法都没有消除噪声造成的整体亮度偏移。

![六组滤波结果](assets/04_filters.png)

### 5. 手动实现彩色中值滤波

分别处理 R、G、B 三个通道。先复制边缘进行填充，再遍历每个像素，取其周围 5×5 区域的中值作为输出。

```python
for c in range(channels):
    channel = image[:, :, c]
    padded_channel = np.pad(channel, pad_width=pad, mode="edge")
    for i in range(height):
        for j in range(width):
            region = padded_channel[i:i + kernel_size, j:j + kernel_size]
            filtered_img[i, j, c] = np.median(region)
```

![滤波函数及手写中值滤波代码](assets/06_filter_source.png)

对椒盐噪声图进行手写中值滤波，并与 OpenCV 的中值滤波结果进行比较。

```text
手写中值滤波耗时: 1.98 秒
与 OpenCV 中值滤波逐像素相同: True
最大像素差: 0
```

两种方法的图像结果一致，主体颜色和轮廓得到较好保留，背景中的大部分噪点被去除。

![手写中值滤波的运行结果](assets/05_manual_result.png)

---

## 四、实验结果

本实验完成了图像读取、颜色空间转换、两类噪声添加及三种滤波方法的比较。

中值滤波对本次椒盐噪声的去除效果最明显，能够减少孤立噪点并保留主体轮廓。均值和高斯滤波能够平滑高斯噪声，但也会损失部分细节。手写彩色中值滤波与 OpenCV 的处理结果逐像素一致，最大像素差为 0。

---

## 五、实验总结

通过本次实验，我熟悉了 OpenCV 图像处理的基本流程。显示图像时需要注意 BGR 和 RGB 的通道顺序，处理噪声图时也要注意浮点数据与 `uint8` 数据的转换。不同类型的噪声适合不同的滤波方法，去噪效果和细节保留之间需要进行取舍。手动实现中值滤波后，我对滑动窗口、边界填充和彩色图像分通道处理有了更直观的理解。
