# 实验一：计算机视觉库的安装与环境配置

## 一、实验目的

1. 掌握 Anaconda 的安装与基本操作。
2. 熟悉 conda 虚拟环境的创建、激活与管理。
3. 完成 Python 计算机视觉实验环境的搭建。
4. 完成 OpenCV 的安装并验证其是否可以正常使用。
5. 为后续计算机视觉实验提供统一、独立的 Python 运行环境。

---

## 二、实验环境

- 操作系统：macOS
- 处理器：Apple Silicon
- 包管理工具：Homebrew
- Python 环境管理工具：Anaconda / conda
- Python 版本：Python 3.12
- 虚拟环境名称：`cv`
- 计算机视觉库：OpenCV

---

## 三、实验内容

### 1. Anaconda 的安装及配置

本实验使用 Homebrew 安装 Anaconda。

在终端中执行：

```bash
brew install --cask anaconda
```

安装完成后，对 conda 进行 shell 初始化：

```bash
/opt/homebrew/anaconda3/bin/conda init zsh
```

重新加载当前 shell：

```bash
exec zsh
```

使用以下命令查看 conda 版本：

```bash
conda --version
```

并查看 conda 当前配置信息：

```bash
conda config --show
```

运行上述命令后能够正常输出 conda 的版本及配置信息，说明 Anaconda 与 conda 已成功安装并可以正常使用。

**实验截图：**

![](assets/Pasted%20image%2020260924134832.png)

---

### 2. 创建 Python 虚拟环境

为了使本课程的依赖与其他 Python 项目相互独立，本实验使用 conda 创建专用的计算机视觉虚拟环境。

执行：

```bash
conda create -n cv python=3.12 -y
```

其中：

- `-n cv`：将虚拟环境命名为 `cv`
- `python=3.12`：指定环境中的 Python 版本为 3.12
- `-y`：自动确认安装过程

创建完成后，使用：

```bash
conda env list
```

查看当前系统中的 conda 环境。

输出结果中可以看到名为 `cv` 的环境，说明虚拟环境创建成功。

**实验截图：**


![](assets/Pasted%20image%2020260924135909.png)
![](assets/Pasted%20image%2020260924140214.png)

---

### 3. 激活虚拟环境

使用以下命令进入刚刚创建的 `cv` 环境：

```bash
conda activate cv
```

环境激活后，终端提示符前会出现：

```text
(cv)
```

随后查看当前 Python 版本：

```bash
python --version
```

能够看到当前环境使用的是 Python 3.12。

这说明终端当前使用的 Python 解释器已经切换至 `cv` 虚拟环境。

**实验截图：**


![](assets/Pasted%20image%2020260924140348.png)

---

### 4. OpenCV 的安装

进入 `cv` 环境后，使用 pip 安装 OpenCV：

```bash
pip install opencv-python
```

安装完成后使用：

```bash
pip list
```

查看当前环境已经安装的软件包。

在软件包列表中可以看到：

```text
opencv-python
numpy
```

说明 OpenCV 及其相关依赖已经成功安装。

**实验截图：**

![](assets/Pasted%20image%2020260924140459.png)



---

### 5. OpenCV 环境验证

为了进一步验证 OpenCV 是否能够被 Python 正常调用，执行：

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

若终端能够正常输出 OpenCV 版本号，例如：

```text
OpenCV: 4.x.x
```

说明当前 Python 环境可以正确加载 OpenCV，计算机视觉实验环境搭建成功。

**实验截图：**

![](assets/Pasted%20image%2020260924140642.png)


---

## 四、实验结果

本实验成功完成了 macOS 环境下 Anaconda、conda 及 OpenCV 的安装与配置。

通过 conda 创建了独立的 `cv` Python 虚拟环境，并在该环境中安装 OpenCV。最终通过 Python 成功导入 `cv2` 模块并输出 OpenCV 的版本号，说明实验环境配置正确，可以用于后续计算机视觉实验。

实验环境结构如下：

```text
Anaconda
└── conda
    └── cv
        ├── Python 3.12
        ├── NumPy
        └── OpenCV
```

后续进行计算机视觉实验时，只需要首先进入该环境：

```bash
conda activate cv
```

即可使用本次实验配置好的 Python 和 OpenCV 环境。

---

## 五、实验总结

本次实验完成了计算机视觉课程所需基础 Python 环境的搭建。

通过本次实验，熟悉了 Homebrew 安装软件的方法，并掌握了 conda 虚拟环境的创建、查看和激活等基本操作。

使用独立虚拟环境能够将不同项目使用的 Python 版本和第三方库相互隔离，从而避免不同项目之间发生依赖冲突。

同时，本实验成功安装并验证了 OpenCV，为后续图像处理、特征提取、图像匹配等计算机视觉实验做好了环境准备。
