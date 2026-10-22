# 实验三开发说明

## 依据与结构
以 `reference/实验三参考代码.pdf` 中 image_fusion.py 为算法依据，沿用 CS Virtual 实验一、二的五部分报告结构。原参考材料只读。

- `image_fusion.py`：读图、SIFT 提取、FLANN 比率筛选、RANSAC 单应估计、透视拼接及结果保存；各阶段采用独立函数。
- `a.jpg`、`b.jpg`：OpenCV opencv_extra 4.x 中 boat1.jpg、boat2.jpg 的原始文件，不预先裁切或修改。
- `results/`：真实执行输出的图像、单应矩阵、参数与环境记录。
- `screenshots/`：本机 VS Code 的真实截图，仅可裁去无关窗口边缘，不重绘代码或输出。
- `README.md`：沿用前两次实验结构的 Markdown 实验报告和复现入口。
- 截图原始文件和裁切记录保存在本次工作目录的 `work/`，不复制进 CS Virtual。

## 规范与运行范围
复用 Conda cv 环境，保留 SIFT 默认设置、FLANN algorithm=1、trees=5、checks=50、k=2、ratio=0.7、RANSAC 阈值5.0。H 的方向为 b 到 a。原图3888×2592，默认使用INTER_AREA等比例缩小为1200×800，参数与坐标均以缩放后图像为准；max-width=0关闭缩放。重叠区域直接由 a 覆盖，遵循参考代码，不宣称实现无缝融合。增加读图失败、描述符为空、匹配不足和无效单应矩阵的错误处理。画布边界用 floor/ceil 计算，避免负坐标截断丢失边缘。

按用户明确要求进行实际实验运行、结果展示与截图；代码完成后只做语法编译检查，不引入测试套件。运行数据直接读取保存的 JSON，不手填指标。

## 素材来源
https://github.com/opencv/opencv_extra/tree/4.x/testdata/stitching
下载日期：2026-10-22。a.jpg=boat1.jpg，b.jpg=boat2.jpg。仅用于课程实验，非本人拍摄。

## 当前进度
2026-10-22：在本机 CS Virtual/exp3 中复用 cv 环境，从 VS Code 集成终端实际执行完成。Python3.12.14、OpenCV5.0.0、NumPy2.5.3；SIFT关键点2495/1748，FLANN查询2495组，比率测试保留751组，RANSAC内点690组（91.88%），内点平均重投影误差0.3172px，输出1677×930。原始完整数值见results/run_info.json。

保存全量匹配、前80组内点预览、拼接图和8张真实截图。截图仅矩形裁切，移除了指针与屏幕共享标识所在的顶部栏，不重绘代码、不更改结果图。README 明确区分751组全量匹配与80组预览，描述亮度接缝和黑边。

首次运行被 VS Code Python 扩展自动切换环境时中断，随后直接使用 cv 环境解释器的绝对路径运行成功；不修改用户的全局 Conda 初始化配置。为采集清晰终端截图再次实际执行，结果一致。

用户最新要求取消 Word 交付，改为项目内实验目录及 README。只更新实验三及项目首页入口，不提交、不推送，等待用户检查。代码完成语法编译检查，未增加测试套件。
