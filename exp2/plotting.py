"""保存并显示实验对比图，供脚本和 Notebook 共用。"""

from pathlib import Path

import matplotlib.pyplot as plt


def show_grid(images, titles, output_path, rows=1, figsize=(12, 4)):
    if len(images) != len(titles) or not images or len(images) % rows:
        raise ValueError("图片、标题数量及行数不匹配")
    columns = len(images) // rows
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    for axis, image, title in zip(axes.flat, images, titles):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
