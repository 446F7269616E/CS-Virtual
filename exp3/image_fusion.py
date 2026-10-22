"""实验三：依据参考代码，使用 SIFT、FLANN、RANSAC 完成图像拼接。"""
from pathlib import Path
import argparse
import json
import platform
import sys
import cv2
import numpy as np


def read_image(path, max_width=1200):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    original_size = list(image.shape[:2][::-1])
    if max_width > 0 and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return image, original_size


def extract_features(img_a, img_b):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or min(len(des_a), len(des_b)) < 2:
        raise ValueError("图像中的有效特征不足")
    return kp_a, des_a, kp_b, des_b


def match_features(des_a, des_b):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_a, des_b, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    if len(good_matches) < 4:
        raise ValueError("优质匹配不足四组，无法估计单应矩阵")
    return matches, good_matches


def estimate_homography(kp_a, kp_b, good_matches):
    # queryIdx 来自 a，trainIdx 来自 b；计算 b -> a 的变换。
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches])
    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = dst_pts.reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or not np.isfinite(H).all() or mask is None:
        raise ValueError("单应矩阵估计失败")
    if int(mask.sum()) < 4:
        raise ValueError("RANSAC 内点不足四组")
    projected = cv2.perspectiveTransform(src_pts, H)
    errors = np.linalg.norm(projected - dst_pts, axis=2).ravel()
    return H, mask.ravel().astype(bool), errors


def stitch_images(img_a, img_b, H):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    corners_b = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]])
    transformed = cv2.perspectiveTransform(corners_b.reshape(-1, 1, 2), H)
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]])
    all_corners = np.concatenate([transformed.reshape(-1, 2), corners_a])
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    width, height = int(x_max - x_min), int(y_max - y_min)
    if width <= 0 or height <= 0 or width * height > 50_000_000:
        raise ValueError("变换后的画布尺寸异常")
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
                           dtype=np.float64)
    stitched = cv2.warpPerspective(img_b, translation @ H, (width, height))
    stitched[-y_min:h_a - y_min, -x_min:w_a - x_min] = img_a
    return stitched, [int(x_min), int(y_min), int(x_max), int(y_max)]


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"图像保存失败：{path}")


def run_experiment(base_dir, max_width=1200, show=False):
    cv2.setRNGSeed(42)
    cv2.setNumThreads(1)
    output = base_dir / "results"
    output.mkdir(exist_ok=True)
    img_a, original_a = read_image(base_dir / "a.jpg", max_width)
    img_b, original_b = read_image(base_dir / "b.jpg", max_width)
    kp_a, des_a, kp_b, des_b = extract_features(img_a, img_b)
    matches, good = match_features(des_a, des_b)
    H, inliers, errors = estimate_homography(kp_a, kp_b, good)
    stitched, bounds = stitch_images(img_a, img_b, H)
    draw_flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    matched = cv2.drawMatches(img_a, kp_a, img_b, kp_b, good, None,
                             matchColor=(0, 255, 0), flags=draw_flags)
    # 全量匹配照常保存，另取距离最小的80组内点便于观察细节。
    best = sorted([m for m, keep in zip(good, inliers) if keep],
                  key=lambda m: m.distance)[:80]
    preview = cv2.drawMatches(img_a, kp_a, img_b, kp_b, best, None,
                             matchColor=(0, 255, 0), flags=draw_flags)
    save_image(output / "01_inputs.jpg", np.hstack([img_a, img_b]))
    save_image(output / "02_matches_all.png", matched)
    save_image(output / "03_inliers_top80.png", preview)
    save_image(output / "04_stitched.png", stitched)
    info = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "opencv": cv2.__version__, "numpy": np.__version__,
        "system": platform.platform(), "original_sizes_wh": [original_a, original_b],
        "input_sizes_wh": [list(img_a.shape[1::-1]), list(img_b.shape[1::-1])],
        "parameters": {"max_width": max_width, "trees": 5, "checks": 50,
                       "k": 2, "ratio": 0.7, "ransac_threshold_px": 5.0, "seed": 42},
        "keypoints_a": len(kp_a), "keypoints_b": len(kp_b),
        "knn_groups": len(matches), "good_matches": len(good),
        "inliers": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
        "inlier_mean_error_px": float(errors[inliers].mean()),
        "H_b_to_a": H.tolist(), "canvas_bounds": bounds,
        "stitched_size_wh": list(stitched.shape[1::-1]),
        "source": "opencv/opencv_extra 4.x testdata/stitching boat1.jpg boat2.jpg"
    }
    (output / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    log = (f"Python {info['python']} | OpenCV {info['opencv']} | NumPy {info['numpy']}\n"
           f"Input A/B: {img_a.shape[1]} x {img_a.shape[0]}\n"
           f"SIFT keypoints A/B: {len(kp_a)} / {len(kp_b)}\n"
           f"FLANN k=2 groups: {len(matches)} | Lowe ratio=0.7: {len(good)}\n"
           f"RANSAC inliers: {inliers.sum()}/{len(good)} ({inliers.mean():.2%})\n"
           f"Inlier mean reprojection error: {errors[inliers].mean():.4f} px\n"
           f"H (b -> a):\n{H}\n"
           f"Stitched size: {stitched.shape[1]} x {stitched.shape[0]}\n"
           "Saved: results/01_inputs.jpg, 02_matches_all.png,\n"
           "       03_inliers_top80.png, 04_stitched.png, run_info.json\n")
    (output / "run.txt").write_text(log)
    print(log)
    if show:
        for title, image in [("Matched SIFT Keypoints", preview), ("Stitched Result", stitched)]:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, image)
            cv2.resizeWindow(title, 1100, round(image.shape[0] * 1100 / image.shape[1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-width", type=int, default=1200)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run_experiment(Path(__file__).resolve().parent, args.max_width, args.show)
