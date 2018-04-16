import tensorflow as tf
import cv2
import functools
import numpy as np
import math
from .clockwise import clockwise


# Use a custom OpenCV function to read the image

# TODO 求出每个gt的最小外接矩形 ，cv2.minAreaRect(cnt)
# TODO
def opencv_handle(image_path, gt_path):
    image = cv2.imread(image_path.decode())
    img_reized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    img_info = np.array(img_reized.shape, dtype=np.uint8)

    resize_info = np.array([image.shape[0] / 512, image.shape[1] / 512], dtype=np.float32)
    # print(resize_info)
    # print(img_info)
    corner_data = gt_text_to_corner_point(gt_path.decode())
    return img_reized, corner_data, img_info, resize_info


def gt_text_to_corner_point(path):
    with open(path, 'r') as f:
        context = f.readlines()

    rearrange_idx = np.array([2, 3, 0, 1])
    corner_data = np.zeros((len(context), 4, 4), dtype=np.float32)
    for idx, line in enumerate(context):
        line = list(map(float, line.split(',')[:8]))
        line = clockwise(line)
        cnt = np.array(list(zip(line[0::2], line[1::2])))
        minRect = cv2.minAreaRect(cnt)
        short_side = min(minRect[1])
        rect = cv2.boxPoints(minRect)
        # 按照左上 右上 右下 左下的顺序重新排列
        rect = rect[rearrange_idx]
        _ss = np.array([short_side] * 4).reshape((4, 1))
        # 给每个角点加上短边长
        corner_box = np.append(rect, _ss, axis=1)
        corner_box = np.append(corner_box, _ss, axis=1)
        # print(corner_box)
        # np.append(corner_data, corner_box)
        corner_data[idx, :, :] = corner_box

    # print(corner_data[:, :1, :])
    # print(corner_data.transpose((1, 0, 2))[:1])
    # 将 (num_gt,4,4) 重新排列成 (4, num_gt,4)
    """
       0：左上
       1：右上
       2：右下
       3：左下
    """
    return corner_data.transpose((1, 0, 2))
