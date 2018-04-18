import tensorflow as tf
import cv2
import functools
import numpy as np
import math
from .clockwise import clockwise
import os

from lib import get_config

proj_path = os.path.abspath(os.curdir)
cfg = get_config(proj_path, 'configure.yml')

# Use a custom OpenCV function to read the image

# TODO 求出每个gt的最小外接矩形 ，cv2.minAreaRect(cnt)
# TODO

RESIZE_X = cfg.COMMON.RESIZE_WIDTH
RESIZE_Y = cfg.COMMON.RESIZE_HEIGHT


def opencv_handle(image_path, gt_path):
    image = cv2.imread(image_path.decode())
    # img_reized = image
    # print(image_path.decode())
    img_reized = cv2.resize(image, (RESIZE_X, RESIZE_X), interpolation=cv2.INTER_CUBIC)
    img_info = np.array(img_reized.shape, dtype=np.int32)

    # shape (h,w,c)
    # resize_info (y_resize_scale, x_resize_scale)
    resize_info = np.array([image.shape[0] / RESIZE_Y, image.shape[1] / RESIZE_X], dtype=np.float32)
    # print(resize_info)
    # print(img_info)
    # convert the gt_text to corner data and segmentation mask
    corner_data, segmentation_mask = gt_text_handler(gt_path.decode(), img_info, resize_info)
    return img_reized, corner_data, img_info, resize_info, segmentation_mask


"""

"""


def gt_text_handler(path, img_info, resize_info):
    with open(path, 'r') as f:
        context = f.readlines()

    rearrange_idx = np.array([2, 3, 0, 1])
    corner_data = np.zeros((len(context), 4, 4), dtype=np.int32)
    for idx, line in enumerate(context):
        line = list(map(float, line.split(',')[:8]))
        line = clockwise(line)
        # 得到缩放后的四边形
        cnt = np.array(list(zip(
            [x / resize_info[1] for x in line[0::2]],
            [x / resize_info[0] for x in line[1::2]])))
        minRect = cv2.minAreaRect(cnt)
        short_side = min(minRect[1])
        rect = cv2.boxPoints(minRect)
        # print(rect)
        # 按照左上 右上 右下 左下的顺序重新排列
        rect = rect[rearrange_idx]
        _ss = np.array([short_side] * 4).reshape((4, 1))
        # 给每个角点加上短边长
        corner_box = np.append(rect, _ss, axis=1)
        corner_box = np.append(corner_box, _ss, axis=1)
        # print(corner_box)
        # np.append(corner_data, corner_box)
        corner_data[idx] = corner_box

    # print(corner_data[:, :1, :])
    # print(corner_data.transpose((1, 0, 2))[:1])

    """
       corner_data
       0：左上角点框
       1：右上角点框
       2：右下角点框
       3：左下角点框
    """

    segmentation_mask = np.zeros((4, img_info[0], img_info[1]), dtype=np.int32)

    split_points = split_bin(corner_data[:, :, :2])

    left_top = np.array([0, 4, 8, 7])
    right_top = np.array([4, 1, 5, 8])
    right_bottom = np.array([8, 5, 2, 6])
    left_bottom = np.array([7, 8, 6, 3])
    position_grid = [left_top, right_top, right_bottom, left_bottom]

    # 对mask 的每层所对应的segment bin区域进行赋值1

    # TODO 按照说明使用cv2.fillPoly(segmentation_mask[i], split_points[:, position_grid[i], :], 1)
    # TODO 失败了，不知道是什么原因，只能先暂时使用两个for 循环
    for j in range(len(split_points)):
        for i in range(4):
            # print(split_points[:, position_grid[i]])
            cv2.fillPoly(segmentation_mask[i], [split_points[j, position_grid[i], :]], 1)

    # 将 (num_gt,4,4) 重新排列成 (4, num_gt,4)
    return corner_data.transpose((1, 0, 2)), segmentation_mask


def split_bin(rects):
    points = np.zeros((rects.shape[0], 9, 2), dtype=np.int32)
    #
    points[:, :4, :] = rects
    points[:, 4, :] = (rects[:, 0, :] + rects[:, 1, :]) / 2
    points[:, 5, :] = (rects[:, 1, :] + rects[:, 2, :]) / 2
    points[:, 6, :] = (rects[:, 2, :] + rects[:, 3, :]) / 2
    points[:, 7, :] = (rects[:, 0, :] + rects[:, 3, :]) / 2
    points[:, 8, :] = (rects[:, 0, :] + rects[:, 2, :]) / 2

    # print(points)

    return points
