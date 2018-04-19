import numpy as np
import os
from lib import bbox_overlaps
from lib import get_config

proj_path = os.path.abspath(os.curdir)
cfg = get_config(proj_path, 'configure.yml')


def corner_py(corner_pred_score, corner_pred_offset, gt_default_box, scales, feat_stride, img_info):
    # TODO 要把输入的tensor 转换一下
    """(num_scales, 4)
       gt_default_box: (4, every corner box number, 4)
                     : 0 left top,
                     : 1 right top
                     : 2 right bottom
                     : 3 left bottom
    """

    assert corner_pred_score.shape[0] == 1, \
        'Only single item batches are supported'

    # q
    num_scales = len(scales)

    per_cell_db = np.array([[0 for _ in scales], [0 for _ in scales], scales, scales], np.int32).transpose()

    height, width = corner_pred_score.shape[1:3]

    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order

    padding_zeros = np.zeros(len(shift_x), np.int32)

    # TODO 每个default box 的表示方法（x,y,ss,ss）ss为scale
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), padding_zeros, padding_zeros)).transpose()

    """shift_x 是展平后的x值 每个x值对应一个pixel的横坐标"""
    all_pixel_num = shift_x.shape[0]  # the number of pixel in feature map
    """为每个cell都生成了num_scale 个 default box
       all_default_box shape (H*W，num_scales, 4)
    """
    all_default_box = (per_cell_db.reshape((1, num_scales, 4)) +
                       shifts.reshape((1, all_pixel_num, 4)).transpose((1, 0, 2)))

    """ all_defalut_box (H*W*num_scales, 4)"""
    all_default_box = all_default_box.reshape(all_pixel_num * num_scales, 4)

    """filter the db out of  the image"""
    idx_indside = np.where(
        (all_default_box[:, 0] - all_default_box[:, 2] / 2) >= 0 &
        (all_default_box[:, 0] + all_default_box[:, 2] / 2 <= img_info[1]) &
        (all_default_box[:, 1] - all_default_box[:, 3] / 2 >= 0) &
        (all_default_box[:, 1] + all_default_box[:, 3] / 2 <= img_info[0])
    )[0]

    default_boxes = all_default_box[idx_indside, :]

    """给 每个default box 匹配一个 corner box"""
    """"""

    """ 对于每个default box ,需要计算它和每种corner box 真值的iou，来确定它是否是属于某种corner point

    """
    # all_overlaps = np.zeros(())
    # gt_default_box shape: (4, gt_text_num, 4) 0 for left_top ...etc

    valid_pixel_num = len(idx_indside)
    # 需要返回的 labels 为 (N H W num_scales q=4 1), 后面再reshape
    labels = np.empty((height, width, num_scales, 4, 1))
    # 需要返回的 box_target 为 （N, H, W, num_scales, q, 4) 后面再reshape
    box_target = np.empty((height, width, num_scales, 4, 4))

    labels.fill(-1)
    """
       gt_default_box shape(4, num_gt_text, 4)
    """
    for ix, gt_corner_box in enumerate(gt_default_box):
        # overlap 返回的 shape (valid_pixel_num * num_scales, gt_box_num)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(default_boxes, dtype=np.float),
            np.ascontiguousarray(gt_corner_box, dtype=np.float))
        # argmax_overlap (valid_pixel_num * num_scales, 1)
        argmax_overlaps = overlaps.argmax(axis=1)  # 找到和每一个gtbox，overlap最大的那个db

        # valid_label 所有有效像素个数 * 每个像素上的scale个数
        valid_label = np.empty((valid_pixel_num * num_scales,), np.int8)
        valid_label.fill(-1)

        max_overlaps = overlaps[np.arange(valid_pixel_num), argmax_overlaps]

        # 最大iou < 0.3 的设置为负例
        valid_label[max_overlaps < cfg.TRAIN.NEGATIVE_OVERLAP] = 0
        # cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.8
        valid_label[max_overlaps >= cfg.TRAIN.POSITIVE_OVERLAP] = 1  # overlap大于0.8的认为是前景

        per_kind_corner_label = np.empty((height * width * num_scales,), np.int8)
        per_kind_corner_label.fill(-1)

        per_kind_corner_label[idx_indside] = valid_label

        labels[:, :, :, ix, :] = per_kind_corner_label.reshape(height, width, num_scales, 1, 1)

        ########################### box target ##################################
        # TODO 对于每个真值是1的default box 需要它有回归目标
        positive_inds = np.where(valid_label == 1)[0]
        per_kind_corner_target = np.empty((height * width * num_scales, 4), np.int32)

        per_kind_corner_target.fill(0)

        # argmax为每个default box对应iou最大的那个gt的下标，从中选出label是正的
        per_kind_corner_target[positive_inds, :] = gt_corner_box[argmax_overlaps[positive_inds]]

        box_target[:, :, :, ix, :] = per_kind_corner_target.reshape(height, width, num_scales, 1, 4)

    num_fg = int(cfg.TRAIN.POSITIVE_RATIO * cfg.TRAIN.DEFAULT_BOX_NUM)  # 0.25*300

    flat_label = labels.reshape((height * width * num_scales * 4,))
    fg_inds = np.where(flat_label == 1)[0]

    assert len(fg_inds) > 0, "The number of positive proposals must be lager than zero"

    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
        flat_label[disable_inds] = -1  # 变为-1

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是300，限制正样本数目最多150，
    num_bg = cfg.TRAIN.DEFAULT_BOX_NUM - np.sum(flat_label == 1)

    bg_inds = np.where(flat_label == 0)[0]

    assert len(bg_inds) > 0, "The number of negtive proposals must be lager than zero"

    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        flat_label[disable_inds] = -1

    """
     labels (1, height, width, num_scales, 4, 1)
     box_target (1, height, width, num_scales, 4, 4)
    """
    # return flat_label.reshape((1, height, width, num_scales, 4, 1))
    return labels, box_target
