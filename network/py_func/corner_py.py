import numpy as np

# from lib.bbox.bbox import bbox_overlaps


def corner_py(corner_pred_score, corner_pred_offset, gt_default_box, scales, feat_stride, img_info):
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
    shifts = np.vstack((shift_x.ravel() , shift_y.ravel(), padding_zeros, padding_zeros)).transpose()

    """shift_x 是展平后的x值 每个x值对应一个pixel的横坐标"""
    K = shift_x.shape[0]  # the number of pixel in feature map
    """为每个cell都生成了num_scale 个 default box
       all_default_box shape (像素个数，num_scales, 4)
    """
    all_default_box = (per_cell_db.reshape((1, num_scales, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))

    """ all_defalut_box (k*num_scales, 4)"""
    all_default_box = all_default_box.reshape(K * num_scales, 4)

    """filter the db out of  the image"""
    idx_indside = np.where(
        (all_default_box[:, 0] - all_default_box[:, 2] / 2) >= 0 &
        (all_default_box[:, 0] + all_default_box[:, 2] / 2 <= img_info[1]) &
        (all_default_box[:, 1] - all_default_box[:, 3] / 2 >= 0) &
        (all_default_box[:, 1] + all_default_box[:, 3] / 2 <= img_info[0])
    )

    default_boxes = all_default_box[idx_indside, :]

    """给 每个default box 匹配一个 corner box"""
    """"""

    """ 对于每个default box ,需要计算它和每种corner box 真值的iou，来确定它是否是属于某种corner point

    """
    # all_overlaps = np.zeros(())
    for gt_corner_box in gt_default_box:
        overlaps = bbox_overlaps(
            np.ascontiguousarray(default_boxes, dtype=np.float),
            np.ascontiguousarray(gt_corner_box, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=0) #找到和每一个gtbox，overlap最大的那个db


def generate_db(scales):
    pass
