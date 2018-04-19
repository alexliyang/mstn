import tensorflow as tf
import os
# from .preprocess import parse_function
from .preprocess import opencv_handle


class producer(object):
    def __init__(self, proj_path, cfg):
        self._root_path = proj_path

        '''prepare path'''
        self._cfg = cfg
        self._abspath = os.path.join(self._root_path, cfg.COMMON.DATA_PATH, cfg.TRAIN.TRAIN_DATA_PATH)
        self._img_path = os.path.join(self._abspath, 'img')
        self._gt_path = os.path.join(self._abspath, 'txt')

        print('Preparing training data......')
        '''img path tensor'''
        img_list = tf.constant([os.path.join(self._img_path, img_name)
                                for img_name in os.listdir(self._img_path)])

        # 根据img list 在其对应的位置上生成gt
        gt_list = tf.constant([os.path.join(self._gt_path, self._get_text_name(img_name))
                               for img_name in os.listdir(self._img_path)])

        assert img_list.shape[0] == gt_list.shape[0], 'Image and ground truth mismatch'

        dataset = tf.data.Dataset.from_tensor_slices((img_list, gt_list))
        dataset = dataset.map(
            lambda img_path, gt_path: tuple(tf.py_func(
                # @return img_reized, corner_data, img_info, resize_info, segmentation_mask
                opencv_handle, [img_path, gt_path], [tf.uint8, tf.int32, tf.int32, tf.float32, tf.int32])),
            num_parallel_calls=100).repeat().batch(
            cfg.TRAIN.BATCH_SIZE)
        # dataset = dataset.map(parse_function).repeat().batch(cfg.TRAIN.BATCH_SIZE)
        self._producer = dataset

    @property
    def producer(self):
        return self._producer

    def _get_text_name(self, img_name):
        return os.path.splitext(img_name)[0] + '.txt'
