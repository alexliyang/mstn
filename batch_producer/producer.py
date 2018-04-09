import tensorflow as tf
import os
from .preprocess import parse_function
from .preprocess import opencv_read


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
        # dataset = dataset.map(
        #     lambda filename, label: tuple(tf.py_func(
        #         opencv_read, [filename, label], [tf.uint8, label.dtype])))
        dataset = dataset.map(parse_function).repeat(50).batch(cfg.TRAIN.BATCH_SIZE)
        self._producer = dataset

    @property
    def producer(self):
        return self._producer

    def _get_text_name(self, img_name):
        return os.path.splitext(img_name)[0] + '.txt'
