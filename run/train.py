import os
import sys
import tensorflow as tf

sys.path.append(os.getcwd())

from lib import get_config
from batch_producer import producer
from solver import TrainWrapper
from network import get_network

proj_path = os.path.abspath(os.curdir)
cfg = get_config(proj_path, 'configure.yml')


def make_TFconfig():
    TFconfig = tf.ConfigProto(allow_soft_placement=True)
    TFconfig.gpu_options.allocator_type = 'BFC'
    TFconfig.gpu_options.per_process_gpu_memory_fraction = 0.9

    return TFconfig


if __name__ == '__main__':
    batch_producer = producer(proj_path, cfg)

    train_net = get_network('train')

    with tf.Session(config=make_TFconfig()) as sess:
        sw = TrainWrapper(proj_path, cfg, train_net)

        sw.train_model(sess=sess, producer=batch_producer.producer,
                       max_iters=cfg.TRAIN.MAX_ITER, restore=False)
