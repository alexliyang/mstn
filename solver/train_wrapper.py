import tensorflow as tf
import os
from lib import get_path
from lib import Timer


class TrainWrapper(object):
    def __init__(self, proj_path, cfg, network = None):
        self._cfg = cfg
        self._root_path = proj_path
        self.net = network
        # self.output_dir = output_dir

        pretrained_model_path = os.path.join(self._root_path,
                                             cfg.COMMON.DATA_PATH, cfg.TRAIN.PRETRAIN)
        pretrained_model = os.listdir(pretrained_model_path)

        assert len(pretrained_model) == 1, 'pretrain model should be one'
        self._pretrain = os.path.join(pretrained_model_path, pretrained_model[0])

        self._ckpt_path = get_path(os.path.join(self._root_path, cfg.COMMON.DATA_PATH, cfg.TRAIN.CKPT))
        self._restore = cfg.TRAIN.RESTORE
        self._max_iter = cfg.TRAIN.MAX_ITER

    def snapshot(self):
        pass

    def build_loss(self):
        pass

    def train_model(self, producer=None, sess=None, max_iters=None, restore=False):

        iterator = producer.make_one_shot_iterator()
        next_element = iterator.get_next()

        timer = Timer()

        wrong = 0
        for _ in range(10):
            while True:
                try:
                    timer.tic()
                    img, corner_data, img_info, reize_info,segmentation_mask = sess.run(next_element)
                    # print(img.shape)
                    # print(corner_data.shape)
                    # print(img_info)
                    # print(reize_info)
                    print(timer.toc())
                    break
                except tf.errors.OutOfRangeError:
                    break
                except:
                    # print(e)
                    wrong += 1
                    print('get batch error')
                    break
        print(wrong)