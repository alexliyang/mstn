import tensorflow as tf
import os
from lib import get_path
from lib import Timer

timer = Timer()


class TrainWrapper(object):
    def __init__(self, proj_path, cfg, network=None):
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

        # store the params
        # self.saver = tf.train.Saver(max_to_keep=10, write_version=tf.train.SaverDef.V2)

    def get_optimizer(self, lr):

        if self._cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(self._cfg.TRAIN.LEARNING_RATE)
        elif self._cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(self._cfg.TRAIN.LEARNING_RATE)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = self._cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)
        return opt

    def snapshot(self):
        pass

    def build_loss(self):
        pass

    def setup(self, sess=None, producer=None, restore=False):
        loss = (total_loss, model_loss, all_cross_entropy, all_regression_loss) = \
            self.net.build_loss(ohem=self._cfg.TRAIN.OHEM)

        # get the batch iterater
        iterator = producer.make_one_shot_iterator()
        next_iterater = iterator.get_next()

        # get the optimizer
        lr = tf.Variable(self._cfg.TRAIN.LEARNING_RATE, trainable=False)
        opt = self.get_optimizer(lr)

        global_step = tf.Variable(0, trainable=False)

        # get the train_op
        tvars = tf.trainable_variables()
        grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
        train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)

        # initiate the varables
        sess.run(tf.global_variables_initializer())

        try:
            print(('Loading pretrained model '
                   'weights from {:s}').format(self._pretrain))
            self.net.load(self._pretrain, sess, True)
        except:
            raise 'Check your pretrained model {:s}'.format(self._pretrain)

        restore_iter = 0
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self._ckpt_path)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise 'Check your pretrained {:s}'.format(self._ckpt_path)

        return next_iterater, train_op, restore_iter, loss, lr

    def console_log(self, iter, max_iters, loss, lr):
        print(
            'iter: %d / %d, total loss: %.4f, model loss: %.4f, cls_cross_entropy: %.4f, box_regression_loss: %.4f, lr: %f' % \
            (iter, max_iters, loss[0], loss[1], loss[2],
             loss[3], lr.eval()))
        print('speed: {:.3f}s / iter'.format(timer.toc()))

    def train_model(self, cfg=None, producer=None, sess=None, max_iters=None, restore=False):

        next_batch, train_op, restore_iter, loss, lr = self.setup(sess=sess, producer=producer, restore=restore)
        total_loss, model_loss, all_cross_entropy, all_regression_loss = loss

        iterator = producer.make_one_shot_iterator()
        next_batch = iterator.get_next()

        # for iter in range(restore_iter, max_iters):
        for iter in range(1):
            # learning rate
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))

            while True:
                try:
                    timer.tic()
                    # get the batch data
                    img, corner_data, img_info, resize_info, segmentation_mask = sess.run(next_batch)

                    print(img.shape)
                    print(corner_data.shape)
                    print(img_info.shape)
                    print(resize_info.shape)
                    print(segmentation_mask.shape)

                    feed_dict = {
                        self.net.img: img,
                        self.net.corner_data: corner_data,
                        self.net.img_info: img_info,
                        self.net.resize_info: resize_info,
                        self.net.segmentation_mask: segmentation_mask,
                    }

                    fetch_list = [total_loss, model_loss, all_cross_entropy, all_regression_loss,
                                  train_op]

                    loss_val = [total_loss_val, model_loss_val, all_cross_entropy_val, all_regression_loss_val] \
                        = sess.run(fetches=fetch_list, feed_dict=feed_dict)

                    if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                        self.console_log(iter, max_iters, loss_val, lr)

                    # if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                    #     last_snapshot_iter = iter
                    #     self.snapshot(sess, iter)

                    break
                except tf.errors.OutOfRangeError:
                    break
                except:
                    # print(e)
                    print('get batch error')
                    break
