import numpy as np
import tensorflow as tf
from lib import get_config
from network.py_func import corner_py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import os

proj_path = os.path.abspath(os.curdir)
cfg = get_config(proj_path, 'configure.yml')

DEFAULT_PADDING = 'SAME'


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.

        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class BaseNetwork(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []  # cache for layer input
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()  # init

    # must be dereived
    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    """load the pre-train model"""

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path, encoding='latin1').item()

        """ key和subkey的例子
        key: conv5_3 -| weights 
                         | biases
        """
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model " + subkey + " to " + key)
                    except ValueError:
                        print("ignore " + key)
                        if not ignore_missing:
                            raise

    # clean the self.input and put some value(input data or some previous layer output)
    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in list(self.layers.items())) + 1
        return '%s_%d' % (prefix, id)

    # make a tensor variable
    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    """
    name: f11_pred
    """

    def predict_module(self, name=None):
        (self.feed(name)
         .conv(1, 1, 256, name=name + '_conv1')
         .conv(1, 1, 256, name=name + '_conv2')
         .conv(1, 1, 1024, name=name + '_conv3'))

        self.feed(name).conv(1, 1, 1024, name=name + '_conv4')

        # return the element wise sum
        return tf.add(self.get_output(name + '_conv3'), self.get_output(name + '_conv4'))

    # TODO deconv module in DSSD name: f11 f10 f9 ...
    def deconv_module(self, name=None, module_num=None):
        """
        :param name: f10 f9...
        :return:
        """
        d = {
            'f10': ['f11', 'pool10'],
            'f9': ['f10', 'pool9'],
            'f8': ['f9', 'pool8'],
            'f7': ['f8', 'pool7'],
            'f4': ['f7', 'pool4'],
            'f3': ['f4', 'pool3'],
        }

        deconv_layer = d[name][0]
        feature_layer = d[name][1]

        (self.feed(deconv_layer)
         .deconv(2, 2, 512, 1, 1, name=name + '_deconv1')
         .conv(3, 3, 512, 1, 1, name=name + '_conv1')
         .batch_normalize(name=name + '_deconv_output'))

        (self.feed(feature_layer)
         .conv(3, 3, 1024, 1, 1, name=name + '_conv2')
         .batch_normalize().relu()
         .conv(3, 3, 1024, 1, 1, name=name + 'conv3')
         .batch_normalize(name=name + '_feature'))

        return tf.multiply(self.get_output(name + '_deconv_output'), self.get_output(name + '_feature'))\
            .relu(name=name)



    @layer
    def corner_detect_layer(self, input, scales=None, name=None, feat_stride=None, img_info=None):
        assert scales and name and feat_stride, 'corner detect layer lack some augment'
        """
        input[0]: ground truth
        input[1]: image info
        """
        with tf.variable_scope(name) as scope:
            """
            :params 0: corner label and its confidence corner_pred_score
                    1: offset target to regression corner_pred_offset
                    2: gt_default_box
                    3: scales
                    4: feat stride
                    5: img_info 
            """
            corner_pred_score, corner_pred_offset = \
                tf.py_func(corner_py, [input[0], input[1],input[3], scales, feat_stride, input[2]],
                           [tf.float32, tf.float32])

            # TODO corner label shape w, h , k, q * 2
            rpn_labels = tf.convert_to_tensor(tf.cast(corner_pred_score, tf.int32),
                                              name='rpn_labels')  # shape is (1 x H x W x A, 2)



    # TODO 输入是一幅特征图input: N , H, W, d_i, 全连接使得 使得网络输出 N , H, W, k, q, d_o
    def deconv_fc(self, d_i, d_o, input, name=None, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            input = tf.reshape(input, [N * H * W, C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [d_i, d_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(input, kernel) + biases
            return tf.reshape(_O, [N, H, W, int(d_o)])

    @layer
    def batch_normalize(self, input, training=True):
        axis = -1
        return tf.layer.batch_normalization(input, axis=axis, training=training)



    # TODO deconv layer uncomplete
    @layer
    def deconv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, relu=True, padding=DEFAULT_PADDING,
               trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]

        deconvlve = lambda i, k: tf.nn.conv2d_transpose(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weight = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_bias = tf.constant_initializer(0.0)

            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weight, trainable,
                                   regularizer=self.l2_regularizer(
                                       cfg.TRAIN.WEIGHT_DECAY
                                   ))

            if biased:
                biases = self.make_var('biases', [c_o], init_bias, trainable)
                deconv = deconvlve(input, kernel)

                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(deconv, biases, name=scope.name)
            else:
                conv = deconvlve(input, kernel)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(conv, biases, name=scope.name)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                                 dtype=tensor.dtype.base_dtype,
                                                 name='weight_decay')
                # return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)
