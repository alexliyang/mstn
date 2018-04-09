import numpy as np
import tensorflow as tf
from lib import get_config

cfg = get_config()

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

    # TODO deconv module in DSSD
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
         .deconv(2, 2, 512, 1, 1)
         .conv(3, 3, 512, 1, 1)
         .batch_normalize(name=name + '_deconv_output'))

        (self.feed(feature_layer)
         .conv(3, 3, 512, 1, 1)
         .batch_normalize().relu()
         .conv(3, 3, 512, 1, 1)
         .batch_normalize(name=name + '_feature'))

        self.feed(name + '_deconv_output',name + '_feature')\
            .eltw_prod().relu(name=name)

        return self.get_output(name)


    def corner_detect(self):
        pass


    @layer
    def eltw_prod(self):
        pass

    # TODO batch norm
    @layer
    def batch_normalize(self):
        """
            x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


        :return:
        """
        pass

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
