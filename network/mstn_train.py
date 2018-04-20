from .base import BaseNetwork as base
import tensorflow as tf


class mstn_train_net(base):
    def __init__(self, cfg, trainable=True):
        self.inputs = []
        self._cfg = cfg

        self.img = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img')
        self.corner_data = tf.placeholder(tf.float32, shape=[None, None, None, 4], name='corner_data')
        self.img_info = tf.placeholder(tf.float32, shape=[None, None], name='img_info')
        self.resize_info = tf.placeholder(tf.float32, shape=[None, None], name='resize_info')
        self.segmentation_mask = tf.placeholder(tf.float32, shape=[None, None, self._cfg.COMMON.RESIZE_HEIGHT,
                                                                   self._cfg.COMMON.RESIZE_WIDTH],
                                                name='segmentation_mask')

        self.layers = dict({
            'img': self.img,
            'corner_data': self.corner_data,
            'img_info': self.img_info,
            'resize_info': self.resize_info,
            'segmentation_mask': self.segmentation_mask
        })
        self.trainable = trainable
        self.setup()

    def setup(self):
        # 详细查询vgg16的参数
        (self.feed('img')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool5'))

        ############# 以上为VGG16前五层 ####################################

        '''
         after conv5 follow by conv6 conv7 conv8 conv9 conv10 conv11
        '''
        # TODO 确定这些卷积核的大小 通道数 和池化层
        (self.conv(3, 3, 1024, 1, 1, name='conv6_1')
         .conv(3, 3, 1024, 1, 1, name='conv6_2')
         # TODO 根据论文 conv6 和conv7之间没有池化层
         .conv(3, 3, 1024, 1, 1, name='conv7_1')
         .conv(3, 3, 1024, 1, 1, name='conv7_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool7')
         .conv(3, 3, 1024, 1, 1, name='conv8_1')
         .conv(3, 3, 1024, 1, 1, name='conv8_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool8')
         .conv(3, 3, 1024, 1, 1, name='conv9_1')
         .conv(3, 3, 1024, 1, 1, name='conv9_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool9')
         .conv(3, 3, 1024, 1, 1, name='conv10_1')
         .conv(3, 3, 1024, 1, 1, name='conv10_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool10')
         .conv(3, 3, 1024, 1, 1, name='conv11_1')
         .conv(3, 3, 1024, 1, 1, name='conv11_2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='f11'))

        # TODO 每个f层需要有自己的输出 用于打分和回归

        # TODO 前一个参数是deconv中的红色层，后一个是蓝色层
        deconv_module_list = ['f10', 'f9', 'f8', 'f7', 'f4', 'f3']

        # TODO 每个feature map上的scale
        f_scales = {
            'f11': [184, 208, 232, 256],
            'f10': [124, 136, 148, 160],
            'f9': [88, 96, 104, 112],
            'f8': [56, 64, 72, 80],
            'f7': [36, 40, 44, 48],
            'f4': [20, 24, 28, 32],
            'f3': [4, 8, 6, 10, 12, 16],
        }

        for deconv_m in deconv_module_list:
            self.deconv_module(name=deconv_m)

            self.predict_module(name=deconv_m + '_pred')
            # TODO 将预测层的N H W 1024 输出送入全连接层准备输出3

            # 第一个参数是特征图的channel数目
            # 每个f层都需要输出预测目标

            # TODO 特征图每个像素 输出k * q * 2个预测得分 deconv_fc 输出 (N,H,W,k * q * 2)
            self.feed(deconv_m + '_pred').deconv_fc(512, 2, name=deconv_m + '_corner_pred_score')
            # TODO 特征图每个像素 输出k * q * 4个预测目标 deconv_fc 输出 (N,H,W,k * q * 4)
            self.feed(deconv_m + '_pred').deconv_fc(512, 4, name=deconv_m + '_corner_pred_offset')

            # TODO 需要计算出feat_stride 即在每个f层上一个像素点对应多少的步长
            self.feed(deconv_m + '_corner_pred_score', 'corner_box', 'img_info', 'gt_default_box') \
                .corner_detect_layer(scales=f_scales[deconv_m], feat_stride=None, name=deconv_m + '_loss_data', )

            # spatial_reshape_layer 将predict score reshape成 （1， height, -1, 2）
            # spatial_softmax 做空间softmax
            self.feed(deconv_m + '_corner_pred_score') \
                .spatial_reshape_layer(2, name=deconv_m + '_corner_reshape_pred_score') \
                .spatial_softmax(name=deconv_m + 'corner_cls_prob')

        # TODO 取出 f3 f4 f7 f8 f9 做segment sensitive map
        # 'f9', 'f8', 'f7', 'f4' 需要缩放到f3的大小
        # f3 shape (1, h, w, 1024)
        new_size = self.get_output('f3').get_shape()[1:3]

        layers_for_segment = ['f9', 'f8', 'f7', 'f4', 'f3']
        for f_layer in layers_for_segment[:-1]:
            self.feed(f_layer).bilinear_upsample(name=f_layer + '_bilinear')
        # TODO 以下卷积层的参数待定
        f_bilinear = [x + '_bilinear' for x in layers_for_segment]
        (self.feed(*f_bilinear)
         .layer_n_eltw_sum(name='segment_feature_map')
         .conv(1, 1, 1024, 1, 1, name='bilinear_conv1')
         .batch_normalize()
         .relu()
         .deconv(2, 2, 4, 1, 1, name='bilinear_deconv1')
         .conv(1, 1, 1024, 1, 1, name='bilinear_conv2')
         .batch_normalize()
         .relu()
         # 输出了 (1, h, w, 4)的特征图, 在计算这个分支的时候取出, 相对于f3，此时feature map已经扩大了4倍
         .deconv(2, 2, 4, 1, 1, name='bilinear_deconv2')
         # 做一次空间softmax 将每个像素输出的值约束在 0-1之间
         .spatial_softmax(name='segmentation_pred'))

    def build_loss(self):
        """
           two parts of loss
           Corner Point Detection
           Position Sensitive Segmentation
        :return:
        """
        deconv_module_list = ['f10', 'f9', 'f8', 'f7', 'f4', 'f3']
        """ corner point loss
            需要取出 'f10', 'f9', 'f8', 'f7', 'f4', 'f3' 每个特征图上的预测结果 计算损失函数
            
        
        """
        all_cross_entropy = 0
        all_regression_loss = 0
        # TODO 这个地方采用把所有角点的损失都统一计算, 需要搞清楚可不可以这样做
        for deconv_m in deconv_module_list:
            # shape(h * w * num_scales * 4, 2)
            corner_cls_score = tf.reshape(self.get_output(deconv_m + '_corner_reshape_pred_score'), [-1, 2])

            # self.get_output(deconv_m + '_loas_data')[0]是形如(1, FM的高，FM的宽，10)的labels
            # 真值标签shape (h * w * num_scales * 4)
            corner_label = tf.reshape(self.get_output(deconv_m + '_loss_data')[0], [-1])

            # 取出标签为1 的label所在的索引，多行一列矩阵 shape=(?,1)
            fg_keep = tf.where(tf.equal(corner_label, 1))

            # 取出标签为1 或者0的label所在的索引，多行一列矩阵
            # 对于是0或者是1的标签是我们感兴趣
            roi_keep = tf.where(tf.not_equal(corner_label, -1))

            # 取出保留的标签所在行的分数
            roi_cls_score = tf.gather(corner_cls_score, roi_keep)  # shape (N, 2)

            # 取出保留的标签所在的标签
            corner_label = tf.gather(corner_label, roi_keep)

            # 交叉熵损失 累加
            rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=corner_label,
                                                                                 logits=roi_cls_score)
            all_cross_entropy += tf.reduce_mean(rpn_cross_entropy_n)

            # shape (N,H,W,num_scales * 4 * 4)
            corner_pred_offset = self.get_output(deconv_m + '_corner_pred_offset')

            # 取出盒子真值用于回归
            # corner_pred_target shape (1, height, width, num_scales, 4, 4) 最后一个维度中，表示这个box 需要回归的四维的值
            corner_pred_target = self.get_output(deconv_m + '_loss_data')[1]

            """
               公式
            
            """

            # 取出标签为1的盒子回归
            # 取出 预测的值
            corner_pred_offset = tf.gather(tf.reshape(corner_pred_offset, [-1, 4]), fg_keep)  # shape (N, 2)

            # 取出 目标值
            corner_pred_target = tf.gather(tf.reshape(corner_pred_target, [-1, 4]), fg_keep)

            corner_loss_box_n = tf.reduce_sum(self.smooth_l1_dist((corner_pred_offset - corner_pred_target)),
                                              reduction_indices=[1])

            corner_regression_loss = tf.reduce_sum(corner_loss_box_n) / (tf.cast(tf.shape(fg_keep)[0], tf.float32) + 1)

            all_regression_loss += corner_regression_loss

        # TODO branch segment dice loss ; to be continued...

        # 取出准备好的segment mask
        segmentation_mask = self.get_output('segmentation_mask')
        # 取出预测的segment
        segmentation_pred = self.get_output('segmentation_pred')

        # 做dice loss
        dice = self.dice_coe(segmentation_pred,segmentation_mask)

        dice_loss = 1 - dice

        model_loss = all_cross_entropy + all_regression_loss + dice_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss
        return total_loss, model_loss, all_cross_entropy, all_regression_loss, dice_loss

    def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
        of two batch of data, usually be used for binary image segmentation
        i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

        Parameters
        -----------
        output : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        target : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        loss_type : string
            ``jaccard`` or ``sorensen``, default is ``jaccard``.
        axis : list of integer
            All dimensions are reduced, default ``[1,2,3]``.
        smooth : float
            This small value will be added to the numerator and denominator.
            If both output and target are empty, it makes sure dice is 1.
            If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
            then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
            so in this case, higher smooth can have a higher dice.

        Examples
        ---------
        >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
        >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

        References
        -----------
        - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
        """
        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        ## old axis=[0,1,2,3]
        # dice = 2 * (inse) / (l + r)
        # epsilon = 1e-5
        # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
        ## new haodong
        dice = (2. * inse + smooth) / (l + r + smooth)
        ##
        dice = tf.reduce_mean(dice)
        return dice

    def dice_hard_coe(output, target, threshold=0.5, axis=[1, 2, 3], smooth=1e-5):
        """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
        of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
        The coefficient between 0 to 1, 1 if totally match.

        Parameters
        -----------
        output : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        target : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        threshold : float
            The threshold value to be true.
        axis : list of integer
            All dimensions are reduced, default ``[1,2,3]``.
        smooth : float
            This small value will be added to the numerator and denominator, see ``dice_coe``.

        References
        -----------
        - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
        """
        output = tf.cast(output > threshold, dtype=tf.float32)
        target = tf.cast(target > threshold, dtype=tf.float32)
        inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
        ## old axis=[0,1,2,3]
        # hard_dice = 2 * (inse) / (l + r)
        # epsilon = 1e-5
        # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
        ## new haodong
        hard_dice = (2. * inse + smooth) / (l + r + smooth)
        ##
        hard_dice = tf.reduce_mean(hard_dice)
        return hard_dice