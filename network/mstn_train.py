from .base import BaseNetwork as base


class mstn_train_net(base):
    def __int__(self):
        pass

    def setup(self):
        # 详细查询vgg16的参数
        (self.feed('data')
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
            # TODO 将预测层的N H W 1024 输出送入全连接层准备输出

            # 第一个参数是特征图的channel数目
            # 每个f层都需要输出预测目标

            # TODO 特征图每个像素 输出k * q * 2个预测得分 deconv_fc 输出 (N,H,W,k * q * 2)
            self.feed(deconv_m + '_pred').deconv_fc(512, 1, name=deconv_m + '_corner_pred_score')
            # TODO 特征图每个像素 输出k * q * 4个预测目标 deconv_fc 输出 (N,H,W,k * q * 4)
            self.feed(deconv_m + '_pred').deconv_fc(512, 4, name=deconv_m + '_corner_pred_offset')

            # TODO 需要计算出feat_stride 即在每个f层上一个像素点对应多少的步长
            self.feed(deconv_m + '_corner_pred_score', 'corner_box','img_info','gt_default_box') \
                .corner_detect_layer(scales=f_scales[deconv_m],feat_stride=None, name=deconv_m + 'data',

                                     )

        # TODO 取出 f3 f4 f7 f8 f9 做segment sensitive map
