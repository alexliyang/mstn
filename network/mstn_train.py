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
        self.deconv_module(name='f10')
        self.deconv_module(name='f9')
        self.deconv_module(name='f8')
        self.deconv_module(name='f7')
        self.deconv_module(name='f4')
        self.deconv_module(name='f3')

        # TODO 取出 f3 f4 f7 f8 f9 f10 f11


        # TODO 取出 f3 f4 f7 f8 f9 做segment sensitive map
