# Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation

**项目结构**

data  数据存放

batch_producer 训练数据生产者

network 包含数据消费 和 网络定义

solver tran_solver 和 test_solver

lib 公共libaray

run train 和 test的入口 



取消数据预处理阶段，直接读取数据并做处理，使用tensorflow data API构建，看多线程读取并预处理，每次取出一批数据进行处理。

输入的数据要求把不规则的gt矩形框求出最小外接矩形

### 数据下载
此项操作会下载VGG16的预训练模型，和10000张图片的训练集并解压都data相应的目录下。10000张图片将被分成9000张的训练集和1000张的测试集合
```
data
 |____checkpoints checkpoints
 |____ICPR_text_train 完整的训练集合
       |__image_10000
       |__text_10000
 |____pretrain
 |____pretrain
       |__VGG_imagenet.npy 预训练模型参数
 |____test   测试数据集
       |__img
       |__txt
 |____tmp
       |__icpr_text_train_10000.zip 下载的10000张图片的压缩包
 |____train  训练集合
       |__img
       |__txt
``` 

进入项目根目录
```
python run/download.py
```

### 训练
进入项目根目录
```
python run/train.py
```
