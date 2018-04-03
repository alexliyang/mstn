# Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation

**项目结构**

data  数据存放

batch_producer 训练数据生产者

network 包含数据消费 和 网络定义

solver tran_solver 和 test_solver

lib 公共libaray

run train 和 test的入口 







1. 在CTPN的基础上进一步省去冗余的步骤，简化代码，使用tensorflow queue机制来多线程处理数据，每次训练从队列中取出一批数据。

