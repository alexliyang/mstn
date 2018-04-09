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