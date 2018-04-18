from run.train import train
from run.inference import inference
"""
主函数，训练固定时间后取出checkpoints进行inference,inference 结果存储在data/temp 
"""
if __name__ == "__main__":
    train()