from .mstn_train import mstn_train_net
from .mstn_test import mstn_test_net


def get_network(name):
    if name == 'train':
        return mstn_train_net()
    elif name == 'test':
        return mstn_test_net()
