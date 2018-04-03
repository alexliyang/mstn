import yaml
from easydict import EasyDict as edict
import os
import pprint


def get_config(project_path, config_path):
    config_path = os.path.join(project_path, config_path)
    print('Using configure form:', config_path)
    with open(config_path, 'r', encoding='UTF-8') as f:
        yaml_cfg = edict(yaml.load(f))
    pprint.pprint(yaml_cfg)
    return yaml_cfg


def get_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


if __name__ == '__main__':
    get_config('configure.yml')
