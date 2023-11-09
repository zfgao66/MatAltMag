"""
This file is used to load YAML file of the project and transform it to Dict
object of Python.

"""
from argparse import ArgumentParser
import yaml
import pprint
import os
import os.path as osp
import numpy as np
import torch


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--file', dest='filename', required=True)
    return parser


def load_config(yaml_filename):
    if os.path.exists(yaml_filename):
        with open(yaml_filename, 'r', encoding='utf-8') as stream:
            content = yaml.load(stream, Loader=yaml.FullLoader)
        return content
    else:
        print('config file don\'t exist!')
        exit(1)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_mkdir(config, dir_keys):
    for key in dir_keys:
        if key not in config:
            continue
        path = config[key].rsplit('/', 1)[0]
        if not osp.exists(path):
            os.makedirs(path)


def update_path(config, keys):
    for key in keys:
        if key in config:
            config[key] = config[key].format(config['experiment_name'])


def process_config(config):
    # set random seed
    if 'random_seed' in config:
        set_random_seed(config['random_seed'])

    # set data type
    if 'precision' in config and config['precision'] == 'float64':
        config['dtype'] = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        config['dtype'] = torch.float32
        torch.set_default_dtype(torch.float32)

    # update path in config
    path_keys = ['log_file', 'ckpt_path', 'pre_ckpt_path', 'figs_train', 'figs_pretrain', 'output_file', 'figs_analysis', 'restore_loss']
    update_path(config, path_keys)
    # mkdir
    dir_keys = ['log_file', 'pre_ckpt_path', 'figs_pretrain', 'output_file', 'restore_loss']
    config_mkdir(config, dir_keys)

    return config


if __name__ == '__main__':
    # This is a test to ensure load YAML file correctly

    from kogger import Logger

    args = get_parser().parse_args()
    config = load_config(yaml_filename=args.filename)
    config = process_config(config)
    logger = Logger('CONFIG')
    logger.info('Load config successfully!')
    logger.info(pprint.pformat(config))
