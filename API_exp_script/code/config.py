"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('API')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './common','./data', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = [0]
    mode = 'train'
    model = {
        'align': True,
        'encoder': 'ResNet101',
        'decoder': '',
    }
    dataset = 'LIDC'  # 'VOC' or 'COCO' or 'FSS' or LIDC
    dataset_path = '../../data/'

    load_snapshot = None
    n_iters = 30
    label_sets = 0
    batch_size = 16
    lr_milestones = [50000]
    kl_loss_scaler = 0.01
    ignore_label = 255

    n_ways = 1
    n_shots =1
    n_queries = 1

    lr = 5e-5
    weight_decay = 1e-3


    exp_str = '_'.join(
        [dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{n_ways}way_{n_shots}shot_{mode}'])

    path = {
        'log_dir': './runs',
        'init_path': './pretrained_model/vgg16-397923af.pth',
        'VOC':{'data_dir': '../../data/Pascal/VOCdevkit/VOC2012/',
               'data_split': 'trainaug',},
        'COCO':{'data_dir': '../../data/COCO/',
                'data_split': 'train',},
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    if config['mode'] == 'test':
        if config['notrain']:
            exp_name += '_notrain'
        if config['scribble']:
            exp_name += '_scribble'
        if config['bbox']:
            exp_name += '_bbox'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
