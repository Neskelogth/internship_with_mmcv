import torch
from functools import partial
# from mmcv.parallel import collate
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    print(cfg)
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     shuffle=True,
                     drop_last=False,
                     pin_memory=False,
                     persistent_workers=False,
                     **kwargs):

    batch_size = videos_per_gpu
    num_workers = workers_per_gpu

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # collate_fn=partial(collate, samples_per_gpu=videos_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs)

    return data_loader
