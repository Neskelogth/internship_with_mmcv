import argparse
import os

import torch

import mmcv
from mmcv import Config
from mmcv import digit_version as dv
from mmcv.engine import single_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.runner import load_checkpoint

from poseConv.dataset.builder import build_dataset, build_dataloader
from poseConv.models.builder import build_model
from poseConv.utils import cache_checkpoint

# from setuptools import find_packages


def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['top_k_accuracy', 'mean_class_accuracy'],
        help='evaluation metrics, which depends on the dataset, e.g.,'
             ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')

    args = parser.parse_args()
    assert args.checkpoint is not None

    return args


def inference_pytorch(args, cfg, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # build the model and load checkpoint
    model = build_model(cfg.model)
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    outputs = single_gpu_test(model, data_loader)

    return outputs


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    out = os.path.join(cfg.work_dir, 'result.pkl') if args.out is None else args.out

    # Load eval_config from cfg
    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if args.eval:
        eval_cfg['metrics'] = args.eval

    mmcv.mkdir_or_exist(os.path.dirname(out))
    _, suffix = os.path.splitext(out)
    assert suffix[1:] in file_handlers, 'The format of the out put file should be json, pickle or yaml'

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    outputs = inference_pytorch(args, cfg, data_loader)
    dataset.dump_results(outputs, out=out)
    eval_res = dataset.evaluate(outputs, **eval_cfg)
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
