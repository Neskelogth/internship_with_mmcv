import argparse
import mmcv
import os
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from poseConv.utils import __version__
from poseConv.api import init_random_seed, train_model
from poseConv.dataset import build_dataset
from poseConv.models import build_model
from poseConv.utils import collect_env, get_root_logger, mc_off, mc_on, test_port


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help='whether to test the best checkpoint (if applicable) after training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    os.environ['RANK'] = '1'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '0.0.0.0'
    os.environ['MASTER_PORT'] = '30080'

    return args


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # config file > default (base filename)
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    # if not hasattr(cfg, 'dist_params'):
    #     cfg.dist_params = dict(backend='nccl')

    # torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())
    # torch.distributed.init_process_group(**cfg.dist_params)
    # print(cfg.dist_params)
    # init_dist(args.launcher, **cfg.dist_params)
    # rank, world_size = get_dist_info()
    # cfg.gpu_ids = range(world_size)

    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = os.path.join(cfg.work_dir, 'latest.pth')
        if os.path.exists(resume_pth):
            cfg.resume_from = resume_pth

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = os.path.basename(args.config)
    meta['work_dir'] = os.path.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(cfg.model)
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    datasets = [build_dataset(cfg.data.train)]

    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            pyskl_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    train_model(model, datasets, cfg, validate=args.validate, test=test_option, timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    main()
