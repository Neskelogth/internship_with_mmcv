import os
import copy
import mmcv
import numpy as np
import torch
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from torch.utils.data import Dataset

from ..utils import auto_mix2, mean_average_precision, mean_class_accuracy, top_k_accuracy
from ..pipelines.compose import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix='',
                 test_mode=False,
                 multi_class=True,
                 num_classes=60,  # number of NTU_60 classes
                 start_index=1,
                 modality='RGB'):

        super().__init__()

        self.ann = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.cli = None

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    def __len__(self):
        return len(self.video_infos)

    @abstractmethod
    def load_annotations(self):
        """Load annotations"""

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def parse_by_class(self):
        pass

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if isinstance(results[0], list) or isinstance(results[0], tuple):
            num_results = len(results[0])
            eval_results = dict()
            for i in range(num_results):
                eval_results_cur = self.evaluate(
                    [x[i] for x in results], metrics, metric_options, logger, **deprecated_kwargs)
                eval_results.update({f'{k}_{i}': v for k, v in eval_results_cur.items()})
            return eval_results

        elif isinstance(results[0], dict):
            eval_results = dict()
            for key in results[0]:
                results_cur = [x[key] for x in results]
                eval_results_cur = self.evaluate(results_cur, metrics, metric_options, logger, **deprecated_kwargs)
                eval_results.update({f'{key}_{k}': v for k, v in eval_results_cur.items()})
            # Ad-hoc for RGBPoseConv3D
            if len(results[0]) == 2 and 'rgb' in results[0] and 'pose' in results[0]:
                rgb = [x['rgb'] for x in results]
                pose = [x['pose'] for x in results]
                preds = auto_mix2([rgb, pose])
                for k in preds:
                    eval_results_cur = self.evaluate(preds[k], metrics, metric_options, logger, **deprecated_kwargs)
                    eval_results.update({f'RGBPose_{k}_{key}': v for key, v in eval_results_cur.items()})

            return eval_results

        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)
        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision']

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                    'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk,)

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_average_precision':
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                mAP = mean_average_precision(results, gt_labels_arrays)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results

    def prepare_for_train(self, idx):
        results = copy.deepcopy(self.video_infos[idx])

        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def prepare_for_test(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])

        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        results['test_mode'] = self.test_mode

        return self.pipeline(results)

    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            path_value = os.path.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_for_test(idx)
        else:
            return self.prepare_for_train(idx)
