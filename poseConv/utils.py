import hashlib
import logging
import os
import requests
import warnings
import socket

import numpy as np

from mmcv.utils import get_logger
from mmcv.runner import get_dist_info

from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash
from mmcv.runner import DistEvalHook as BasicDistEvalHook

__version__ = '0.1.0'


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "pyskl".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


def cache_checkpoint(filename, cache_dir='.cache'):
    if filename.startswith('http://') or filename.startswith('https://'):
        url = filename.split('//')[1]
        basename = filename.split('/')[-1]
        filehash = hashlib.md5(url.encode('utf8')).hexdigest()[-8:]
        os.makedirs(cache_dir, exist_ok=True)
        local_pth = os.path.join(cache_dir, basename.replace('.pth', f'_{filehash}.pth'))
        if not os.path.exists(local_pth):
            download_file(filename, local_pth)
        filename = local_pth
    return filename


def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    response = requests.get(url)
    open(filename, 'wb').write(response.content)


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def warning_r0(warn_str):
    rank, _ = get_dist_info()
    if rank == 0:
        warnings.warn(warn_str)


def comb(scores, coeffs):
    ret = [x * coeffs[0] for x in scores[0]]
    for i in range(1, len(scores)):
        ret = [x + y for x, y in zip(ret, [x * coeffs[i] for x in scores[i]])]
    return ret


def auto_mix2(scores):
    assert len(scores) == 2
    return {'1:1': comb(scores, [1, 1]), '2:1': comb(scores, [2, 1]), '1:2': comb(scores, [1, 2])}


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)
    print(cf_mat)
    exit(42)
    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]








def mc_on(port=22077, launcher='pytorch', size=60000, min_size=6):
    # size is mb, allocate 24GB memory by default.
    mc_exe = 'memcached' if launcher == 'pytorch' else '/mnt/lustre/share/memcached/bin/memcached'
    os.system(f'{mc_exe} -p {port} -m {size}m -I {min_size}m -d')

def mc_off():
    os.system('killall memcached')


def test_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    assert isinstance(ip, str)
    if isinstance(port, str):
        port = int(port)
    assert 1 <= port <= 65535
    result = sock.connect_ex((ip, port))
    return result == 0


def collect_env():
    env_info = collect_basic_env()
    env_info['pyskl'] = (
        __version__ + '+' + get_git_hash(digits=7))
    return env_info



class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', seg_interval=None, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        self.seg_interval = seg_interval
        if seg_interval is not None:
            assert isinstance(seg_interval, list)
            for i, tup in enumerate(seg_interval):
                assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
                if i < len(seg_interval) - 1:
                    assert tup[1] == seg_interval[i + 1][0]
            assert self.by_epoch
        assert self.start is None

    def _find_n(self, runner):
        current = runner.epoch
        for seg in self.seg_interval:
            if current >= seg[0] and current < seg[1]:
                return seg[2]
        return None

    def _should_evaluate(self, runner):
        if self.seg_interval is None:
            return super()._should_evaluate(runner)
        n = self._find_n(runner)
        assert n is not None
        return self.every_n_epochs(runner, n)
