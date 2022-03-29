from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np


"""
Define task metrics, loss functions and model trainer here.
"""


class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()


def create_task_flags(task, dataset, with_noise=False):
    """
    Record task and its prediction dimension.
    Noise prediction is only applied in auxiliary learning.
    """
    nyu_tasks = {'seg': 13, 'depth': 1, 'normal': 3}
    cityscapes_tasks = {'seg': 19, 'part_seg': 10, 'disp': 1}

    tasks = {}
    if task != 'all':
        if dataset == 'nyuv2':
            tasks[task] = nyu_tasks[task]
        elif dataset == 'cityscapes':
            tasks[task] = cityscapes_tasks[task]
    else:
        if dataset == 'nyuv2':
            tasks = nyu_tasks
        elif dataset == 'cityscapes':
            tasks = cityscapes_tasks

    if with_noise:
        tasks['noise'] = 1
    return tasks


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str


def get_weight_str_ranked(weight, tasks, rank_num):
    """
    Record top-k ranked task weighting.
    """
    rank_idx = np.argsort(weight)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), weight[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), weight[rank_idx[i]])

    return 'Task Weighting | {}| {}'.format(top_str, bot_str)


def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    if task_id in ['seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    if task_id in ['normal', 'depth', 'disp', 'noise']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss


class TaskMetric:
    def __init__(self, train_tasks, pri_tasks, batch_size, epochs, dataset, include_mtl=False):
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.dataset = dataset
        self.include_mtl = include_mtl
        self.metric = {key: np.zeros([epochs, 2]) for key in train_tasks.keys()}  # record loss & task-specific metric
        self.data_counter = 0
        self.epoch_counter = 0
        self.conf_mtx = {}

        if include_mtl:  # include multi-task performance (relative averaged task improvement)
            self.metric['all'] = np.zeros(epochs)
        for task in self.train_tasks:
            if task in ['seg', 'part_seg']:
                self.conf_mtx[task] = ConfMatrix(self.train_tasks[task])

    def reset(self):
        """
        Reset data counter and confusion matrices.
        """
        self.epoch_counter += 1
        self.data_counter = 0

        if len(self.conf_mtx) > 0:
            for i in self.conf_mtx:
                self.conf_mtx[i].reset()

    def update_metric(self, task_pred, task_gt, task_loss):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_gt: {'TASK_ID1': TASK_GT1, 'TASK_ID2': TASK_GT2, ...}
            :param task_loss: [TASK_LOSS1, TASK_LOSS2, ...]
        """
        curr_bs = task_pred[0].shape[0]
        r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter
        self.data_counter += 1

        with torch.no_grad():
            for loss, pred, (task_id, gt) in zip(task_loss, task_pred, task_gt.items()):
                self.metric[task_id][e, 0] = r * self.metric[task_id][e, 0] + (1 - r) * loss.item()

                if task_id in ['seg', 'part_seg']:
                    # update confusion matrix (metric will be computed directly in the Confusion Matrix)
                    self.conf_mtx[task_id].update(pred.argmax(1).flatten(), gt.flatten())

                if 'class' in task_id:
                    # Accuracy for image classification tasks
                    pred_label = pred.data.max(1)[1]
                    acc = pred_label.eq(gt).sum().item() / pred_label.shape[0]
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * acc

                if task_id in ['depth', 'disp', 'noise']:
                    # Abs. Err.
                    invalid_idx = -1 if task_id == 'disp' else 0
                    valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
                    abs_err = torch.mean(torch.abs(pred - gt).masked_select(valid_mask)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * abs_err

                if task_id in ['normal']:
                    # Mean Degree Err.
                    valid_mask = (torch.sum(gt, dim=1) != 0).to(pred.device)
                    degree_error = torch.acos(torch.clamp(torch.sum(pred * gt, dim=1).masked_select(valid_mask), -1, 1))
                    mean_error = torch.mean(torch.rad2deg(degree_error)).item()
                    self.metric[task_id][e, 1] = r * self.metric[task_id][e, 1] + (1 - r) * mean_error

    def compute_metric(self, only_pri=False):
        metric_str = ''
        e = self.epoch_counter
        tasks = self.pri_tasks if only_pri else self.train_tasks  # only print primary tasks performance in evaluation

        for task_id in tasks:
            if task_id in ['seg', 'part_seg']:  # mIoU for segmentation
                self.metric[task_id][e, 1] = self.conf_mtx[task_id].get_metrics()

            metric_str += ' {} {:.4f} {:.4f}'\
                .format(task_id.capitalize(), self.metric[task_id][e, 0], self.metric[task_id][e, 1])

        if self.include_mtl:
            # Pre-computed single task learning performance using trainer_dense_single.py
            if self.dataset == 'nyuv2':
                stl = {'seg': 0.4337, 'depth': 0.5224, 'normal': 22.40}
            elif self.dataset == 'cityscapes':
                stl = {'seg': 0.5620, 'part_seg': 0.5274, 'disp': 0.84}
            elif self.dataset == 'cifar100':
                stl = {'class_0': 0.6865, 'class_1': 0.8100, 'class_2': 0.8234, 'class_3': 0.8371, 'class_4': 0.8910,
                       'class_5': 0.8872, 'class_6': 0.8475, 'class_7': 0.8588, 'class_8': 0.8707, 'class_9': 0.9015,
                       'class_10': 0.8976, 'class_11': 0.8488, 'class_12': 0.9033, 'class_13': 0.8441, 'class_14': 0.5537,
                       'class_15': 0.7584, 'class_16': 0.7279, 'class_17': 0.7537, 'class_18': 0.9148, 'class_19': 0.9469}

            delta_mtl = 0
            for task_id in self.train_tasks:
                if task_id in ['seg', 'part_seg'] or 'class' in task_id:  # higher better
                    delta_mtl += (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]
                elif task_id in ['depth', 'normal', 'disp']:
                    delta_mtl -= (self.metric[task_id][e, 1] - stl[task_id]) / stl[task_id]

            self.metric['all'][e] = delta_mtl / len(stl)
            metric_str += ' | All {:.4f}'.format(self.metric['all'][e])
        return metric_str

    def get_best_performance(self, task):
        e = self.epoch_counter
        if task in ['seg', 'part_seg'] or 'class' in task:  # higher better
            return max(self.metric[task][:e, 1])
        if task in ['depth', 'normal', 'disp']:  # lower better
            return min(self.metric[task][:e, 1])
        if task in ['all']:  # higher better
            return max(self.metric[task][:e])
