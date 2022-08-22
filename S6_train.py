"""Training file for sentence classification.

Author: Min Wang; wangmin0918@csu.edu.cn

Usage:
    python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
"""
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings

warnings.filterwarnings('ignore')  # “error”, “ignore”, “always”, “default”, “module” or “once”

from typing import Dict, List
import numpy as np
import os
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from visdom import Visdom
from tqdm import tqdm, trange
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule)

from utils import get_csv_logger
from tools.pytorchtools import EarlyStopping
from S7_evaluate import Metrics
from S5_loss import FocalLoss_MultiLabel


class Trainer:
    """Trainer for bert-base-uncased.
    """

    def __init__(self,
                 model, data_loader: Dict[str, DataLoader],
                 device, config):
        """Initialize trainer with model, data, device, and config.
        Initialize optimizer, scheduler, criterion.

        Args:
            model: model to be evaluated
            data_loader: dict of torch.utils.data.DataLoader
            device: torch.device('cuda') or torch.device('cpu')
            config:
                config.experiment_name: experiment name
                config.model_type: 'bert'
                config.lr: learning rate for optimizer
                config.num_epoch: epoch number
                config.num_warmup_steps: warm-up steps number
                config.gradient_accumulation_steps: gradient accumulation steps
                config.max_grad_norm: max gradient norm

        """
        self.model = model
        self.device = device
        self.config = config
        self.data_loader = data_loader
        self.config.num_training_steps = config.num_epoch * (len(data_loader['train']) // config.batch_size)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.criterion = FocalLoss_MultiLabel()
        # self.criterion = nn.BCELoss()
        self.viz = Visdom(env=self.config.experiment_name)
        self.mymetrics = Metrics()

    def _get_optimizer(self):
        """Get optimizer for different models.

        Returns:
            optimizer
        """
        no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters()
        #                 if not any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in self.model.named_parameters()
        #                 if any(nd in n for nd in no_decay)],
        #      'weight_decay_rate': 0.0}]
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay_rate': 0.01, 'lr': self.config.lr_ptm},
            {'params': [p for n, p in self.model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay) and 'bert' not in n],
             'weight_decay_rate': 0.01, 'lr': self.config.lr_others},
            {'params': [p for n, p in self.model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay_rate': 0.0, 'lr': self.config.lr_ptm},
            {'params': [p for n, p in self.model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay) and 'bert' not in n],
             'weight_decay_rate': 0.0, 'lr': self.config.lr_others},
        ]
        optimizer = AdamW(
            optimizer_parameters,
            # self.model.parameters(),
            betas=(0.9, 0.999),
            weight_decay=1e-8,
            correct_bias=False)
        return optimizer

    def _get_scheduler(self):
        """Get scheduler for different models.
        Returns:
            scheduler
        """
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config.warmup_ratio * self.config.num_training_steps),
            num_training_steps=self.config.num_training_steps)
        return scheduler

    def _evaluate(self, data_loader):
        """Evaluate model on data loader in device for train.

        Args:
            data_loader: torch.utils.data.DataLoader

        Returns:
            answer list
        """
        self.model.eval()
        answer_list, labels = [], []
        for _, batch in enumerate(data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits, _ = self.model(*batch[:-1])
            labels.extend(batch[-1].detach().cpu().numpy().tolist())
            predictions = nn.Sigmoid()(logits)
            compute_pred = [[1 if one > self.config.sigmoid_threshold else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, labels

    def evaluate_train_valid(self):
        # Evaluate model for train and valid set
        predictions, labels = self._evaluate(data_loader=self.data_loader['train'])
        train_result = self.mymetrics.compute_metrics(labels=labels, preds=predictions, )
        valid_predictions, valid_labels = self._evaluate(data_loader=self.data_loader['valid_train'])
        valid_result = self.mymetrics.compute_metrics(labels=valid_labels, preds=valid_predictions, )

        return train_result, valid_result

    def _epoch_evaluate_update_description_log(
            self, tqdm_obj, logger, epoch):
        """Evaluate model and update logs for epoch.

        Args:
            tqdm_obj: tqdm/trange object with description to be updated
            logger: logging.logger
            epoch: int

        Return:
            train_acc, train_f1, valid_acc, valid_f1
        """
        train_result, valid_result = self.evaluate_train_valid()

        # Update tqdm description for command line
        tqdm_obj.set_description(
            'Epoch: {:d}, train: {{acc: {:.6f}, precision: {:.6f}, recall: {:.6f}, f1: {:.6f}}} '
            'valid: {{acc: {:.6f}, precision: {:.6f}, recall: {:.6f}, f1: {:.6f}}}'.format(
                epoch, train_result['accuracy'], train_result['precision'], train_result['recall'], train_result['f1'],
                valid_result['accuracy'], valid_result['precision'], valid_result['recall'], valid_result['f1'])
        )
        # Logging
        logger.info(','.join(
            [str(epoch)] + ['\n    train: '] +
            [str(k) + ': ' + str(format(v, '.6f')) for k, v in train_result.items() if k != 'subclass_confusion_matrix']
            + ['\n    valid: '] + [str(k) + ': ' + str(format(v, '.6f')) for k, v in valid_result.items() if
                                   k != 'subclass_confusion_matrix'] + ['\n    '] +
            [''.join(np.array2string(valid_result['subclass_confusion_matrix']).splitlines())] + ['\n'])
        )
        return train_result, valid_result

    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)

    def _log_init(self):
        epoch_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-epoch.csv'),
            title='epoch, train_acc, train_precision, train_recall, train_f1_micro, train_f1_macro, train_f1, '
                  'valid_acc, valid_precision, valid_recall, valid_f1_micro, valid_f1_macro, valid_f1, '
                  'valid_subclass_confusion_matrix',
            log_format='%(asctime)s - %(message)s')
        step_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-step.csv'),
            title='step,loss',
            log_format='%(asctime)s - %(message)s')

        return epoch_logger, step_logger

    def train(self, ReTrain: bool = False):
        """Train model on train set and evaluate on train and valid set.

        Returns:
            state dict of the best model with highest valid f1 score
        """
        epoch_logger, step_logger = self._log_init()

        if ReTrain:  # 读入最新模型
            temporary = self.load_last_model(
                model=self.model,
                model_path=os.path.join(self.config.model_path, self.config.experiment_name,
                                        self.config.model_type + '-last_model.bin'),
                optimizer=self.optimizer, multi_gpu=False)
            self.model, self.optimizer, start_epoch, self.best_f1 = temporary

            self.save_model(os.path.join(
                self.config.model_path, self.config.experiment_name,
                str(time.asctime()) + '-saved_model.bin'))

            self.scheduler.last_epoch = start_epoch
            self.steps_left = (self.config.num_epoch - start_epoch) * len(self.data_loader['train'])
        else:
            epoch_logger.info("--------------------training model...-------------------\n")
            self.steps_left = self.config.num_epoch * len(self.data_loader['train'])
            self.best_f1 = 0
            start_epoch = 0
            self.viz.line([[float("inf"), 0., 0.]], [0], win='train',
                          opts=dict(title=self.config.experiment_name, legend=['loss', 'train_f1', 'valid_f1']))

        self.model.to(self.device)
        if self.config.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        best_model_state_dict = None
        progress_bar = trange(self.config.num_epoch - start_epoch, desc='Epoch', ncols=220)
        self.earlystop = EarlyStopping(patience=3, verbose=True)
        # self._epoch_evaluate_update_description_log(
        #     tqdm_obj=progress_bar, logger=epoch_logger, epoch=0)

        # start training.
        for epoch in range(start_epoch, self.config.num_epoch):
            self.model.train()
            try:
                with tqdm(self.data_loader['train'], desc='step: ', ascii=False, ncols=140, position=0) as tqdm_obj:
                    epoch_loss = self.run_epoch(tqdm_obj=tqdm_obj)

            except KeyboardInterrupt:
                tqdm_obj.close()
                raise
            tqdm_obj.close()
            progress_bar.update(1)

            step_logger.info(str(epoch) + ', train mode: ' + self.config.train_mode + ', loss: ' + str(epoch_loss))

            train_result, valid_result = self._epoch_evaluate_update_description_log(
                tqdm_obj=progress_bar, logger=epoch_logger, epoch=epoch + 1)
            self.viz.line([[epoch_loss, train_result['f1'], valid_result['f1']]], [epoch], win='train',
                          update='append')
            # 分别保存最新模型，最优模型
            torch.save({'model_state_dict': self.model.state_dict(),
                        'epoch': epoch + 1,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_f1': valid_result['f1'],
                        }, os.path.join(self.config.model_path, self.config.experiment_name,
                                        self.config.model_type + '-last_model.bin'))
            if valid_result['f1'] > self.best_f1:
                self.save_model(os.path.join(
                    self.config.model_path, self.config.experiment_name,
                    self.config.model_type + '-best_model.bin'))
                best_model_state_dict = deepcopy(self.model.state_dict())
                self.best_f1 = valid_result['f1']

            self.earlystop(epoch_loss, self.model)
            if self.earlystop.early_stop:
                epoch_logger.info("Early stop \n")
                break
        self.viz.save([self.config.experiment_name])

        return best_model_state_dict

    def run_epoch(self, tqdm_obj):
        train_loss_sum = 0
        for step, batch in enumerate(tqdm_obj):
            batch = tuple(t.to(self.device) for t in batch)
            logits, _ = self.model(*batch[:-1])  # the last one is label
            predictions = nn.Sigmoid()(logits)
            loss = self.criterion(predictions.float(), batch[-1].float())
            # loss = nn.BCEWithLogitsLoss()(predictions, true_labels.to(self.device).float())

            train_loss_sum += loss.item()
            if self.config.gradient_accumulation_steps > 1:  # 多次叠加
                loss = loss / self.config.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm)  # 梯度裁剪
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            tqdm_obj.update(1)

        return train_loss_sum / len(tqdm_obj)

    @staticmethod
    def load_last_model(model, model_path, optimizer,
                        multi_gpu: bool = False):
        """Load state dict to model.

        Args:
            model: model to be loaded
            model_path: state dict file path
            optimizer: optimizer structure
            multi_gpu: Use multiple GPUs or not

        Returns:
            loaded model, loaded optimizer, start_epoch, best_acc
        """
        pretrained_model_dict = torch.load(model_path)
        from collections import OrderedDict
        if multi_gpu:
            new_state_dict = OrderedDict()
            for k, value in pretrained_model_dict['model_state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = value
            model.load_state_dict(new_state_dict, strict=True)
        else:
            model.load_state_dict(pretrained_model_dict['model_state_dict'], strict=True)

        optimizer.load_state_dict(pretrained_model_dict['optimizer_state_dict'])
        start_epoch = pretrained_model_dict['epoch']
        best_acc = pretrained_model_dict['best_f1']

        return model, optimizer, start_epoch, best_acc
