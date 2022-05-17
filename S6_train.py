"""Training file for sentence classification.

Author: Min Wang; wangmin0918@csu.edu.cn

Usage:
    python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py \
        --config_file 'config/bert_config.json'
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')  # “error”, “ignore”, “always”, “default”, “module” or “once”

from typing import Dict, List
import argparse
import json
import os
import time
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule)

from S4_dataset import Data
from S5_model import BertForClassification
from S7_evaluate import evaluate, calculate_accuracy_f1, get_labels_from_file

from utils import get_csv_logger, get_path, load_torch_model
from tools.pytorchtools import EarlyStopping


class Trainer:
    """Trainer for bert-base-uncased.

    """

    def __init__(self,
                 model, data_loader: Dict[str, DataLoader],
                 # train_labels, valid_labels,
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
        self.criterion = nn.CrossEntropyLoss()
        # self.writer = SummaryWriter()

    def _get_optimizer(self):
        """Get optimizer for different models.

        Returns:
            optimizer
        """
        # # no_decay = ['bias', 'gamma', 'beta']
        # no_decay = ['bias', 'LayerNorm.weight']
        # # optimizer_parameters = [
        # #     {'params': [p for n, p in self.model.named_parameters()
        # #                 if not any(nd in n for nd in no_decay) and p.requires_grad],
        # #      'weight_decay_rate': 0.01},
        # #     {'params': [p for n, p in self.model.named_parameters()
        # #                 if any(nd in n for nd in no_decay) and p.requires_grad],
        # #      'weight_decay_rate': 0.0}]
        # optimizer_parameters = [
        #     {'params': [p for p in self.model.parameters() if p.requires_grad],
        #      'weight_decay_rate': 0.01}
        # ]
        optimizer = AdamW(
            # [p for p in self.model.parameters() if p.requires_grad],
            # optimizer_parameters,
            filter(lambda p: p.requires_grad, self.model.parameters()),
            # self.model.parameters(),
            lr=self.config.lr,
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

    def _evaluate(self, data_loader) -> List[str]:
        """Evaluate model on data loader in device for train.

        Args:
            data_loader: torch.utils.data.DataLoader

        Returns:
            answer list
        """
        self.model.eval()
        answer_list, labels = [], []
        # for batch in tqdm(data_loader, desc='Evaluation', ascii=False, ncols=80, position=0, total=len(data_loader)):
        for _, batch in enumerate(data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits, _ = self.model(*batch[:-1])
            labels.extend(batch[-1].detach().cpu().numpy().tolist())
            answer_list.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

        return [str(x) for x in answer_list], [str(x) for x in labels]

    # def _evaluate_for_train_valid(self):
    #     """Evaluate model on train and valid set and get acc and f1 score.
    #
    #     Returns:
    #         train_acc, train_f1, valid_acc, valid_f1
    #     """
    #     train_predictions = self.evaluate(data_loader=self.data_loader['valid_train'])
    #     # valid_predictions = evaluate(
    #     #     model=self.model, data_loader=self.data_loader['valid_valid'],
    #     #     device=self.device)
    #     # train_answers = get_labels_from_file(self.config.train_file_path)
    #     # valid_answers = get_labels_from_file(self.config.valid_file_path)
    #     train_acc, train_f1 = calculate_accuracy_f1(
    #         [str(x) for x in self.train_labels], train_predictions)
    #     # valid_acc, valid_f1 = calculate_accuracy_f1(
    #     #     [str(x) for x in self.valid_labels], valid_predictions)
    #     valid_acc, valid_f1 = train_acc, train_f1
    #
    #     return train_acc, train_f1, valid_acc, valid_f1

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
        # 原_evaluate_for_train_valid()内容
        predictions, labels = self._evaluate(data_loader=self.data_loader['train'])
        train_acc, train_f1 = calculate_accuracy_f1(labels, predictions)
        # valid_predictions = evaluate(
        #     model=self.model, data_loader=self.data_loader['valid_valid'],
        #     device=self.device)
        # valid_answers = get_labels_from_file(self.config.valid_file_path)
        # valid_acc, valid_f1 = calculate_accuracy_f1(
        #     [str(x) for x in self.valid_labels], valid_predictions)
        valid_predictions, valid_labels = self._evaluate(data_loader=self.data_loader['valid_train'])
        valid_acc, valid_f1 = calculate_accuracy_f1(valid_labels, valid_predictions)
        results = train_acc, train_f1, valid_acc, valid_f1

        # Evaluate model for train and valid set
        # results = self._evaluate_for_train_valid()
        # train_acc, train_f1, valid_acc, valid_f1 = results
        # Update tqdm description for command line
        tqdm_obj.set_description(
            'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, '
            'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
                epoch, train_acc, train_f1, valid_acc, valid_f1))
        # Logging
        logger.info(','.join([str(epoch)] + [str(s) for s in results]))
        return train_acc, train_f1, valid_acc, valid_f1

    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)

    def train(self, ReTrain=False):
        """Train model on train set and evaluate on train and valid set.

        Returns:
            state dict of the best model with highest valid f1 score
        """
        epoch_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-epoch.csv'),
            title='epoch,train_acc,train_f1,valid_acc,valid_f1',
            log_format='%(asctime)s - %(name)s - %(message)s')
        step_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-step.csv'),
            title='step,loss',
            log_format='%(asctime)s - %(name)s - %(message)s'
        )

        epoch_logger.info("--------------------loading model and optimizer...-------------------\n")
        if ReTrain:  # 读入最新模型
            temporary = self.load_last_model(model=self.model,
                                             model_path=os.path.join(self.config.model_path,
                                                                     self.config.experiment_name,
                                                                     self.config.model_type + '-last_model.bin'),
                                             optimizer=self.optimizer,
                                             multi_gpu=False)
            self.model, self.optimizer, start_epoch, self.best_f1 = temporary
            self.scheduler.last_epoch = start_epoch
            self.steps_left = (self.config.num_epoch - start_epoch) * len(self.data_loader['train'])
            # self.config.num_training_steps = config.num_epoch * (len(data_loader['train']) // config.batch_size)
        else:
            self.steps_left = self.config.num_epoch * len(self.data_loader['train'])
            self.best_f1 = 0
            start_epoch = 0

        self.model.to(self.device)
        if self.config.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        epoch_logger.info("--------------------training model...-------------------\n")
        best_model_state_dict = None
        progress_bar = trange(self.config.num_epoch - start_epoch, desc='Epoch', ncols=160)
        self.earlystop = EarlyStopping(patience=5, verbose=True)
        self._epoch_evaluate_update_description_log(
            tqdm_obj=progress_bar, logger=epoch_logger, epoch=0)

        # start training.
        for epoch in range(start_epoch, self.config.num_epoch):
            self.model.train()
            train_loss_sum = 0
            try:
                with tqdm(self.data_loader['train'], desc='step: ', ascii=False, ncols=80, position=0) as tqdm_obj:
                    for step, batch in enumerate(tqdm_obj):
                        batch = tuple(t.to(self.device) for t in batch)
                        logits, _ = self.model(*batch[:-1])  # the last one is label
                        # pred = torch.argmax(logits, dim=1).long()
                        loss = self.criterion(logits, batch[-1])
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
                            # self.steps_left -= 1
                            # tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))
                            step_logger.info(str(self.steps_left) + ',' + str(loss.item()))
                        tqdm_obj.update(1)
            except KeyboardInterrupt:
                tqdm_obj.close()
                raise
            tqdm_obj.close()
            # progress_bar.update(1)
            results = self._epoch_evaluate_update_description_log(
                tqdm_obj=progress_bar, logger=epoch_logger, epoch=epoch + 1)
            # 分别保存分步模型，最新模型，最优模型
            # self.save_model(os.path.join(
            #     self.config.model_path, self.config.experiment_name,
            #     self.config.model_type + '-' + str(epoch + 1) + '.bin'))
            torch.save({'model_state_dict': self.model.state_dict(),
                        'epoch': epoch + 1,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_f1': results[-1],
                        }, os.path.join(self.config.model_path, self.config.experiment_name,
                                        self.config.model_type + '-last_model.bin'))
            if results[-1] > self.best_f1:
                self.save_model(os.path.join(
                    self.config.model_path, self.config.experiment_name,
                    self.config.model_type + '-best_model.bin'))
                best_model_state_dict = deepcopy(self.model.state_dict())
                self.best_f1 = results[-1]

            self.earlystop(train_loss_sum / len(self.data_loader['train']), self.model)
            if self.earlystop.early_stop:
                epoch_logger.info("Early stop \n")
                break

        return best_model_state_dict

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


def main(config_file='config/bert_config.json'):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)

    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type)
    datasets = data.load_train_and_valid_files(
        train_file='data/matchtxt.xlsx', train_sheet='Sheet1', train_txt='data/sent_label/',
        valid_file='data/matchtxt.xlsx', valid_sheet='Sheet1', valid_txt='data/sent_label/',
    )
    # train_set, train_labels, valid_set_train, valid_labels, valid_set_valid = datasets
    train_set, valid_set_train, valid_set_valid = datasets
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('cuda is available!')
        # rank = 0
        # torch.cuda.set_device(rank)
        # torch.distributed.init_process_group(backend="nccl", rank=0)
        # sampler_train = DistributedSampler(train_set)
    else:
        device = torch.device('cpu')
        # sampler_train = RandomSampler(train_set)

    data_loader = {
        # 'train': DataLoader(
        #     train_set, sampler=sampler_train, batch_size=config.batch_size),
        'train': DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
        'valid_valid': DataLoader(
            valid_set_valid, batch_size=config.batch_size, shuffle=False)
    }

    # 2. Build model
    model = BertForClassification(config)
    model.to(device)
    # if torch.cuda.is_available():
    #     model = torch.nn.parallel.DistributedDataParallel(
    #     model, find_unused_parameters=True)

    # 3. Train
    trainer = Trainer(model=model, data_loader=data_loader,
                      # train_labels=train_labels, valid_labels=valid_labels,
                      device=device, config=config)
    best_model_state_dict = trainer.train(ReTrain=True)
    # 4. Save model
    torch.save(best_model_state_dict,
               os.path.join(config.model_path, 'model.bin'))
    # 5. evaluate only
    # model, _, _, _ = trainer.load_last_model(trainer.model,
    #                                          os.path.join(trainer.config.model_path, trainer.config.experiment_name,
    #                                                       trainer.config.model_type + '-last_model.bin'),
    #                                          trainer.optimizer, False)
    model = load_torch_model(trainer.model,
                             os.path.join(trainer.config.model_path, trainer.config.experiment_name,
                                          trainer.config.model_type + '-best_model.bin'),
                             False)
    valid_predictions = evaluate(model, data_loader['valid_valid'], device)
    # print(train_labels)
    print(valid_predictions)
    # train_acc, train_f1 = calculate_accuracy_f1(
    #     [str(x) for x in train_labels], train_predictions)
    # print(train_acc, train_f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()

    main(args.config_file)
