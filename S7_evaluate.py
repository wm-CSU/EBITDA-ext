"""Evaluate model and calculate results.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

from typing import List
import numpy as np
from collections import Counter
import os
import codecs
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, multilabel_confusion_matrix
import fire

LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']


class Metrics:
    def __init__(self):
        """Initialize my metrics.
        """
        super(Metrics, self).__init__()

    def calculate_accuracy_f1(self,
                              golds: List[str], predicts: List[str]) -> tuple:
        """Calculate accuracy and f1 score.

        Args:
            golds: answers
            predicts: predictions given by model

        Returns:
            accuracy, f1 score
        """
        return accuracy_score(golds, predicts), \
               f1_score(
                   golds, predicts,
                   labels=LABELS, average='macro')

    def subclass_confusion_matrix(self, targetSrc, predSrc):
        # targetSrc = [[0, 1, 1, 1], [0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
        # predSrc = [[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0]]
        target = np.array(targetSrc)
        pred = np.array(predSrc)

        mcm1 = multilabel_confusion_matrix(target, pred)  # target在前，pred在后
        # 返回（19, 2, 2）的数组
        return mcm1

    def compute_metrics(self, labels, preds):
        precision, recall, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        mcm = self.subclass_confusion_matrix(targetSrc=labels, predSrc=preds)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1': (f1_micro + f1_macro) / 2,
            'subclass_confusion_matrix': mcm,
        }

    def perf_measure(self, confusion_matrix):
        """
        compute subclass's relative metrics from confusion matrix.
        :param confusion_matrix:
        :return:
        """
        subclass_metrics = {}
        for index, one in enumerate(confusion_matrix):
            tn, fp, fn, tp = one[0][0], one[0][1], one[1][0], one[1][1]
            accuracy = (tp + tn) / (tn + fp + fn + tp)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            subclass_metrics[index + 1] = {
                'accuracy': round(accuracy, 6),
                'precision': round(precision, 6),
                'recall': round(recall, 6),
                'f1': round(f1, 6),
            }

        return subclass_metrics

    def statistic_misjudgement(self, labels, preds):
        """
        get label misjudgement matrix.
        :param labels, preds:  type:Tensor
        :return:
        """
        labels_array = np.array(labels)
        pred_array = np.array(preds)
        label_number = labels_array.shape[1]
        confusion_matrix = np.zeros((label_number, label_number), dtype=float)
        miss_sample = np.zeros(label_number, dtype=int)

        for subclass in range(label_number):
            sub_index = np.where(labels_array[:, subclass] > 0)
            if len(sub_index[0]):
                for index in sub_index[0]:
                    if pred_array[index, subclass] == 1:  # pred right
                        confusion_matrix[subclass, subclass] += 1
                    else:  # pred error
                        misclass = np.where(pred_array[index, :] > 0)
                        if len(misclass[0]):
                            for one in misclass:
                                confusion_matrix[subclass, one] += 1 / misclass[0].shape[0]
                        else:
                            miss_sample[subclass] += 1
            else:
                continue

        return np.around(confusion_matrix, 3), miss_sample


def evaluate(tokenizer, model, data_loader, device, multi_class: bool = False):
    """Evaluate model on data loader in device.

    Args:
        tokenizer: to decode sentence
        model: model to be evaluate
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu
        multi_class:
    Returns:
        answer list, sent_list
    """
    from torch import nn
    model.eval()
    answer_list, sent_list = [], []
    for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
        # for _, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits, _ = model(*batch)

        sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in batch[0]])

        if multi_class:
            predictions = nn.Sigmoid()(logits)
            compute_pred = [[1 if one > 0.90 else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class
        else:
            answer_list.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            answer_list = [str(x) for x in answer_list]

    return answer_list, sent_list
