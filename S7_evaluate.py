"""Evaluate model and calculate results.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

from typing import List
import os
import codecs
import torch

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import fire

LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']


def calculate_accuracy_f1(
        golds: List[str], predicts: List[str]) -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        golds: answers
        predicts: predictions given by model

    Returns:
        accuracy, f1 score
    """
    return metrics.accuracy_score(golds, predicts), \
           metrics.f1_score(
               golds, predicts,
               labels=LABELS, average='macro')


def subclass_confusion_matrix(targetSrc, predSrc):
    from sklearn.metrics import multilabel_confusion_matrix
    import numpy as np

    # targetSrc = [[0, 1, 1, 1], [0, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    # predSrc = [[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0]]
    target = np.array(targetSrc)
    pred = np.array(predSrc)

    mcm1 = multilabel_confusion_matrix(target, pred)  # target在前，pred在后
    # 返回（19, 2, 2）的数组
    return mcm1


def compute_metrics(labels, preds):
    precision, recall, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    mcm = subclass_confusion_matrix(targetSrc=labels, predSrc=preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1': (f1_micro + f1_macro) / 2,
        'subclass_confusion_matrix': mcm,
    }


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return TP, FP, TN, FN


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

