"""Evaluate model and calculate results.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

from typing import List
import os
import codecs
import torch

from tqdm import tqdm
from sklearn import metrics
import fire

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
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


def evaluate(tokenizer, model, data_loader, device, multi_class: bool = False):
    """Evaluate model on data loader in device.

    Args:
        tokenizer: to decode sentence
        model: model to be evaluate
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu

    Returns:
        answer list, sent_list
    """
    from torch import nn
    model.eval()
    # outputs = torch.tensor([], dtype=torch.float).to(device)
    answer_list, sent_list = [], []
    for batch in tqdm(data_loader, desc='Evaluation', ascii=True, ncols=80, leave=True, total=len(data_loader)):
    # for _, batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits, _ = model(*batch)
        # outputs = torch.cat([outputs, torch.argmax(logits, dim=1)])

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

