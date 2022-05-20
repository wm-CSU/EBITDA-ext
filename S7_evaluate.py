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


def get_labels_from_file(filename):
    """Get labels on the last column from file.

    Args:
        filename: file name

    Returns:
        List[str]: label list
    """
    labels = []
    with codecs.open(filename, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            labels.append(line.strip().split(',')[-1])
    return labels


def eval_file(golds_file, predicts_file):
    """Evaluate submission file

    Args:
        golds_file: file path
        predicts_file:  file path

    Returns:
        accuracy, f1 score
    """
    golds = get_labels_from_file(golds_file)
    predicts = get_labels_from_file(predicts_file)
    return calculate_accuracy_f1(golds, predicts)


def evaluate(tokenizer, model, data_loader, device) -> List[str]:
    """Evaluate model on data loader in device.

    Args:
        tokenizer: to decode sentence
        model: model to be evaluate
        data_loader: torch.utils.data.DataLoader
        device: cuda or cpu

    Returns:
        answer list, sent_list
    """
    model.eval()
    # outputs = torch.tensor([], dtype=torch.float).to(device)
    answer_list, sent_list = [], []
    # for batch in tqdm(data_loader, desc='Evaluation', ascii=True, ncols=80, leave=True, total=len(data_loader)):
    for _, data in enumerate(data_loader):
        batch = tuple(t.to(device) for t in data)
        with torch.no_grad():
            logits, _ = model(*batch)
        # outputs = torch.cat([outputs, torch.argmax(logits, dim=1)])
        # sent_list.extend(tokenizer.decode_batch(data[0], skip_special_tokens=True))
        sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in batch[0]])
        answer_list.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

    return [str(x) for x in answer_list], sent_list


def evaluate_main(golden_file='data/task1.2-test-labels.csv', predict_file='result/bert-test1.csv'):
    acc, f1_score = eval_file(golden_file, predict_file)
    print("acc: {}, f1: {}".format(acc, f1_score))


if __name__ == '__main__':
    # fire.Fire(main)
    # 实验结果计算
    # for i in range(10):
    #     file = os.path.join('task1.2/result/', 'bert-test1-' + str(i) + '.csv')
    #     evaluate_main(predict_file=file)
    pass
