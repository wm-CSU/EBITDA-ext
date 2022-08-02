"""Evaluate model and calculate results.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

from typing import List
import numpy as np
from collections import Counter
import time
import xlwt
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support,
                             multilabel_confusion_matrix, confusion_matrix)
import fire


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
        LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
        return accuracy_score(golds, predicts), \
               f1_score(
                   golds, predicts,
                   labels=LABELS, average='macro')

    def compute_metrics_b1(self, labels, preds):
        precision = precision_score(labels, preds, )
        recall = recall_score(labels, preds, )
        f1 = f1_score(labels, preds, )
        acc = accuracy_score(labels, preds)
        mcm = confusion_matrix(labels, preds)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': mcm,
        }

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

    def statistic_misjudgement(self, labels, preds, sent):
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

    def metrics_output(self, pred_to_file, target_list, pred_list, sent_list, b1_preds, b1_labels,
                       filename: str = 'result/result.txt', ):
        sentence_mcm = self.subclass_confusion_matrix(targetSrc=target_list, predSrc=pred_list)
        sentence_subclass_metrics = self.perf_measure(sentence_mcm)
        sentence_total_cm, sentence_miss = self.statistic_misjudgement(labels=target_list, preds=pred_list,
                                                                       sent=sent_list)
        result = self.compute_metrics(labels=target_list, preds=pred_list)
        b1_result = self.compute_metrics_b1(labels=b1_labels, preds=b1_preds, )
        print(b1_result)
        # samples_mcm = self.sample_cm()
        # sample_subclass_metrics = self.perf_measure(samples_mcm)

        with open(filename, 'a+') as f:
            f.write('\n\n\n' + time.asctime() + '   PredictionWithlabels    ->  ' + pred_to_file + '\n')
            f.write('branch 1:  ' + str([str(k) + ': ' + str(format(v, '.6f')) for k, v in b1_result.items() if
                    k != 'confusion_matrix']) + ''.join(np.array2string(b1_result['confusion_matrix']).splitlines()) + '\n')
            f.write(str([str(k) + ': ' + str(format(v, '.6f')) for k, v in result.items() if
                         k != 'subclass_confusion_matrix']) + '\n')

            f.write('sentence_mcm: \n')
            # f.write(''.join(np.array2string(sentence_mcm).splitlines()))
            for k, v in sentence_subclass_metrics.items():
                f.write(str(k) + ': ' + ''.join(np.array2string(sentence_mcm[k - 1]).splitlines())
                        + '\t' + str(v) + '\n')
                f.write('   class misjudge:' + ''.join(np.array2string(sentence_total_cm[k - 1]).splitlines()) + '\n')
            f.write('\nsentence miss: ' + np.array2string(sentence_miss))

            # f.write('\nsamples_mcm: \n')
            # for k, v in sample_subclass_metrics.items():
            #     f.write(str(k) + ': ' + ''.join(np.array2string(samples_mcm[k - 1]).splitlines())
            #             + '\t' + str(v) + '\n')
        f.close()

        return

    def sample_cm(self, data, label_map):
        subclass_mcm = np.empty([19, 2, 2], dtype=int)

        for name, index in label_map.items():
            one = data[name.replace('_sentence', '')].tolist()
            tn = one.count(0)
            fn = one.count(1)
            tp = one.count(2)
            fp = one.count(-1)
            subclass_mcm[index - 1] = np.array([[tn, fp], [fn, tp]])

        return subclass_mcm

    def excel_init(self, label_map):
        result_excel = xlwt.Workbook(encoding='utf-8')
        current = {}
        labelnames = []
        for label, index in label_map.items():
            sheet = result_excel.add_sheet(str(index) + '-' + label.replace('_sentence', ''))
            labelnames.append(str(index) + '-' + label.replace('_sentence', ''))
            sheet.write(0, 0, 'filename')
            sheet.write(0, 1, '漏判句')
            sheet.write(0, 2, '漏判句标签')
            sheet.write(0, 3, '漏判句预测结果')
            sheet.write(0, 4, 'filename')
            sheet.write(0, 5, '误判句')
            sheet.write(0, 6, '误判句标签')
            sheet.write(0, 7, '误判句预测结果')
            current[str(index) + '-' + label.replace('_sentence', '')] = [1, 1]

        return result_excel, labelnames, current

    def misjudge_export(self, filename, target_list, pred_list, sent_list,
                        result_excel, labelnames, current):

        for target, pred, sentence in zip(target_list, pred_list, sent_list):
            if target == pred:
                continue
            for i in range(len(target)):
                sheet = result_excel.get_sheet(labelnames[i])
                if target[i] == 1 and pred[i] == 0:  # 漏判
                    sheet.write(current[labelnames[i]][0], 0, filename)
                    sheet.write(current[labelnames[i]][0], 1, sentence)

                    target_int = [i + 1 for i, x in enumerate(target) if x == 1]
                    sheet.write(current[labelnames[i]][0], 2, str(target_int))
                    pred_int = [i + 1 for i, x in enumerate(pred) if x == 1]
                    sheet.write(current[labelnames[i]][0], 3, str(pred_int))
                    current[labelnames[i]][0] += 1

                elif target[i] == 0 and pred[i] == 1:  # 误判
                    sheet.write(current[labelnames[i]][1], 4, filename)
                    sheet.write(current[labelnames[i]][1], 5, sentence)

                    target_int = [i + 1 for i, x in enumerate(target) if x == 1]
                    sheet.write(current[labelnames[i]][1], 6, str(target_int))
                    pred_int = [i + 1 for i, x in enumerate(pred) if x == 1]
                    sheet.write(current[labelnames[i]][1], 7, str(pred_int))
                    current[labelnames[i]][1] += 1

                else:
                    continue

        return result_excel, labelnames, current


class Evaluator:
    def __init__(self):
        """Initialize my evaluator.
        """
        super(Evaluator, self).__init__()
        # , model, device
        # self.model = model
        # self.device = device

    def evaluate(slef, tokenizer, model, data_loader, device, sigmoid_threshold=0.80):
        """Evaluate model on data loader in device.

        Args:
            tokenizer: to decode sentence
            model: model to be evaluate
            data_loader: torch.utils.data.DataLoader
            device: cuda or cpu
            sigmoid_threshold:
        Returns:
            answer list, sent_list
        """
        model.eval()
        answer_list, sent_list = [], []
        for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
            # for _, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits, _ = model(*batch)

            sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in batch[0]])

            predictions = nn.Sigmoid()(logits)
            compute_pred = [[1 if one > sigmoid_threshold else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, sent_list

    def evaluate2stage(self, tokenizer, model, data_loader, device, sigmoid_threshold=0.80):
        """Evaluate model on data loader in device.

        Args:
            tokenizer: to decode sentence
            model: model to be evaluate
            data_loader: torch.utils.data.DataLoader
            device: cuda or cpu
            sigmoid_threshold:
        Returns:
            answer list, sent_list
        """
        from torch import nn
        model.eval()
        answer_list, sent_list = [], []
        for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits, logits2, to_index = model(*batch)

            if not to_index.numel():
                continue

            goal_sent = torch.index_select(batch[0].long(), 0, to_index.to(device))
            sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in goal_sent])

            predictions = nn.Sigmoid()(logits2)
            compute_pred = [[1 if one > sigmoid_threshold else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, sent_list

    def evaluate_with_labels(self, tokenizer, model, data_loader, device, sigmoid_threshold=0.80):
        model.eval()
        answer_list, sent_list, labels = [], [], []
        for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits, _ = model(*batch[:-1])
            labels.extend(batch[-1].detach().cpu().numpy().tolist())
            sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in batch[0]])

            predictions = nn.Sigmoid()(logits)
            compute_pred = [[1 if one > sigmoid_threshold else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, sent_list, labels

    def evaluate2stage_with_labels(self, tokenizer, model, data_loader, device, sigmoid_threshold):
        model.eval()
        answer_list, sent_list, labels = [], [], []
        b1_preds, b1_labels = [], []
        for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits, logits2, to_index = model(*batch[:-1])

            if not to_index.numel():
                continue

            # pred_b1 = nn.Sigmoid()(logits)  # , dim=1
            # index_b1 = torch.unsqueeze(torch.max(batch[-1], dim=1).values, 1)
            # # target (batch_size, 19) -> (batch_size, 2)
            # target_b1 = torch.zeros(batch[0].shape[0], 2).to(device).scatter_(1, index_b1, 1)

            pred_b1 = torch.max(nn.functional.softmax(logits, dim=1), dim=1).indices  # (batchsize, 1)
            b1_preds.extend(pred_b1.detach().cpu().numpy().tolist())

            index_b1 = torch.unsqueeze(torch.max(batch[-1], dim=1).values, 1)
            mid = torch.zeros(1, 2).to(device).scatter_(1, index_b1, 1)
            target_b1 = torch.max(mid, dim=1).indices
            b1_labels.extend(target_b1.detach().cpu().numpy().tolist())

            target_b2 = torch.index_select(batch[-1].long(), 0, to_index.to(device))
            labels.extend(target_b2.detach().cpu().numpy().tolist())

            goal_sent = torch.index_select(batch[0].long(), 0, to_index.to(device))
            sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in goal_sent])

            predictions = nn.Sigmoid()(logits2).detach().cpu().numpy().tolist()
            compute_pred = [[1 if one > sigmoid_threshold else 0 for one in row] for row in predictions]
            for idx, pred in enumerate(compute_pred):
                if pred == [0] * len(pred):
                    pred[predictions[idx].index(max(predictions[idx]))] = 1
                    compute_pred[idx] = pred
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, sent_list, labels, b1_preds, b1_labels
