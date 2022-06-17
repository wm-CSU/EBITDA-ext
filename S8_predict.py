"""Predict.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os
import json
import numpy as np
import time
from torch.utils.data import DataLoader
from utils import read_annotation
from S1_preprocess import Drop_Redundance
from S4_dataset import TestData
from S7_evaluate import evaluate


class Prediction:
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test.xlsx',
                 test_sheet='Sheet1',
                 test_txt='data/test_adjust_txt/'
                 ):
        self.label_map = self._get_label_map()
        ori_data = read_annotation(filename=test_file, sheet_name=test_sheet)
        self.data = Drop_Redundance(ori_data, 'data/adjust_test.xlsx', Train=False)
        self.data_preprocess()
        self.test_txt = test_txt
        self.dataset_tool = TestData(vocab_file, max_seq_len=max_seq_len)

    def evaluate_for_all(self, model, device,
                         to_file='data/predict.xlsx', to_sheet='Sheet1',
                         multi_class: bool = False):
        '''
        遍历测试集，逐条数据预测
        :param model:
        :param device:
        :param to_file:
        :param to_sheet:
        :param multi_class:
        :return:
        '''
        for index, one in self.data.iterrows():
            filename = os.path.join(self.test_txt, index + '.txt')
            one_dataset = self.dataset_tool.load_from_txt(filename)
            one_loader = DataLoader(one_dataset, batch_size=1, shuffle=False)

            predictions, sent = evaluate(self.dataset_tool.tokenizer, model, one_loader,
                                         device, multi_class=multi_class)
            if multi_class:
                self.data.loc[index, :] = self.sent_data_align_multi_class(sent, predictions=predictions, one_data=one)
            else:
                self.data.loc[index, :] = self.sent_data_align(sent, predictions=predictions, one_data=one)

        self.data.to_excel(to_file, to_sheet)

        return

    def sent_data_align(self, sent_list, predictions, one_data):
        """
        对齐句子与标签
        :param sent_list: [str, ...]
        :param predictions: [int, ...]
        :return:
        """
        for sent, pred in zip(sent_list, predictions):
            if pred != '0':
                label = [k for k, v in self.label_map.items() if str(v) == pred]
                one_data[label[0].replace('_sentence', '')] = 1
                one_data[label[0]] = ' '.join([one_data[label[0]], sent])
            else:
                continue

        return one_data

    def sent_data_align_multi_class(self, sent_list, predictions, one_data):
        """
        对齐句子与标签
        :param sent_list: [str, ...]
        :param predictions: [[int], [int]...]
        :return:
        """
        for sent, pred in zip(sent_list, predictions):
            if pred != [0] * len(pred):  # 有预测值，定位对应文本
                labels_int = [i + 1 for i, x in enumerate(pred) if x == 1]
                labels = [k for k, v in self.label_map.items() if v in labels_int]
                for label in labels:
                    one_data[label.replace('_sentence', '')] = 1
                    one_data[label] = '; '.join([one_data[label], sent])
            else:
                continue

        return one_data

    def data_preprocess(self):
        self.data.fillna('', inplace=True)
        for k, v in self.label_map.items():
            self.data[k.replace('_sentence', '')] = 0
        return

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map


class PredictionWithlabels:
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test.xlsx',
                 test_sheet='Sheet1',
                 test_txt='data/test_adjust_txt/'
                 ):
        self.label_map = self._get_label_map()
        ori_data = read_annotation(filename=test_file, sheet_name=test_sheet)
        self.data = Drop_Redundance(ori_data, 'data/adjust_test.xlsx', Train=False)
        # self.data = read_annotation(filename=test_file, sheet_name=test_sheet)
        from S3_sentence_division import Division
        self.division = Division(self.data)
        self.test_file = test_file
        self.test_txt = test_txt
        self.dataset_tool = TestData(vocab_file, max_seq_len=max_seq_len)

    def evaluate_for_all(self, model, device,
                         to_file='data/predict.xlsx', to_sheet='Sheet1',
                         multi_class: bool = False):
        '''
        遍历测试集，逐条数据预测
        :param model:
        :param device:
        :param to_file:
        :param to_sheet:
        :param multi_class:
        :return:
        '''
        target_list, pred_list = [], []
        for index, one in self.data.iterrows():
            filename = os.path.join(self.test_txt, index + '.txt')
            one_sent = self.division.txt2sent(filename=filename)
            one_label = self.division.sent_label(one_data=one, one_sent=one_sent,
                                                 label_map=self.label_map)
            one_dataset, _ = self.dataset_tool.load_one(one_sent=one_sent, one_label=one_label)
            one_loader = DataLoader(one_dataset, batch_size=1, shuffle=False)

            predictions, sent, labels = self.evaluate_for_test(self.dataset_tool.tokenizer, model, one_loader, device)
            target_list.extend(labels)
            pred_list.extend(predictions)

            self.data.loc[index, :] = self.sent_data_align_multi_class(sent, predictions=predictions, one_data=one)

        self.data.to_excel(to_file, to_sheet)

        from S7_evaluate import subclass_confusion_matrix, compute_metrics
        mcm = subclass_confusion_matrix(targetSrc=target_list, predSrc=pred_list)
        result = compute_metrics(labels=target_list, preds=pred_list)
        with open('result/result.txt', 'a+') as f:
            f.write('\n\n\n' + time.asctime() + '   PredictionWithlabels   ' + self.test_file + ' -> ' + to_file + '\n')
            f.write(str([str(k) + ': ' + str(format(v, '.6f')) for k, v in result.items() if
                         k != 'subclass_confusion_matrix']) + '\n')
            f.write(''.join(np.array2string(mcm).splitlines()))
        f.close()

        return

    def evaluate_for_test(self, tokenizer, model, data_loader, device):
        import torch
        from torch import nn
        model.eval()
        answer_list, sent_list, labels = [], [], []
        # for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
        for _, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits, _ = model(*batch[:-1])
            labels.extend(batch[-1].detach().cpu().numpy().tolist())
            sent_list.extend([tokenizer.decode(x, skip_special_tokens=True) for x in batch[0]])

            predictions = nn.Sigmoid()(logits)
            compute_pred = [[1 if one > 0.90 else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, sent_list, labels

    def sent_data_align_multi_class(self, sent_list, predictions, one_data):
        """
        对齐句子与标签
        :param sent_list: [str, ...]
        :param predictions: [[int], [int]...]
        :return:
        """
        for sent, pred in zip(sent_list, predictions):
            if pred != [0] * len(pred):  # 有预测值，定位对应文本
                labels_int = [i + 1 for i, x in enumerate(pred) if x == 1]
                labels = [k for k, v in self.label_map.items() if v in labels_int]
                for label in labels:
                    if one_data[label.replace('_sentence', '')] == 1:
                        one_data[label.replace('_sentence', '')] = 2
                        one_data[label] = ' '.join([str(one_data[label]), '\n\n'])
                    if one_data[label.replace('_sentence', '')] == 0:
                        one_data[label.replace('_sentence', '')] = -1
                    one_data[label] = '; '.join([str(one_data[label]), sent])
            else:
                continue

        return one_data

    def data_preprocess(self):
        self.data.fillna('', inplace=True)
        for k, v in self.label_map.items():
            self.data[k.replace('_sentence', '')] = 0
        return

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map
