"""Predict.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os
import json
from torch.utils.data import DataLoader
from utils import read_annotation
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
        self.data = read_annotation(filename=test_file, sheet_name=test_sheet)
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
                labels_int = [pred.index(k) + 1 for k in pred if k == 1]
                labels = [k for k, v in self.label_map.items() if v in labels_int]
                for label in labels:
                    one_data[label.replace('_sentence', '')] = 1
                    one_data[label] = ' '.join([one_data[label], sent])
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
