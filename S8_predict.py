"""Predict.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

import os
import json
from utils import read_annotation


class Predict_postprocess:
    def __init__(self, test_file='data/test.xlsx',
                 test_sheet='Sheet1',
                 test_txt='data/test_adjust_txt/'
                 ):
        self.label_map = self._get_label_map()
        self.data = read_annotation(filename=test_file, sheet_name=test_sheet)
        self.data_preprocess()
        self.para_list = self._get_test_set(test_txt=test_txt)
        pass

    def deal(self, sent_list, predictions,
             to_file='data/predict.xlsx', to_sheet='Sheet1'
             ):
        self.sent_data_align(sent_list, predictions=predictions)
        self.data.to_excel(to_file, to_sheet)
        return

    def data_preprocess(self):
        self.data.fillna('', inplace=True)
        for k, v in self.label_map.items():
            self.data[k.replace('_sentence', '')] = 0
        return

    def _get_test_set(self, test_txt='data/test_adjust_txt/'):
        """to get sentence list.
        :param test_file:
        :param test_sheet:
        :param test_txt:
        :return:
            para_list: [[para1, para2]]
        """
        para_list = []
        for index, one in self.data.iterrows():
            filename = os.path.join(test_txt, index + '.txt')
            with open(filename, 'r', encoding='utf8') as f:
                paragraph = f.read()
            f.close()
            para_list.append(paragraph)

        return para_list

    def sent_data_align(self, sent_list, predictions, ):
        for (index, _), para in zip(self.data.iterrows(), self.para_list):
            for sent, pred in zip(sent_list, predictions):
                if pred != '0' and sent in para:
                    label = [k for k, v in self.label_map.items() if str(v) == pred]
                    self.data.loc[index, label[0].replace('_sentence', '')] = 1
                    self.data.loc[index, label[0]] = ' '.join([self.data.loc[index, label[0]], sent])
                else:
                    continue
        return self.data

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map


