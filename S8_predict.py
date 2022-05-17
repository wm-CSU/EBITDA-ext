"""Predict.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

import os
import json
from utils import read_annotation


class Predict_postprocess:
    def __init__(self):
        self.LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19']
        self.label_map = self._get_label_map()

        pass

    def deal(self, predictions,
             test_file='data/test.xlsx', test_sheet='Sheet1', test_txt='data/test_adjust_txt/',
             to_file='data/predict.xlsx', to_sheet='Sheet1'
             ):
        sent_list = self.get_test_set(test_file=test_file, test_sheet=test_sheet, test_txt=test_txt)
        sent_label_pair = self.sent_label_align(sent_list, predictions=predictions)
        self.predict2excel(sent_label_pair, to_file=to_file, to_sheet=to_sheet)

        return

    def get_test_set(self, test_file='data/test.xlsx',
                     test_sheet='Sheet1',
                     test_txt='data/test_adjust_txt/'):
        """to get sentence list.
        :param test_file:
        :param test_sheet:
        :param test_txt:
        :return:
            sent_list: [[sent1, sent2]]
        """
        from S3_sentence_division import Division
        ori_data = read_annotation(filename=test_file, sheet_name=test_sheet)
        division = Division(ori_data)
        self.data = division.data

        sent_list = []
        for index, one in division.data.iterrows():
            filename = os.path.join(test_txt, index + '.txt')
            sentences = division.txt2sent(filename=filename)
            sent_list.append(sentences)

        return sent_list

    def sent_label_align(self, sent_list, predictions):
        sent_label_pair = []
        index = 0
        for one_data in sent_list:
            one_pair = {}
            for sent in one_data:
                one_pair[sent] = predictions[index]
                index += 1
            sent_label_pair.append(one_pair)

        return sent_label_pair

    def predict2excel(self, sent_label_pair, to_file='data/predict.xlsx', to_sheet='Sheet1'):
        """change predict to excel.
        :param sent_label_pair: [{sent: class, }]
        :param to_file:
        :param to_sheet:
        :return:
        """
        for _, one, sent_label in zip(self.data.iterrows(), sent_label_pair):
            for sent, label in sent_label.items():
                if label == '0':
                    continue
                else:
                    column = [k for k, v in self.label_map.items() if str(v) == label]
                    one[column[0]] += sent
        self.data.to_excel(to_file, to_sheet)

        return

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map


if __name__ == '__main__':
    pred = Predict_postprocess()
    sent_list = pred.get_test_set(test_file='data/test.xlsx', test_sheet='Sheet1', test_txt='data/test_adjust_txt/')
    sent_label_pair = pred.sent_label_align(sent_list, predictions=__Pred)
    pred.predict2excel(sent_label_pair, to_file='data/predict.xlsx', to_sheet='Sheet1')
    pass
