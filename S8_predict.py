"""Predict.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from utils import read_annotation
from S1_preprocess import Drop_Redundance
from S4_dataset import TestData
from S7_evaluate import Metrics, Evaluator


class pred_tools:
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test.xlsx',
                 test_sheet='Sheet1',
                 test_txt='data/test_adjust_txt/'
                 ):
        """Initialize my evaluator.
        """
        super(pred_tools, self).__init__()
        self.test_file = test_file
        self.test_txt = test_txt

        self.label_map = self._get_label_map()
        self.evaluator = Evaluator()

        self.dataset_tool = TestData(vocab_file, max_seq_len=max_seq_len)
        ori_data = read_annotation(filename=test_file, sheet_name=test_sheet)
        self.data = Drop_Redundance(ori_data, Train=False)

    def sent_data_align(self, sent_list, predictions, one_data):
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

    def data_preprocess(self, data):
        data.fillna('', inplace=True)
        for k, v in self.label_map.items():
            data[k.replace('_sentence', '')] = 0
        return

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map


class Prediction(pred_tools):
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test.xlsx',
                 test_sheet='Sheet1',
                 test_txt='data/test_adjust_txt/'
                 ):
        super(Prediction, self).__init__(
            vocab_file, max_seq_len, test_file, test_sheet, test_txt)
        self.data_preprocess(self.data)

    def evaluate_for_all(self, model, device,
                         to_file='data/predict.xlsx', to_sheet='Sheet1',
                         sigmoid_threshold=0.80):
        ''' 遍历测试集，逐条数据预测
        '''
        for index, one in self.data.iterrows():
            filename = os.path.join(self.test_txt, index + '.txt')
            one_dataset = self.dataset_tool.load_from_txt(filename)
            one_loader = DataLoader(one_dataset, batch_size=1, shuffle=False)

            predictions, sent = self.evaluator.evaluate2stage(self.dataset_tool.tokenizer, model, one_loader,
                                                              device, sigmoid_threshold=sigmoid_threshold)
            self.data.loc[index, :] = self.sent_data_align(sent, predictions=predictions, one_data=one)

        self.data.to_excel(to_file, to_sheet)

        return


class PredictionWithlabels(pred_tools):
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test.xlsx',
                 test_sheet='Sheet1',
                 test_txt='data/test_adjust_txt/'
                 ):
        super(PredictionWithlabels, self).__init__(
            vocab_file, max_seq_len, test_file, test_sheet, test_txt)

        from S3_sentence_division import Division
        self.division = Division(self.data)

        self.metrics = Metrics()

    def evaluate_for_all(self, model, device,
                         to_file='data/predict.xlsx', to_sheet='Sheet1',
                         metrics_save_path: str = 'result/result',
                         sigmoid_threshold=0.80):
        '''
        遍历测试集，逐条数据预测
        :param model:
        :param device:
        :param to_file:
        :param to_sheet:
        :param sigmoid_threshold:
        :return:
        '''
        target_list, pred_list, sent_list = [], [], []
        b1_preds_list, b1_labels_list = [], []
        result_excel, labelnames, current = self.metrics.excel_init(self.label_map)

        for index, one in self.data.iterrows():
            filename = os.path.join(self.test_txt, index + '.txt')
            one_sent = self.division.txt2sent(filename=filename)
            one_label = self.division.sent_label(one_data=one, one_sent=one_sent,
                                                 label_map=self.label_map)
            one_dataset, _ = self.dataset_tool.load_one(one_sent=one_sent, one_label=one_label)
            one_loader = DataLoader(one_dataset, batch_size=1, shuffle=False)
            if not one_loader:
                print('{} is null! Please change it.'.format(filename))

            predictions, sent, labels, b1_preds, b1_labels = self.evaluator.evaluate2stage_with_labels(
                self.dataset_tool.tokenizer, model, one_loader, device, sigmoid_threshold
            )
            target_list.extend(labels)
            pred_list.extend(predictions)
            sent_list.extend(sent)
            b1_preds_list.extend(b1_preds)
            b1_labels_list.extend(b1_labels)

            result_excel, labelnames, current = self.metrics.misjudge_export(
                filename=index, target_list=labels, pred_list=predictions, sent_list=sent,
                result_excel=result_excel, labelnames=labelnames, current=current
            )

            self.data.loc[index, :] = self.align_with_labels(sent, predictions=predictions, one_data=one)
        print('pred success.')
        # self.data.to_excel(to_file, to_sheet)
        result_excel.save(metrics_save_path + '-eval_yj_sentence.xls')
        self.metrics.metrics_output(pred_to_file=to_file, target_list=target_list, pred_list=pred_list,
                                    sent_list=sent_list, b1_preds=b1_preds_list, b1_labels=b1_labels_list,
                                    filename=metrics_save_path + '-metrics.txt')
        self.eval_yj(save_path=metrics_save_path + '-eval_yj.xls')

        return

    def align_with_labels(self, sent_list, predictions, one_data):
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
                        one_data[label] = ''.join([str(one_data[label]), '\n\n'])
                    if one_data[label.replace('_sentence', '')] == 0:
                        one_data[label.replace('_sentence', '')] = -1
                        one_data[label] = ''
                    one_data[label] = '; '.join([str(one_data[label]), sent])
            else:
                continue

        return one_data

    def eval_yj(self, save_path):
        y_true, y_pred, sent = [], [], []
        for index, one in self.data.iterrows():
            one_pair = {}
            one_true, one_pred = [], []
            for i in range(len(one)):
                if one[i] == -1:
                    one_true.append(0)
                    one_pred.append(1)
                    class_num = self.label_map[self.data.columns.tolist()[i + 1]] - 1
                    one_pair[class_num] = one[i + 1]
                elif one[i] == 0:
                    one_true.append(0)
                    one_pred.append(0)
                elif one[i] == 1:
                    one_true.append(1)
                    one_pred.append(0)
                elif one[i] == 2:
                    one_true.append(1)
                    one_pred.append(1)
                    class_num = self.label_map[self.data.columns.tolist()[i + 1]] - 1
                    one_pair[class_num] = one[i + 1]
                else:
                    continue
            y_true.append(one_true)
            y_pred.append(one_pred)
            sent.append(one_pair)

        import xlwt
        from re_7_8 import evaluate, save_errorfile
        result_excel = xlwt.Workbook(encoding='utf-8')
        # 评估结果
        evaluate(np.array(y_true), np.array(y_pred), result_excel)

        filename = self.data.index.tolist()
        # 保存错误文件信息
        save_errorfile(np.array(y_true), np.array(y_pred), sent, filename, result_excel)
        result_excel.save(save_path)

        return
    # unused func.
    # def misjudge_export(self, target_list, pred_list, sent_list, filename):
    #     result_excel, labelnames, current = self._excel_init()
    #     for target, pred, sentence in zip(target_list, pred_list, sent_list):
    #         if target == pred:
    #             continue
    #         for i in range(len(target)):
    #             sheet = result_excel.get_sheet(labelnames[i])
    #             if target[i] == 1 and pred[i] == 0:  # 漏判
    #                 sheet.write(current[labelnames[i]][0], 0, sentence)
    #
    #                 target_int = [i + 1 for i, x in enumerate(target) if x == 1]
    #                 sheet.write(current[labelnames[i]][0], 1, str(target_int))
    #                 pred_int = [i + 1 for i, x in enumerate(pred) if x == 1]
    #                 sheet.write(current[labelnames[i]][0], 2, str(pred_int))
    #                 current[labelnames[i]][0] += 1
    #             elif target[i] == 0 and pred[i] == 1:  # 误判
    #                 sheet.write(current[labelnames[i]][1], 3, sentence)
    #
    #                 target_int = [i + 1 for i, x in enumerate(target) if x == 1]
    #                 sheet.write(current[labelnames[i]][1], 4, str(target_int))
    #                 pred_int = [i + 1 for i, x in enumerate(pred) if x == 1]
    #                 sheet.write(current[labelnames[i]][1], 5, str(pred_int))
    #                 current[labelnames[i]][1] += 1
    #             else:
    #                 continue
    #
    #     result_excel.save(filename)
    #     return
