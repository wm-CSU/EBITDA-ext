# coding: utf-8
"""analysis excel and derive read_excel.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os.path
import re
import copy
import numpy as np
import pandas as pd
from utils import get_path, read_annotation, find_lcsubstr
from S1_preprocess import Drop_Redundance


class Excel_analysis:
    """
    机器读取结果可视化，用于改进标注数据
    """

    def __init__(self, data, Train: bool = True):
        self.num_classes = 19
        if Train:
            self.data = Drop_Redundance(data, excel_path=r'../data/new.xlsx', )  # 原生数据预处理（冗余删除）
        else:
            self.data = data

        self.read_data = copy.deepcopy(self.data)
        drop_col = [
            'asset_writedown_sentence',
            'asset_disp_loss_sentence',
            'restruct_exp_sentence',
            'liti_settelment_loss_sentence',
            'goodwill_impairment_sentence',
            'acquisition_charge_sentence',
            'debt_extinguish_loss_sentence',
            'disconti_loss_sentence',
            'relocation_exp_sentence',
            'in_process_rnd_charge_sentence',
            'acct_change_loss_sentence',
            'asset_disp_gain_sentence',
            'asset_writeup_sentence',
            'insur_proceed_gain_sentence',
            'restruct_inc_sentence',
            'disconti_gain_sentence',
            'debt_extinguish_gain_sentence',
            'liti_settelment_gain_sentence',
            'acct_change_gain_sentence',
        ]
        for col in drop_col:
            col_index = list(self.data).index(col)
            self.read_data.drop(col, axis=1, inplace=True)
            self.read_data.insert(col_index, col, '')
            # self.read_data = pd.concat([self.read_data, pd.DataFrame(columns=drop_col)], sort=False)
        self.read_data.fillna('', inplace=True)
        self.Marking_length_statistics(drop_col)

    def Marking_length_statistics(self, col):
        # mark = [i for i in self.data[col] if pd.isna(i)==False]
        mark = []
        for one in col:
            mark.extend([i for i in self.data[one] if pd.isna(i) == False])
        # print(mark)
        mark_len = [len(i) for i in mark]
        print('min: {}\nmean: {}\n95 percentile:{}\nmax: {}'.format(
            np.min(mark_len), np.mean(mark_len), np.percentile(mark_len, 95), np.max(mark_len)))

    def deal(self, txtfilepath, tofilepath):
        # labels = []
        for index, one in self.data.iterrows():
            filename = os.path.join(txtfilepath, index + '.txt')
            if os.path.isfile(filename):
                one_sent = self.txt2sent(filename=filename)
                one_label = self.sent_label(one_data=one, one_sent=one_sent, read_one=self.read_data.loc[index, :])
                self.read_data.loc[index, :] = one_label

        self.read_data.to_excel(tofilepath)

        return

    def txt2sent(self, filename):
        # 先根据 excel 将标准答案句抽取出来，再根据规则分句。
        sentence = []
        with open(filename, 'r', encoding='utf8') as f:
            paragraph = f.readlines()
            for para in paragraph:
                para2sent = self.sent_split(paragraph=para)
                sentence.extend(para2sent)
        f.close()

        return [x for x in sentence if x != '']

    def sent_split(self, paragraph):
        # 段落划分为句子并除去过短元素（如单数字或空格）
        para2sent = re.split(';|\.|\([\s\S]{1,4}\)', paragraph.strip())
        # 保留分割符号，置于句尾，比如标点符号
        seg_word = re.findall(';|\.|\([\s\S]{1,4}\)', paragraph.strip())
        seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
        para2sent = [x + y for x, y in zip(para2sent, seg_word)]  # 顺序可根据需求调换
        # 除去句尾的括号项
        para2sent = [re.sub('\([\s\S]{1,4}\)$', '', sent) for sent in para2sent]

        return [one for one in para2sent if len(one) > 10]

    def sent_label(self, one_data, one_sent, read_one):
        # 一条数据的sentence与label对应
        # label = one_data
        for (name, value) in one_data.items():
            if pd.isna(value):
                continue
            elif 'sentence' in name:
                value = value.replace('，', ',')
                for sent in one_sent:
                    # sub_str, _ = find_lcsubstr(value.strip(), sent.strip())
                    # if len(sub_str.split(' ')) >= 3 and sub_str.strip().endswith((',', ')', ';')):
                    if value.strip() in sent.strip() or sent.strip() in value.strip():
                        read_one[name.replace('_sentence', '')] = 1
                        read_one[name] = ' '.join([read_one[name], sent.strip()])  # multi-class
                    else:
                        continue
            else:
                continue

        return read_one


if __name__ == '__main__':
    ebitda_txt = r'../data/adjust_txt/'
    # 两批数据处理
    ori_data1 = read_annotation(filename=r'../data/train.xlsx', sheet_name='Sheet1')
    analysis1 = Excel_analysis(ori_data1)
    # analysis1.deal(txtfilepath=ebitda_txt, tofilepath=r'../data/batch_one_read.xlsx')

    # ori_data2 = read_annotation(filename=r'../data/batch_two.xlsx', sheet_name='Sheet1')
    # analysis2 = Excel_analysis(ori_data2)
    # analysis2.deal(txtfilepath=ebitda_txt, tofilepath=r'../data/batch_two_read.xlsx')

    # test
    test_ebitda_txt = r'../data/test_adjust_txt/'
    ori_data3 = read_annotation(filename=r'../data/test_yqh_with answer.xlsx', sheet_name='Sheet1')
    analysis3 = Excel_analysis(ori_data3)
    # analysis3.deal(txtfilepath=test_ebitda_txt, tofilepath=r'../data/batch_test_read.xlsx')
    # str1 = ' exclusive of extraordinary items and gains or-losses on sales of assets outside the ordinary course ' \
    #        'of business, in the aggregate arising from the sale，'
    # str2 = ' earnings before interest, taxes on income, depreciation and amortization (exclusive of extraordinary ' \
    #        'items and gains or-losses on sales of assets outside the ordinary course of business, plus'
    # sub_str, sub_len = find_lcsubstr(str1, str2)
    # print(sub_str, sub_len)
    # print(sub_str.split(' '), len(sub_str.split(' ')))
    pass
