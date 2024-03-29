# coding: utf-8
"""Paragraphs are divided into short sentences for subsequent classification.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import time
import difflib
import json
import os.path
import re
import pandas as pd
from utils import get_path, read_annotation, sent_process
from S1_preprocess import Drop_Redundance
from S2_EBITDA_locate import Paragraph_Extract


class Division:
    def __init__(self, data, Train: bool = True):
        self.num_classes = 19
        if Train:
            self.data = Drop_Redundance(data)  # 原生数据预处理（冗余删除）
        else:
            self.data = data

    def deal(self, label_map, txtfilepath, labelfilepath):
        # 处理整个数据集
        label_map = self._get_label_map(filename=label_map)
        get_path(labelfilepath)
        labels = []
        for index, one in self.data.iterrows():
            filename = os.path.join(txtfilepath, index + '.txt')
            toname = os.path.join(labelfilepath, index + '.txt')
            if os.path.isfile(filename):
                one_sent = self.txt2sent(filename=filename)
                one_label = self.sent_label(one_data=one, one_sent=one_sent, label_map=label_map)
                labels.append(one_label)
                self.label2txt(label_dict=one_label, filename=toname)
                print('sentence divide:  ', filename, 'is dealed.')
            else:
                labels.append({})
                print('sentence divide:  ', filename, 'is not filename.')

        return labels

    def txt2sent(self, filename):
        sentence = []
        with open(filename, 'r', encoding='utf8') as f:
            paragraph = f.readlines()
            for para in paragraph:
                if len(para) < 8:
                    continue
                mid = self.sent_split(paragraph=para)
                para2sent = self.sent_resplit(paragraph=mid)
                for one in para2sent:
                    sentence.append(sent_process(one).lower())
                # sentence.extend(para2sent)
        f.close()

        return [x for x in sentence if x != '']

    def sent_split(self, paragraph: str):
        # 划分完成后，括号保留
        para2sent = re.split(';|\D\.(?!\d)|\D\((?!loss)[\s\S]{1,4}\)', paragraph.strip())  # 句号 分号划分
        seg_word = re.findall(';|\D\.(?!\d)|\D\((?!loss)[\s\S]{1,4}\)',
                              paragraph.strip())  # 保留分割符号，置于句尾，比如标点符号
        seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
        para2sent = [x + y for x, y in zip(para2sent, seg_word)]  # 顺序可根据需求调换

        para2sent = [re.sub('\((?!loss)[\s\S]{1,4}\)$', '', sent) for sent in para2sent]  # 除去句尾的括号项
        para2sent = [re.sub('( +|\t|\n|_+)', " ", sent) for sent in para2sent]

        return [one for one in para2sent if len(one) > 8]

    def sent_resplit(self, paragraph: list):
        new_para = []
        for index, para in enumerate(paragraph):
            if len(para.split()) > 100:  # or not bool(re.search(r'\(.{1,3}?\)', para))
                re_sent = self._loog_str_div(para)
                new_para.extend(re_sent)
            else:
                new_para.append(para)
                continue

        return [one for one in new_para if len(one) > 8]

    def _loog_str_div(self, para: str = '') -> list:
        re_sent = re.split(',|\(.{5,}?\)', para.strip())
        # 保留分割符号，置于句尾，比如标点符号
        seg_word = re.findall(',|\(.{5,}?\)', para.strip())
        seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
        re_sent = [x + y for x, y in zip(re_sent, seg_word)]  # 顺序可根据需求调换
        for one in range(len(re_sent) - 1):
            if re_sent[one].endswith(')'):
                re_sent[one] = re_sent[one] + re_sent[one + 1]
                re_sent[one + 1] = ''

        # 聚合过短句子
        end_sent = []
        mid = ''
        for sent in re_sent:
            if len(mid) + len(sent) < 300:
                mid = mid + sent
            else:
                end_sent.append(mid)
                mid = sent
        end_sent.append(mid)

        return [one for one in end_sent if len(one) > 8]

    def sent_label(self, one_data, one_sent, label_map):
        # 一条数据的sentence与label对应
        label = {}
        for sent in one_sent:
            label[sent.strip()] = [0] * self.num_classes

        for (name, value) in one_data.items():
            if pd.isna(value):
                continue
            elif 'sentence' in name or 'Sentence' in name:
                value = value.replace('，', ',')
                for sent in one_sent:
                    if self._determine_same(sent, value) > 0.9:
                        label[sent.strip()][label_map[name] - 1] = 1  # multi-class
                    else:
                        continue
            else:
                continue

        return label

    def _determine_same(self, split_sent, target_sent):
        if len(split_sent) < 3 or len(target_sent) < 3:
            return 0
        split2 = sent_process(split_sent).lower()
        target2 = sent_process(target_sent).lower()
        if split2 == target2:
            score = 1
        elif len(split2.split()) > 20 and (target2 in split2 or split2 in target2):
            score = 1
        else:
            score = difflib.SequenceMatcher(None, split_sent.strip(), target_sent.strip()).ratio()
            pass

        return score

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map

    def label2txt(self, label_dict, filename):
        with open(filename, 'w', encoding='utf8') as f:
            for sent, label in label_dict.items():
                f.write(sent + '\t' + str(label) + '\n')
            f.close()

        return


if __name__ == '__main__':
    # TRAIN_NAMES = ['蔡子悦', '高尚', '李佳慧', '马成辰', '宋瑶', '童婧曦', '王珏']
    TRAIN_NAMES = ['caiziyue', 'gaoshang', 'lijiahui', 'machengchen', 'songyao', 'tongjingxi', 'wangjue']
    for name in TRAIN_NAMES:
        filename = 'data/train_file/' + name + '.xlsx'
        ori_data = read_annotation(filename=filename, sheet_name='Sheet1')
        ext = Paragraph_Extract(ori_data, Train=False)
        start = time.time()
        # data = ext.deal(input_path=os.path.join('data/train_txt/', name),
        #                 output_path=os.path.join(r'data/train_adjust_txt/', name))
        division = Division(ori_data, Train=True)
        division.deal(label_map=r'data/label_map.json', txtfilepath=os.path.join(r'data/train_adjust_txt/', name),
                      labelfilepath=os.path.join(r'data/sent_multi_label/', name))
        end = time.time()
        print('{} use time: {} s'.format(name, end - start))

    txt_set = r'data/txt_set/'
    ebitda_txt = r'data/train_adjust_txt/'
    multi_class_sent_txt = r'data/sent_multi_label'
    # 两批数据处理
    # ori_data = read_annotation(filename=r'data/train.xlsx', sheet_name='Sheet1')
    # # ext = Paragraph_Extract(ori_data)
    # # data = ext.deal(input_path=txt_set, output_path=ebitda_txt)
    # division = Division(ori_data, Train=False)
    # division.deal(label_map=r'data/label_map.json', txtfilepath=ebitda_txt, labelfilepath=multi_class_sent_txt)
    # # test
    # multi_class_test_sent_txt = r'data/sent_multi_label'
    # ori_data = read_annotation(filename=r'data/batch_test.xlsx', sheet_name='Sheet1')
    # # ext = Paragraph_Extract(ori_data)
    # # data = ext.deal(input_path=txt_set, output_path=ebitda_txt)
    # division = Division(ori_data)
    # division.deal(label_map=r'data/label_map.json', txtfilepath=ebitda_txt, labelfilepath=multi_class_test_sent_txt)

    pass
