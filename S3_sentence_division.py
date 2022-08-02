# coding: utf-8
"""Paragraphs are divided into short sentences for subsequent classification.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import difflib
import json
import os.path
import re
import pandas as pd
from utils import get_path, read_annotation
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
            else:
                labels.append({})
            print(filename, 'is dealed.')

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
                    sentence.append(self.sent_process(one))
                # sentence.extend(para2sent)
        f.close()

        return [x for x in sentence if x != '']

    def sent_split(self, paragraph):
        # 划分完成后，括号保留
        para2sent = re.split(';|\D\.(?!\d)|\D\((?!loss)[\s\S]{1,4}\)', paragraph.strip())  # 句号 分号划分
        seg_word = re.findall(';|\D\.(?!\d)|\D\((?!loss)[\s\S]{1,4}\)',
                              paragraph.strip())  # 保留分割符号，置于句尾，比如标点符号
        seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
        para2sent = [x + y for x, y in zip(para2sent, seg_word)]  # 顺序可根据需求调换

        para2sent = [re.sub('\((?!loss)[\s\S]{1,4}\)$', '', sent) for sent in para2sent]  # 除去句尾的括号项
        para2sent = [re.sub('( +|\t|\n|_+)', " ", sent) for sent in para2sent]

        return [one for one in para2sent if len(one) > 8]

    def sent_resplit(self, paragraph):
        for para in paragraph:
            if len(para.split()) > 100:  # resplit
                re_sent = re.split(',|\(.{5,}?\)', para.strip())
                # 保留分割符号，置于句尾，比如标点符号
                seg_word = re.findall(',|\(.{5,}?\)', para.strip())
                seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
                re_sent = [x + y for x, y in zip(re_sent, seg_word)]  # 顺序可根据需求调换
                for one in range(len(re_sent) - 1):
                    if re_sent[one].endswith(')'):
                        re_sent[one] = re_sent[one] + re_sent[one + 1]
                        re_sent[one + 1] = ''
                re_sent = [one for one in re_sent if len(one) > 8]

                paragraph.extend(re_sent)
                paragraph.remove(para)
            else:
                continue

        return [one for one in paragraph if len(one) > 8]

    def sent_label(self, one_data, one_sent, label_map):
        # 一条数据的sentence与label对应
        label = {}
        for sent in one_sent:
            label[sent.strip()] = [0] * self.num_classes

        for (name, value) in one_data.items():
            if pd.isna(value):
                continue
            elif 'sentence' in name:
                value = value.replace('，', ',')
                for sent in one_sent:
                    if self._determine_same(sent, value) > 0.9:
                        label[sent.strip()][label_map[name] - 1] = 1  # multi-class
                    else:
                        continue
            else:
                continue

        return label

    def _determine_same(self, split_sent, ori_sent):
        if len(split_sent) < 3 or len(ori_sent) < 3:
            return 0
        if split_sent in ori_sent.strip() or ori_sent.strip() in split_sent:
            score = 1
        else:
            score = difflib.SequenceMatcher(None, split_sent.strip(), ori_sent.strip()).ratio()
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

    def sent_process(self, para: str = None):
        para = re.sub(r'[*_+0-9]', '', para)
        words = para.split()
        need_del = []
        for index, word in enumerate(words):
            if not bool(re.search(r'[a-zA-Z]', word)):
                need_del.append(index)
        for i in sorted(need_del, reverse=True):
            del words[i]

        return ' '.join(words)


if __name__ == '__main__':
    txt_set = r'data/txt_set/'
    ebitda_txt = r'data/adjust_txt/'
    # sent_txt = r'data/sent_label'
    multi_class_sent_txt = r'data/sent_multi_label'
    # 两批数据处理
    ori_data = read_annotation(filename=r'data/train.xlsx', sheet_name='Sheet1')
    # ext = Paragraph_Extract(ori_data)
    # data = ext.deal(input_path=txt_set, output_path=ebitda_txt)
    division = Division(ori_data)
    division.deal(label_map=r'data/label_map.json', txtfilepath=ebitda_txt, labelfilepath=multi_class_sent_txt)

    # # test
    # multi_class_test_sent_txt = r'data/sent_multi_label'
    # ori_data = read_annotation(filename=r'data/batch_test.xlsx', sheet_name='Sheet1')
    # # ext = Paragraph_Extract(ori_data)
    # # data = ext.deal(input_path=txt_set, output_path=ebitda_txt)
    # division = Division(ori_data)
    # division.deal(label_map=r'data/label_map.json', txtfilepath=ebitda_txt, labelfilepath=multi_class_test_sent_txt)

    pass
