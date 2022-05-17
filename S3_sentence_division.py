# coding: utf-8
"""Paragraphs are divided into short sentences for subsequent classification.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import json
import os.path
import re
import pandas as pd
from utils import get_path, read_annotation
from S1_preprocess import Drop_Redundance
from S2_EBITDA_locate import Paragraph_Extract


class Division:
    def __init__(self, data):
        self.data = Drop_Redundance(data)

    # def previous_deal(self, txtset_path: str = r'data/txt_set/',
    #                   ebitda_txt_path: str = r'data/adjust_txt/'):
    #     ext = Paragraph_Extract(self.ori_data)
    #     self.data = ext.deal(input_path=txtset_path, output_path=ebitda_txt_path)
    #     return self.data

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

        return labels

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
        para2sent = re.split(';|\.|\([\s\S]{1,3}\)', paragraph.strip())
        return [one for one in para2sent if len(one) > 10]

    def sent_label(self, one_data, one_sent, label_map):
        # 一条数据的sentence与label对应
        label = {}
        for sent in one_sent:
            label[sent.strip()] = 0

        for (name, value) in one_data.items():
            if pd.isna(value):
                continue
            elif 'sentence' in name:
                for sent in one_sent:
                    if value in sent.strip() or sent.strip() in value:
                        label[sent.strip()] = label_map[name]
                        # 一键多值,处理重复标注
                        # if label[sent.strip()]==0:
                        #     label[sent.strip()] = label_map[name]
                        # else:
                        #     label[sent.strip()] = [label[sent.strip()], label_map[name]]
                    else:
                        continue
            else:
                continue

        return label

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map

    def sent2txt(self, sent):
        # 标准答案句换进分句结果中，解决过度划分问题
        # 最后再解决叭
        pass

    def label2txt(self, label_dict, filename):
        with open(filename, 'w', encoding='utf8') as f:
            for sent, label in label_dict.items():
                f.write(sent + '\t' + str(label) + '\n')
            f.close()

        return


if __name__ == '__main__':
    ori_data = read_annotation(filename=r'data/matchtxt.xlsx', sheet_name='Sheet1')

    txt_set = r'data/txt_set/'
    ebitda_txt = r'data/adjust_txt/'
    ext = Paragraph_Extract(ori_data)
    data = ext.deal(input_path=txt_set, output_path=ebitda_txt)

    sent_txt = r'data/sent_label'
    division = Division(ori_data)
    division.deal(label_map=r'data/label_map.json', txtfilepath=ebitda_txt, labelfilepath=sent_txt)
    pass