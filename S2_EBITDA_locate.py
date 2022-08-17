# coding: utf-8
"""Regular expression positioning EBITDA.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import time
import re
import os
import numpy as np
import pandas as pd

from utils import get_path, read_annotation, sent_process
from S1_preprocess import Drop_Redundance


class pdf_Paragraph_Extract:

    def __init__(self, ori_data):
        self.data = Drop_Redundance(ori_data)  # 原生数据预处理（冗余删除）

    def deal(self, input_path, output_path,
             excel_path: str = r'data/matchtxt.xlsx',
             sheet_name: str = 'new'):
        get_path(output_path)
        import pdfplumber
        for index, row in self.data.iterrows():
            filename = input_path + str(index) + '.pdf'
            out_filename = output_path + str(index) + '.txt'
            if os.path.isfile(filename):
                pdf = pdfplumber.open(filename)
                _, NI_exist = self.goal_locate(pdf, out_filename)
                row['nest'] = 1 if NI_exist else 0
                print("{} is dealed.".format(filename))
            else:
                print("ERROR! File is not accessible.")

        import openpyxl
        with pd.ExcelWriter(excel_path, engin='openpyxl') as writer:
            # self.data.to_excel(writer, sheet_name='Sheet3')
            writer.book = openpyxl.load_workbook(writer.path)
            self.data.to_excel(excel_writer=writer, sheet_name=sheet_name)
            writer.save()

        return

    def goal_locate(self, pdf, txtpath):
        '''
        单个 pdf 抽取 EBITDA/Net Income 定义段落
        :param pdf: 传入的 pdf 对象
        :return:
        goal: [str] 目标段落列表
        NI_exist: bool 是否嵌套
        '''
        # 构造正则表达式的pattern
        ebitda = re.compile(r"EBITDA[\s\S]{0,40}(?:mean|:)")
        net_income_exist = re.compile(r"Net Income")
        net_income_place = re.compile(r"Net Income[\s\S]{0,40}(?:mean|:)")
        # 逐页遍历找到段落
        Page_cohesion = False
        NI_exist = False  # 该参数表示 EBITDA 中是否嵌套使用了 Net Income
        cont = False  # 判定冒号后续是否追加的参数
        goal = []
        for page in pdf.pages:
            text = page.extract_text(layout=True, x_density=7.25, y_density=13)  # 得到整页文本  str类型
            paragraph = self.text2paragraph(text)
            if not paragraph:  # 列表为空直接跳过
                continue

            if Page_cohesion and paragraph:  # 前后页衔接操作
                goal = [previous_end + ' ' + paragraph[0].strip() if str(previous_end) in i else i for i in goal]

            # 判断 该页的文本 是否为 EBITDA定义/Net Income定义
            for para in range(len(paragraph)):
                if cont:
                    for add in range(0, len(paragraph) - para):  # 以（开头加入
                        if paragraph[para + add].strip().startswith('('):
                            goal.append(paragraph[para + add].strip())
                    cont = False
                    continue

                if ebitda.search(paragraph[para]):  # 找到EBITDA段落
                    goal.append(paragraph[para])
                    if paragraph[para].endswith(':'):  # 判定以冒号结尾则向下继续
                        cont = True
                    if net_income_exist.search(paragraph[para]):  # 若嵌套存在性判定通过，则置True
                        NI_exist = True
                elif net_income_place.search(paragraph[para]):  # 非 EBITDA 段落 且 找到 Net Income 定义所在段落
                    goal.append(paragraph[para])
                    if paragraph[para].endswith(':'):  # 判定以冒号结尾则向下继续
                        cont = True
                else:
                    continue

            previous_end = paragraph[-1]  # 衔接前后页
            if any(paragraph[-1] in item for item in goal):  # 若最后一段选入goal，则应当衔接
                cont, Page_cohesion = True, True

        # print(goal)
        self.to_txt(goal, txtpath)
        return goal, NI_exist

    def text2paragraph(self, text):
        '''
        pdf单页 文本转段落 （策略：用空格分段  若空格数较上一行变多则是新段落的开始）
        :param text: str
        :return: paragraph: [str]
        '''
        lines = text.split('\n')  # 以/n化为list
        lines = [x for x in lines if x != '']  # 除去多余空串
        del lines[0], lines[-1]  # 除去页眉页脚
        blank_match = re.compile('\s*')
        base = self.baseblank_calculate(blank_match, lines)
        # 下面遍历整页文本
        paragraph = []
        midstr = ''
        for para in lines:
            blank = blank_match.match(para)
            if blank:
                number = blank.end() - blank.start()  # 计算空格数
                if number >= base + 3:  # 判定属于段落开头，需将上一轮的str加入段落列表，str重新赋值
                    paragraph.append(midstr)
                    midstr = para
                else:  # 判定属于段落的后续行，原基础上加
                    midstr += ' '
                    midstr += para.strip()
            else:  # 若不以空格开始则直接加进段落中
                midstr += ' '
                midstr += para
        paragraph.append(midstr)  # 补充最后一段
        return paragraph

    def baseblank_calculate(self, match, text):
        # 计算基本空格数(整页统计)   除去页眉页尾
        number = [100, ]

        for i in range(1, len(text) - 1):
            base_blank = match.match(text[i])
            if base_blank:
                blank_number = base_blank.end() - base_blank.start()
                number.append(blank_number)

        blank_number = min(number)
        return blank_number

    def to_txt(self, text, filepath):
        with open(filepath, 'w', encoding='utf8') as f:
            for line in text:
                f.write(line)
        f.close()


class Paragraph_Extract:
    def __init__(self, ori_data, Train: bool = True):
        if Train:
            self.data = Drop_Redundance(ori_data)  # 原生数据预处理（冗余删除）
        else:
            self.data = ori_data

    def deal(self, input_path, output_path):
        get_path(output_path)
        for index, row in self.data.iterrows():
            filename = os.path.join(input_path, str(index) + '.txt')
            out_filename = os.path.join(output_path, str(index) + '.txt')
            if os.path.isfile(filename):
                goal_para, nest = self.goal_locate(filename)
                self.to_txt(goal_para, out_filename)
                # row['nest'] = 1 if nest else 0
                print("EBITDA locate:  {} is dealed.".format(filename))
            else:
                print("~~~~~~ERROR!!! File {} is not accessible.".format(filename))

        return self.data

    def goal_locate(self, filename):
        '''
        单个 ori_txt 抽取 EBITDA/Net Income 定义段落
        :param
            filename:
        :return:
            goal: [str] 目标段落列表
            NI_exist: bool 是否嵌套
        '''
        f = open(filename, 'r', encoding='utf8')
        ori_txt = f.readlines()
        paragraph = self.text2paragraph(ori_txt)
        # 构造正则表达式的pattern
        ebitda = re.compile(r'^[\s\S]{0,200}(EBIT| EBT|Special Charges)[\s\S]{0,50}?(?:mean|:|\.|—)')
        net_income_place = re.compile(r'^[\s\S]{0,200}Net Income[\s\S]{0,50}?(?:mean|:|\.)')

        # 逐段遍历找到段落
        EBITDA_exist, NI_exist, cont = False, False, False  # 判定 EBITDA 中是否嵌套使用了 Net Income, 冒号后续是否追加
        goal = []
        # 判断 该页的文本 是否为 EBITDA定义/Net Income定义
        for index, para in enumerate(paragraph):
            if cont:
                for add in range(0, len(paragraph) - index):  # 以（开头加入
                    if paragraph[index + add].strip().startswith('('):
                        goal.append(paragraph[index + add].strip())
                    else:
                        break
                cont = False
            else:
                if ebitda.search(para):  # 找到EBITDA段落
                    goal.append(para)
                    if para.endswith(':'):  # 判定以冒号结尾则向下继续
                        cont = True
                    EBITDA_exist = True
                elif net_income_place.search(para):  # 非 EBITDA 段落 且 找到 Net Income 定义所在段落
                    goal.append(para)
                    NI_exist = True
                    if para.endswith(':'):  # 判定以冒号结尾则向下继续
                        cont = True
                else:
                    continue

        if EBITDA_exist and NI_exist:
            nest = True
        else:
            nest = False

        return goal, nest

    def text2paragraph(self, text):
        '''
        txt 文本转段落 （策略：空行分段  空行是段落的划分）
        :param text: [str]
        :return: paragraph: [str]
        '''
        paralist = []
        if len(text) < 100:
            paralist = self.post_processing([x for x in text if x.strip() != ''])
            return paralist
        Page_continue, Segment = False, False
        para = ''
        punctuation = re.compile(r'.*[.!?]$')
        page_number = re.compile(r'^(.*?[0-9]+)$|^(.*?- [0-9]+ -)$|^(.*?-[0-9]+-)$|^(.*?Page [0-9]+)$')
        text = [line for line in text if not page_number.match(line.strip())]
        for line in text:
            if line.strip() == '':
                Segment = True
            else:
                if Segment and not Page_continue:
                    paralist.append(para.replace('\xa0', ' '))
                    para = line.strip()
                    Segment = False
                else:
                    para = ' '.join([para, line.strip()])
                    Segment = False

            Page_continue = False if punctuation.match(para) else True

        paralist.append(para)

        if len(paralist) < 350 or sum(len(i.split()) > 800 for i in paralist) > 30:
            paralist = self.post_processing(paralist)

        return paralist

    def post_processing(self, paragraph):
        '''
            txt 文本转段落的后处理 （策略：统计，过长段落按 “**”mean 截取）
        :param paragraph: [str]
        :return: paragraph: [str]
        '''
        # 删去过短的元素
        paragraph = [x for x in paragraph if len(x.strip()) >= 10]
        post_para = []
        for para in paragraph:
            # res = re.split(r'(?<=\.)([\s]*?“.+?”.*?(?:mean|:)+?.*?\.[\s]*?)(?=“)', para)
            res = re.split(r'(?<=(\.|:))([\s]*?[0-9]*?[\s]*?“.+?”.*?(?<!(No|no))\.[\s]*?[0-9]*?[\s]*?)(?=“)', para)
            res = [i for i in res if i]
            if res:
                post_para.extend(res)
            else:
                # para = sent_process(para)
                res3 = re.split(r'(?<=\.)([\s]*?.*?(?<!(No|no))\.[\s]*)', para)
                post_para.extend([i for i in res3 if i])

        post_para = [i for i in post_para if 10 <= len(i) < 100000]

        if len(post_para) < 350:
            post_para = self.twice_post_processing(post_para)

        return post_para

    def twice_post_processing(self, paragraph):
        # 删去过短的元素
        post_para = []
        for para in paragraph:
            if len(para.split()) <= 800:
                post_para.append(para)
                continue
            res = re.split(r'(?<=(\.))([\s]*?.*?(?:mean|:).*?(?<!(No|no))\.+?[\s]*?)', para)
            res = [i for i in res if i]
            if res:
                post_para.extend(res)
            else:
                res3 = re.split(r'(?<=\.)([\s]*?.*?(?<!(No|no))\.[\s]*?)', para)
                post_para.extend([i for i in res3 if i])

        post_para = [i for i in post_para if 10 <= len(i) < 100000]

        return post_para

    def to_txt(self, text, filepath):
        with open(filepath, 'w', encoding='utf8') as f:
            for line in text:
                f.write(line + '\n')
        f.close()


if __name__ == '__main__':
    # ori_data = read_annotation(filename=r'data/todeal_2.xlsx', sheet_name='Sheet1')
    # ext = Paragraph_Extract(ori_data, Train=False)
    # start = time.time()
    # print('start time: ', time.strftime('%Y-%m-%d %H:%M:%S'))
    # ext.deal(input_path=r'data/todeal_2/', output_path=r'data/dealed_txt/A_2/')
    # end = time.time()
    # print('end time: ', time.strftime('%Y-%m-%d %H:%M:%S'))
    # print('use time: {} s'.format(end - start))
    # # todeal_1  start time:  2022-08-16 15:29:57;  end time:  2022-08-16 16:40:25;  use time: 4227.419719457626 s
    # # todeal_2  start time:  2022-08-16 16:44:35;  end time:  2022-08-16 17:41:06;  use time: 3390.262848138809 s
    # # todeal_3  start time:  2022-08-16 12:01:36;  end time:  2022-08-16 13:29:17;  use time: 5261.6615421772 s

    TRAIN_NAMES = ['蔡子悦', '高尚', '李佳慧', '马成辰', '宋瑶', '童婧曦', '王珏']
    for name in TRAIN_NAMES:
        filename = 'data/train_file/' + name + '.xlsx'
        ori_data3 = read_annotation(filename=filename, sheet_name='Sheet1')
        ext3 = Paragraph_Extract(ori_data3, Train=False)
        start = time.time()
        ext3.deal(input_path=os.path.join('data/train_txt/', name),
                  output_path=os.path.join(r'data/train_adjust_txt/', name))
        end = time.time()
        print('{} use time: {} s'.format(name, end - start))
    # one file debug
    # filename = 'data/test_txt/1408075_140807512000016_2.txt'
    # out_filename = 'data/test_adjust_txt/1408075_140807512000016_2.txt'
    # if os.path.isfile(filename):
    #     goal_para, nest = ext3.goal_locate(filename)
    #     ext3.to_txt(goal_para, out_filename)
    #     print("{} is dealed.".format(filename))
    # else:
    #     print("ERROR! File {} is not accessible.".format(filename))

    pass
