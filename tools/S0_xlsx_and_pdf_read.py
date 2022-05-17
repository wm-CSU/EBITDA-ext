#!/usr/bin/env python
# coding: utf-8
# Func: File Operations.

import pdfplumber
import os
import pandas as pd
import re
import shutil


def read_annotation(filename, sheet_name):
    df_NI = pd.read_excel(filename, sheet_name=sheet_name)
    df_NI['facstartdate'] = df_NI['facstartdate'].dt.strftime('%Y%m%d')
    # df_NI['pdf_id'] = df_NI['gvkey'].map(str) + '_' + df_NI['facstartdate']
    # df_NI.set_index('pdf_id', inplace=True)
    df_NI['txt_id'] = df_NI['file_name']
    df_NI.set_index('txt_id', inplace=True)
    df_NI.head()

    return df_NI


def get_all_pdf(filepath, data):
    # 获取文件夹下所有pdf名称
    file_list = os.listdir(filepath)
    print(file_list)
    lst_name = []
    for item in filepath:
        if item.endswith('.pdf'):
            lst_name.append(filepath + item)
            # pdf2txt(filepath + item, )
        else:
            continue

    return lst_name


def read_pdf(pdf_name):
    # for item in lst_name:
    # pdf = pdfplumber.open(pdf_name)
    with pdfplumber.open(pdf_name) as pdf:
        # print(len(pdf.pages))
        for page in pdf.pages:
            text = page.extract_text()
            paragraph = text.split('\n')
            print(paragraph)
            break
    pass


def move_txt(soure_file_abspath, dirname):
    ''' 移动文件，例：xxx(6).txt
        soure_file_abspath：源文件绝对路径
        dirname ： 目标文件夹或者目标文件绝对路径
    '''
    file_suffix = '.txt'
    # 判断系统
    # if platform.system().find('Windows') != -1:
    #     re_str = '\\'
    # else:
    #     re_str = '/'
    try:
        # 处理传入文件或者文件夹
        # assert os.path.isfile(dirname) or os.path.isdir(dirname), '请填写正确路径'
        if os.path.isfile(dirname):
            dirname, file_name = os.path.split(dirname)
        elif os.path.isdir(dirname):
            file_name = soure_file_abspath.split('/')[-1]
        else:
            file_name = soure_file_abspath.split('/')[-1]
        # 当文件夹不存在是创建文件夹
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        assert os.path.exists(soure_file_abspath) or os.path.isfile(soure_file_abspath), '源文件不存在或不是文件'
        # 文件移动
        if not os.path.exists(os.path.join(dirname, file_name)):
            shutil.move(soure_file_abspath, dirname)
            return

        ref1 = [x for x in os.listdir(dirname) if x.find(file_name.replace('%s' % file_suffix, ''))!=-1]
        # 正则用于，自定义文件名
        ref_out = [int(re.findall('\((\d+)\)%s' % file_suffix, x)[0]) for x in ref1 if
                   re.findall('\((\d+)\)%s' % file_suffix, x)]
        # 当文件名重复时处理
        if not ref_out:
            new_file_abspath = os.path.join(dirname, ('(1)%s' % file_suffix).join(
                file_name.split('%s' % file_suffix)))
        else:
            new_file_abspath = os.path.join(dirname, ('(%s)%s' % ((max(ref_out) + 1), file_suffix)).join(
                file_name.split('%s' % file_suffix)))
            shutil.move(soure_file_abspath, new_file_abspath)

    except Exception as e:
        print('err', e)


if __name__ == '__main__':
    filepath = r'../data/陈炫衣agreement/'
    filename = r'data/Nonrecurring Items_陈炫衣_20220226.xlsx'
    # read_pdf('data/陈炫衣agreement/160684_20110916.pdf')
    # move_txt(r'data/txt/160684_20110916.txt', r'data/txt_set')