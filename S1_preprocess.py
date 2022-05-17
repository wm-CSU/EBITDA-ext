# coding: utf-8
"""Data preprocessing.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os.path

from utils import read_annotation, move_txt


def Drop_Redundance(data,
                    excel_path: str = r'data/new.xlsx',
                    sheet_name: str = 'Sheet1'
                    ):
    '''
    包括：调整sentence keywords名称；未标注数据删除（没有文件）；重复数据删除（相同文件）；
    :param data:
    :return: data(dealed)
    '''
    # print(data.columns, data.index)
    # 调整列名
    for i in range(len(data.columns.values)):
        if 'sentence' in data.columns[i]:
            data.rename(columns={data.columns[i]: data.columns[i - 1] + '_sentence'}, inplace=True)
        if 'keyword' in data.columns[i]:
            data.rename(columns={data.columns[i]: data.columns[i - 2] + '_keywords'}, inplace=True)
    # 删除汇总列和无用列，避免影响句子分类
    drop_name = ['exclude_nonrec', 'exclude_nonrec_sentence', 'exclude_nonrec_keywords',
                 'exclude_nonrec_loss', 'exclude_nonrec_loss_sentence', 'exclude_nonrec_loss_keywords',
                 'exclude_nonrec_gain', 'exclude_nonrec_gain_sentence', 'exclude_nonrec_gain_keywords',
                 'other_nonrec_loss', 'other_nonrec_loss_sentence', 'other_nonrec_loss_keywords',
                 'other_nonrec_gain', 'other_nonrec_gain_sentence', 'other_nonrec_gain_keywords',
                 'secdate', 'secexhib', 'secform', 'ticker', 'facid', 'packageid',
                 'fcovenant_c', 'company', 'maturity', 'loantype', 'facilityamt', 'bcoid',
                 'primarypurpose', 'secured', 'collateral', 'performance', 'rating', 'conm',
                 'title', 'url', '备注']
    data = data.drop(drop_name, axis=1)
    # 未标注数据删除
    data.dropna(axis=0, thresh=19, inplace=True)  # 57/3=19 至少应有24个元素非空
    # 重复数据删除(默认保留第一条)
    data.dropna(axis=0, subset=['file_name'], inplace=True)
    data.drop_duplicates(subset=['file_name'], inplace=True)

    data.to_excel(excel_path, sheet_name=sheet_name)

    return data


def batch_move(data, ori_path, new_path):
    for index, row in data.iterrows():
        ori_file = os.path.join(ori_path, index.split('_')[0], 'txt', index + '.txt')
        print(ori_file)
        move_txt(ori_file, new_path)

    return


if __name__ == '__main__':
    ori_data = read_annotation(filename=r'data/matchtxt.xlsx', sheet_name='Sheet1')
    data = Drop_Redundance(ori_data)
    data.to_csv('data/new.csv')
    batch_move(data, 'data/ori_txt', 'data/txt_set')
