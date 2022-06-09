# coding: utf-8
"""Data preprocessing.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os.path

from utils import read_annotation, move_txt


def Drop_Redundance(data,
                    excel_path: str = r'data/new.xlsx',
                    sheet_name: str = 'Sheet1',
                    Train: bool = True
                    ):
    '''
    包括：调整sentence keywords名称；未标注数据删除（没有文件）；重复数据删除（相同文件）；
    :param data:
    :return: data(dealed)
    '''
    # print(data.columns, data.index)
    # 调整列名
    for i in range(len(data.columns.values)):
        if 'sentence' in data.columns[i] or 'Sentence' in data.columns[i]:
            data.rename(columns={data.columns[i]: data.columns[i - 1] + '_sentence'}, inplace=True)
        if 'keyword' in data.columns[i]:
            data.rename(columns={data.columns[i]: data.columns[i - 2] + '_keywords'}, inplace=True)
    # 保留有用列
    use_col = ['file_name',
        'asset_writedown', 'asset_writedown_sentence',
        'asset_disp_loss', 'asset_disp_loss_sentence',
        'restruct_exp', 'restruct_exp_sentence',
        'liti_settelment_loss', 'liti_settelment_loss_sentence',
        'goodwill_impairment', 'goodwill_impairment_sentence',
        'acquisition_charge', 'acquisition_charge_sentence',
        'debt_extinguish_loss', 'debt_extinguish_loss_sentence',
        'disconti_loss', 'disconti_loss_sentence',
        'relocation_exp', 'relocation_exp_sentence',
        'in_process_rnd_charge', 'in_process_rnd_charge_sentence',
        'acct_change_loss', 'acct_change_loss_sentence',
        'asset_disp_gain', 'asset_disp_gain_sentence',
        'asset_writeup', 'asset_writeup_sentence',
        'insur_proceed_gain', 'insur_proceed_gain_sentence',
        'restruct_inc', 'restruct_inc_sentence',
        'disconti_gain', 'disconti_gain_sentence',
        'debt_extinguish_gain', 'debt_extinguish_gain_sentence',
        'liti_settelment_gain', 'liti_settelment_gain_sentence',
        'acct_change_gain', 'acct_change_gain_sentence',
    ]
    data = data[use_col]
    if Train:
        # 未标注数据删除
        data.dropna(axis=0, thresh=19, inplace=True)  # 57/3=19 至少应有19个元素非空
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
    #
    # ori_data = read_annotation(filename=r'data/batch_one.xlsx', sheet_name='Sheet1')
    # data = Drop_Redundance(ori_data, 'data/new.xlsx')
    # batch_move(data, 'data/ori_txt', 'data/txt_set')
    #
    # # 第二批数据处理
    # ori_data2 = read_annotation(filename=r'data/batch_two.xlsx', sheet_name='Sheet1')
    # data2 = Drop_Redundance(ori_data2, 'data/new2.xlsx')
    # batch_move(data2, 'data/ori_txt', 'data/txt_set')
    #
    # import pandas as pd
    # merge_data = pd.concat([data, data2])
    # merge_data.to_excel('data/merge_data.xlsx')

    # 测试数据处理
    ori_data = read_annotation(filename=r'data/test_yqh.xlsx', sheet_name='Sheet1')
    batch_move(ori_data, r'E:/mycode/_DataSets/会计项目/郁绮虹', r'E:/mycode/EBITDA-Ext/data/test_txt_set')
