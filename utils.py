"""Utils tools.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import logging
import pandas as pd
import os
import re
import shutil
from collections import OrderedDict
import torch


def get_path(path):
    """Create the path if it does not exist.

    Args:
        path: path to be used

    Returns:
        Existed path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_csv_logger(log_file_name,
                   title='',
                   log_format='%(message)s',
                   log_level=logging.INFO):
    """Get csv logger.

    Args:
        log_file_name: file name
        title: first line in file
        log_format: default: '%(message)s'
        log_level: default: logging.INFO

    Returns:
        csv logger
    """
    logger = logging.getLogger(log_file_name)
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file_name, 'w+')
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    if title:
        logger.info(title)
    return logger


def read_annotation(filename, sheet_name):
    """Read annotation file.

    Args:
        filename:
        sheet_name:

    Returns:
        df_NI: dataframe
    """
    df_NI = pd.read_excel(filename, sheet_name=sheet_name)
    df_NI['txt_id'] = df_NI['file_name']
    df_NI.set_index('txt_id', inplace=True)
    # df_NI.head()

    return df_NI


def move_txt(soure_file, dirname):
    """move txt file，eg: xxx(6).txt

    Args:
        soure_file：
        dirname: new dir (folder)
    """
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
            file_name = soure_file.split('/')[-1]
        else:
            file_name = soure_file.split('/')[-1]
        # 当文件夹不存在 创建文件夹
        get_path(dirname)
        # 若源文件不存在则报错返回
        assert os.path.exists(soure_file) or os.path.isfile(soure_file), '源文件不存在或不是文件'
        # 文件移动
        if not os.path.exists(os.path.join(dirname, file_name)):
            shutil.move(soure_file, dirname)
            return

        ref1 = [x for x in os.listdir(dirname) if x.find(file_name.replace('%s' % file_suffix, '')) != -1]
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
            shutil.move(soure_file, new_file_abspath)
        print(soure_file)

    except Exception as e:
        print('err', e)


def load_torch_model(model, model_path,
                     multi_gpu: bool = False):
    """Load state dict to model.

    Args:
        model: model to be loaded
        model_path: state dict file path
        multi_gpu: Use multiple GPUs or not

    Returns:
        loaded model
    """
    pretrained_model_dict = torch.load(model_path)

    if multi_gpu:
        new_state_dict = OrderedDict()
        for k, value in pretrained_model_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(pretrained_model_dict, strict=True)

    return model


def find_lcsubstr(s1, s2):
    # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度
