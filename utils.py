"""Utils tools.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import logging
import pandas as pd
import numpy as np
import os
import re
import json
import shutil
from collections import OrderedDict, Counter
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
    file_handler = logging.FileHandler(log_file_name, mode='a')
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    if title:
        logger.info(title)
    return logger


def get_label_map(filename: str = r'data/label_map.json'):
    with open(filename, 'r') as f:
        label_map = json.load(f)
        f.close()

    return label_map


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
        print(new_file_abspath)

    except Exception as e:
        print('err', e)


def sent_process(para: str = None) -> str:
    para = re.sub(r'[*_+0-9]', '', para)
    words = para.split()  # 若改用 bert-base-uncased 模型，此处的 lower 应取消！！！
    need_del = []
    for index, word in enumerate(words):
        if not bool(re.search(r'[a-zA-Z]', word)):
            need_del.append(index)
    for i in sorted(need_del, reverse=True):
        del words[i]

    return ' '.join(words)


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


def get_label_cooccurance_matrix(labels, filename='result/co_mat.xlsx'):
    """
    get label Co-occurance matrix.
    :param labels: dataset[-1] type:Tensor
    :return:
    """
    # class_label_counts = labels.sum(axis=0, keepdims=False, dtype=torch.int)
    label_number = labels.shape[1]
    co_mat = np.zeros((label_number, label_number), dtype=float)
    co_mat_normalization = np.zeros((label_number, label_number), dtype=float)
    for one in labels:
        label = one.cpu().numpy().tolist()
        one_num = Counter(label)[1]
        if one_num > 1:
            index = [i for i, x in enumerate(label) if x == 1]
            for i in range(len(index)):
                for j in range(i, len(index)):
                    co_mat[index[i], index[j]] += 1 / one_num  # 得到的矩阵为上三角矩阵，最后记得复制到下三角即可
        elif one_num == 1:
            index = label.index(1)
            co_mat[index, index] += 1
        else:
            continue

    for a in range(label_number):
        for b in range(a + 1, label_number):
            if co_mat[a, b] < 0.5:  # 清除过低的共现情况
                co_mat[a, b] = 0
            co_mat[b, a] = co_mat[a, b]

    for a in range(label_number):
        co_matSum = np.sum(co_mat[:, a])
        co_mat_normalization[:, a] = co_mat[:, a] / co_matSum

    writer = pd.ExcelWriter(filename)
    co_mat = pd.DataFrame(co_mat, index=list(range(1, co_mat.shape[0] + 1)),
                          columns=list(range(1, co_mat.shape[1] + 1)))
    co_mat.to_excel(writer, '共现矩阵', float_format='%.3f')

    normal = pd.DataFrame(co_mat_normalization, index=list(range(1, co_mat_normalization.shape[0] + 1)),
                          columns=list(range(1, co_mat_normalization.shape[1] + 1)))
    normal.to_excel(writer, '归一化结果', float_format='%.3f')
    writer.save()
    writer.close()

    return co_mat_normalization


def get_sampler(train_set):
    from torch.utils.data.sampler import WeightedRandomSampler
    # # with WeightedRandomSampler
    target = train_set[:][-1]
    class_sample_counts = target.sum(axis=0, keepdims=False, dtype=torch.float)
    weights = 1. / class_sample_counts
    mid = np.array([(weights * t).sum(axis=0, dtype=torch.float) for t in target])

    print('采样前样例情况: ', (np.sum(mid == 0.), class_sample_counts))

    samples_weights = np.where(mid == 0., 0.003, mid)
    samples_weights = np.where(samples_weights > 0.1, 0.1, samples_weights)
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights) // 4, replacement=True)

    return sampler


def static_loader(dataloader, num_classes):
    sampler_target = torch.zeros(num_classes)
    noise_sample = 0
    for batch in dataloader:
        sample = [1 for i in batch[-1][:] if i.equal(torch.zeros(num_classes, dtype=torch.long))]
        noise_sample += sample.count(1)
        sampler_target.add_(batch[-1].sum(axis=0, keepdims=False, dtype=torch.int))

    return noise_sample, sampler_target
