"""Test model for multi label classification.

Author: wangmin0918@csu.edu.cn

"""

import json
import os
from types import SimpleNamespace

# import fire
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from S4_dataset import Data, TestData
from S5_model import BertForClassification
from S6_train import Trainer
# from S5_model2 import BertForClassification
# from S6_train2 import Trainer
from S8_predict import Prediction
from utils import load_torch_model, get_path, get_label_cooccurance_matrix, get_sampler, static_loader

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          '10', '11', '12', '13', '14', '15', '16', '17', '18']
# TRAIN_NAMES = ['train', '蔡子悦', '高尚', '李佳慧', '马成辰', '宋瑶', '童婧曦', '王珏']
TRAIN_NAMES = ['train', 'caiziyue', 'gaoshang', 'lijiahui', 'machengchen', 'songyao', 'tongjingxi', 'wangjue']


def main(config_file='config/bert_config.json',
         need_train: bool = False,
         ReTrain: bool = False):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)

    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len)
    # changed_tokenizer = data.for_add_tokens(filename=config.train_file,
    #                                         sheet_name=config.train_sheet, txt_path=config.train_txt)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('cuda is available!')
    else:
        device = torch.device('cpu')

    # # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
    # model.BaseBert.bert.resize_token_embeddings(len(changed_tokenizer))
    # changed_tokenizer.save_pretrained(config.model_path)

    datasets = data.load_train_and_valid_files(
        train_list=TRAIN_NAMES, train_sheet=config.train_sheet, train_txt=config.train_txt, split_ratio=0.9
    )
    train_set, valid_set_train, train_b2set = datasets

    label_set = train_set[:][-1].sum(axis=1, keepdims=False, dtype=torch.float)
    print('训练集  无标签句子: {}; 有标签句子: {}'.format(sum(i == 0 for i in label_set), sum(i > 0 for i in label_set)))
    valid_label_set = valid_set_train[:][-1].sum(axis=1, keepdims=False, dtype=torch.float)
    print('验证集  无标签句子: {}; 有标签句子: {}'.format(sum(i == 0 for i in valid_label_set), sum(i > 0 for i in valid_label_set)))
    sampler = get_sampler(train_set)
    data_loader = {
        'train': DataLoader(
            train_set,
            sampler=sampler,
            batch_size=config.batch_size, shuffle=False, drop_last=True),
        'train_b2': DataLoader(
            train_b2set, batch_size=config.batch_size, shuffle=False, drop_last=True),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False, drop_last=True),
    }
    print('采样后样例情况：', static_loader(data_loader['train'], config.num_classes))

    config.labels_co_mat = torch.Tensor(
        get_label_cooccurance_matrix(train_set[:][-1], filename='result/co_mat-train.xls')).to(device)
    # # # 导出测试集共现统计矩阵
    # testdata = TestData(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
    #                     max_seq_len=config.max_seq_len)
    # test_set, test_label = testdata.load_file(
    #     filename=config.test_file, sheet_name=config.test_sheet, txt_path=config.test_txt
    # )
    # get_label_cooccurance_matrix(test_set[:][-1], filename='result/co_mat-test.xls')

    # 2. Build model
    model = BertForClassification(config)
    model.to(device)

    if need_train:

        # 3. Train
        trainer = Trainer(model=model, data_loader=data_loader,
                          device=device, config=config)
        best_model_state_dict = trainer.train(ReTrain=ReTrain)
        # 4. Save model
        torch.save(best_model_state_dict,
                   os.path.join(config.model_path, 'model.bin'))
    # else:
    #     # 3. Valid
    #     trainer = Trainer(model=model, data_loader=data_loader,
    #                       device=device, config=config)
    #     trainer.model, _, _, _ = trainer.load_last_model(
    #         model=model,
    #         model_path=os.path.join(config.model_path, config.experiment_name, config.model_type + '-last_model.bin'),
    #         optimizer=trainer.optimizer,
    #         multi_gpu=False
    #     )
    #     _ = trainer.evaluate_train_valid()
    #     # print(train_result, valid_result)

    # 5. evaluate
    model = load_torch_model(
        model,
        model_path=os.path.join(config.model_path, config.experiment_name, config.model_type + '-best_model.bin'),
        multi_gpu=False
    )
    # # # print(model.comat_multi.params)
    # import pandas as pd
    # writer = pd.ExcelWriter('result/'+config.experiment_name + '-trained_co_mat.xls')
    # trained_comat = model.comat_multi.params.detach().cpu().numpy()
    # co_mat = pd.DataFrame(trained_comat, index=list(range(1, trained_comat.shape[0] + 1)),
    #                       columns=list(range(1, trained_comat.shape[1] + 1)))
    # co_mat.to_excel(writer, 'trained-共现矩阵')
    # writer.save()
    # writer.close()

    from S8_predict import PredictionWithlabels
    pred_tool = PredictionWithlabels(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                                     max_seq_len=config.max_seq_len,
                                     test_file=config.test_file,
                                     test_sheet=config.test_sheet,
                                     test_txt=config.test_txt,
                                     )
    # pred_tool = Prediction(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
    #                        max_seq_len=config.max_seq_len,
    #                        test_file=config.test_file,
    #                        test_sheet=config.test_sheet,
    #                        test_txt=config.test_txt,
    #                        )
    pred_tool.evaluate_for_all(model=model, device=device,
                               to_file='result/prediction-for yl-' + config.experiment_name,
                               to_sheet=config.prediction_sheet,
                               metrics_save_path='result/' + config.experiment_name,
                               sigmoid_threshold=config.sigmoid_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()
    main(args.config_file, need_train=True, ReTrain=True)
    # fire.Fire(main)
