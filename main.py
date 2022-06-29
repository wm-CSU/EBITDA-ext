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
from torch.utils.data.sampler import WeightedRandomSampler

from S4_dataset import Data
# from S5_model import BertForClassification
# from S6_train import Trainer
from S5_model2 import BertForClassification
from S6_train2 import Trainer
from S8_predict import Prediction
from utils import load_torch_model, get_path, get_label_cooccurance_matrix

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']


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

    # 2. Build model
    model = BertForClassification(config)
    model.to(device)
    # # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
    # model.BaseBert.bert.resize_token_embeddings(len(changed_tokenizer))
    # changed_tokenizer.save_pretrained(config.model_path)

    datasets = data.load_train_and_valid_files(
        train_file=config.train_file, train_sheet=config.train_sheet, train_txt=config.train_txt,
    )
    train_set, valid_set_train = datasets

    # with WeightedRandomSampler
    target = train_set[:][-1]
    class_sample_counts = target.sum(axis=0, keepdims=False, dtype=torch.float)
    weights = 1. / class_sample_counts
    mid = np.array([(weights * t).sum(axis=0, dtype=torch.float) for t in target])
    samples_weights = np.where(mid == 0., 0.003, mid)
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights) // 2, replacement=True)
    data_loader = {
        'train': DataLoader(
            train_set, sampler=sampler, batch_size=config.batch_size, shuffle=False, drop_last=True),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False, drop_last=True),
    }

    # sampler_target = torch.zeros(config.num_classes)
    # noise_sample = 0
    # for batch in data_loader['train']:
    #     sample = [1 for i in batch[-1][:] if i.equal(torch.zeros(config.num_classes, dtype=torch.long))]
    #     noise_sample += sample.count(1)
    #     sampler_target.add_(batch[-1].sum(axis=0, keepdims=False, dtype=torch.int))

    config.labels_co_mat = torch.Tensor(get_label_cooccurance_matrix(train_set[:][-1])).to(device)

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
    #     train_result, valid_result = trainer.evaluate_train_valid()
    #     # print(train_result, valid_result)

    # 5. evaluate
    model = load_torch_model(
        model,
        model_path=os.path.join(config.model_path, config.experiment_name, config.model_type + '-best_model.bin'),
        multi_gpu=False
    )
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
    #                        # test_file='data/batch_two_for_test.xlsx', test_sheet='Sheet1',
    #                        # test_txt='data/adjust_txt/',
    #                        )
    pred_tool.evaluate_for_all(model=model, device=device,
                               to_file=config.prediction_file, to_sheet=config.prediction_sheet,
                               metrics_save_file=config.prediction_metrics_save_file,
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
    main(args.config_file, need_train=True, ReTrain=False)
    # fire.Fire(main)
