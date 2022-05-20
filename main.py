"""Test model for SMP-CAIL2021-ArgumentationUnderstanding Task1.1.

Author: Yixu GAO yxgao19@fudan.edu.cn

Usage:
    python main.py --model_config 'config/bert_config.json' \
                   --in_file 'data/SMP-CAIL2021-test1.csv' \
                   --out_file 'bert-submission-test-1.csv'
"""

import json
import os
from types import SimpleNamespace

# import fire
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader

from S4_dataset import Data
from S5_model import BertForClassification
from S6_train import Trainer
from S7_evaluate import evaluate, evaluate_main
from S8_predict import Predict_postprocess
from utils import load_torch_model, get_path

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']


def main(config_file='config/bert_config.json', need_train: bool = False):
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
                max_seq_len=config.max_seq_len,
                model_type=config.model_type)
    datasets = data.load_train_and_valid_files(
        train_file='data/merge_data.xlsx', train_sheet='Sheet1', train_txt='data/sent_label/',
        valid_file='data/test.xlsx', valid_sheet='Sheet1', valid_txt='data/test_adjust_txt/',
    )
    train_set, valid_set_train, valid_set_valid = datasets

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('cuda is available!')
    else:
        device = torch.device('cpu')

    data_loader = {
        'train': DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
        'valid_valid': DataLoader(
            valid_set_valid, batch_size=config.batch_size, shuffle=False)
    }

    # 2. Build model
    model = BertForClassification(config)
    model.to(device)

    if need_train:
        # 3. Train
        trainer = Trainer(model=model, data_loader=data_loader,
                          device=device, config=config)
        best_model_state_dict = trainer.train(ReTrain=True)
        # 4. Save model
        torch.save(best_model_state_dict,
                   os.path.join(config.model_path, 'model.bin'))
    else:
        # 5. evaluate only
        model = load_torch_model(
            model,
            model_path=os.path.join(config.model_path, config.experiment_name, config.model_type + '-best_model.bin'),
            multi_gpu=False
        )

        valid_predictions, valid_sent = evaluate(data.tokenizer, model, data_loader['valid_valid'], device)
        # print(valid_predictions)
        pred = Predict_postprocess(test_file='data/test.xlsx', test_sheet='Sheet1',
                                   test_txt='data/test_adjust_txt/')
        pred.deal(
            sent_list=valid_sent, predictions=valid_predictions,
            to_file='data/predict.xlsx', to_sheet='Sheet1'
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()

    main(args.config_file, need_train=False)
    # fire.Fire(main)
