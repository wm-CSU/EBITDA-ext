"""Data processor for sentence classification.

In data file, each line contains (19+5)*3=72 attributes (XX, XX_sentence, XX_keywords) and other attributes.
The data processor convert each sentence into (sentence, class number) pair,
each sample with 1 sentence and 1 label.

Usage:
    from data import Data
    # For BERT model
    # For training, load train and valid set
    data = Data('model/bert/vocab.txt', model_type='bert')
    datasets = data.load_train_and_valid_files('adjust.csv', 'SMP-CAIL2021-valid.csv')
    train_set, valid_set_train, valid_set_valid = datasets
    # For testing, load test set
    data = Data('model/bert/vocab.txt', model_type='bert')
    test_set = data.load_file('SMP-CAIL2021-test.csv', train=False)
"""

import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from transformers import BertTokenizer
from tqdm import tqdm
from utils import read_annotation
from S3_sentence_division import Division


class Data:
    """Data processor for pretrained model for sentence classification.

    Attributes:
        model_type: 'bert'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
    """

    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512,
                 model_type: str = 'bert'):
        """Initialize data processor.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert'
        """
        # self.model_type = model_type
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def load_file(self,
                  filename: str = 'data/Nonrecurring Items_陈炫衣_20220226.xlsx',
                  sheet_name: str = 'Sheet1',
                  txt_path: str = 'data/sent_label/',
                  train: str = 'train'):
        """Load train file and construct TensorDataset.

        Args:
            file_path: train file
            sheet_name: sheet name
            txt_path:
                If True, txt with 'sentence \t label'
                Otherwise, txt with paragraph
            train:
                If True, train file with 'sentence \t label' in txt_path
                Otherwise, test file without label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                [torch.utils.data.TensorDataset]
                    each record: (input_ids, input_mask, segment_ids)
        """
        ori_data = read_annotation(filename=filename, sheet_name=sheet_name)
        sent_list, label_list = [], []
        if train == 'test':
            division = Division(ori_data, Train=False)
            # dataset = []
            for index, one in division.data.iterrows():
                # sent_list = []
                filename = os.path.join(txt_path, index + '.txt')
                sentences = division.txt2sent(filename=filename)
                for sent in sentences:
                    sent_list.append(self.tokenizer.tokenize(sent))

                # dataset.append(self._convert_sentence_to_bert_dataset(sent_list, label_list))
            dataset = self._convert_sentence_to_bert_dataset(sent_list, label_list)

        else:
            division = Division(ori_data)
            for index, one in division.data.iterrows():
                filename = os.path.join(txt_path, index + '.txt')
                f = open(filename, 'r', encoding='utf8')
                for line in f.readlines():
                    one = line.split('\t')
                    sent_list.append(self.tokenizer.tokenize(one[0]))
                    label_list.append(int(one[1]))
                    if int(one[1]) != 0:
                        for _ in range(9):
                            sent_list.append(self.tokenizer.tokenize(one[0]))
                            label_list.append(int(one[1]))

            dataset = self._convert_sentence_to_bert_dataset(sent_list, label_list)
            # if train == 'train':
            #     dataset = self._convert_sentence_to_bert_dataset(sent_list, label_list)
            # elif train == 'valid':
            #     dataset = self._convert_sentence_to_bert_dataset(sent_list, [])
            # else:
            #     label_list = []
            #     dataset = self._convert_sentence_to_bert_dataset([], [])
            #     print('mode error!')

        return dataset, label_list

    def load_train_and_valid_files(self,
                                   train_file, train_sheet, train_txt,
                                   valid_file, valid_sheet, valid_txt,
                                   ):
        """Load all files for task.

        Args:
            train_file, valid_file: files for sentence classification.

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        mydataset, data_labels = self.load_file(train_file, sheet_name=train_sheet, txt_path=train_txt, train='train')
        train_set, valid_set = self.dataset_split(mydataset)
        print(len(train_set), 'train records loaded.', len(valid_set), 'valid records loaded.')
        print('Loading valid records...')
        valid_set_valid, _ = self.load_file(valid_file, sheet_name=valid_sheet,
                                            txt_path=valid_txt, train='test')
        print(len(valid_set_valid), 'valid records loaded.')

        return train_set, valid_set, valid_set_valid

    def dataset_split(self, dataset, split_ratio=0.7):
        train, valid = random_split(
            dataset,
            lengths=[int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))],
            generator=torch.Generator().manual_seed(0))

        return train, valid

    def _convert_sentence_to_bert_dataset(
            self, sent_list, label_list=None):
        """Convert sentence-label to dataset for BERT model.

        Args:
            sent_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(sent_list), ncols=80):
            tokens = ['[CLS]'] + sent_list[i] + ['[SEP]']
            # segment_ids = [0] * len(tokens)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
                # segment_ids = segment_ids[:self.max_seq_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_mask += [0] * (self.max_seq_len - len(input_ids))
            input_ids += [0] * (self.max_seq_len - len(input_ids))  # 补齐剩余位置
            segment_ids = [0] * self.max_seq_len

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert-base-uncased/vocab.txt', model_type='bert', max_seq_len=200)
    train, _, _, = data.load_train_and_valid_files(
        train_file='data/matchtxt.xlsx', train_sheet='Sheet1', train_txt='data/sent_label/',
        valid_file='data/matchtxt.xlsx', valid_sheet='Sheet1', valid_txt='data/sent_label/',
    )


if __name__ == '__main__':
    test_data()
