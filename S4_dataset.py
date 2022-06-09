"""Data processor for sentence classification.

In data file, each line contains (19+5)*3=72 attributes (XX, XX_sentence, XX_keywords) and other attributes.
The data processor convert each sentence into (sentence, class number) pair,
each sample with 1 sentence and 1 label.

Usage:
    from data import Data
    # For BERT model
    # For training, load train and valid set
    data = Data('model/bert/vocab.txt')
    datasets = data.load_train_and_valid_files(train_file='data/batch_one.xlsx', train_sheet='Sheet1',
                                                train_txt='data/sent_multi_label/',)
    train_set, valid_set_train = datasets
    # For testing, load test set
    data = TestData('model/bert/vocab.txt')
    test_set = data.load_from_txt(os.path.join(test_txt, index + '.txt'))
"""
import re
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
                 max_seq_len: int = 512):
        """Initialize data processor.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
        """
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def load_file(self,
                  filename: str = 'data/Nonrecurring Items_陈炫衣_20220226.xlsx',
                  sheet_name: str = 'Sheet1',
                  txt_path: str = 'data/sent_label/'):
        """Load train file and construct TensorDataset.

        Args:
            file_path: train file
            sheet_name: sheet name
            txt_path:
                If True, txt with 'sentence \t label'
                Otherwise, txt with paragraph

        Returns:
            dataset:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
        """
        ori_data = read_annotation(filename=filename, sheet_name=sheet_name)
        sent_list, label_list = [], []

        division = Division(ori_data)
        for index, one in division.data.iterrows():
            filename = os.path.join(txt_path, index + '.txt')
            f = open(filename, 'r', encoding='utf8')
            for line in f.readlines():
                one = line.split('\t')
                sent_list.append(self.tokenizer.tokenize(one[0]))
                # label_list.append(int(one[1]))
                label_list.append(eval(one[1]))

                if eval(one[1]) != [0] * division.num_classes:
                    for _ in range(4):
                        sent_list.append(self.tokenizer.tokenize(one[0]))
                        label_list.append(eval(one[1]))

        dataset = self._convert_sentence_to_bert_dataset(sent_list, label_list)

        return dataset, label_list

    def load_train_and_valid_files(self,
                                   train_file, train_sheet, train_txt,):
        """Load train files for task.

        Args:
            train_file: files for sentence classification.

        Returns:
            train_set, valid_set_train
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        mydataset, labels = self.load_file(train_file, sheet_name=train_sheet, txt_path=train_txt)
        train_set, valid_set = self.dataset_split(mydataset)
        print(len(train_set), 'train records loaded.', len(valid_set), 'valid records loaded.')

        return train_set, valid_set

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


class TestData:
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512,):
        """Initialize data processor.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
        """
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len
        self.datatool = Data(vocab_file, max_seq_len=max_seq_len)

    def load_from_txt(self, filename):
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
        sent_list = []

        sentences = self.txt2sent(filename=filename)
        for sent in sentences:
            sent_list.append(self.tokenizer.tokenize(sent))

        dataset = self.datatool._convert_sentence_to_bert_dataset(sent_list, [])

        return dataset

    def load_one(self, one_sent, one_label):
        """带标签测试集读取的傻逼函数，最后一定要删掉.

        Args:
            file_path: train file
            sheet_name: sheet name
            txt_path:
                If True, txt with 'sentence \t label'
                Otherwise, txt with paragraph

        Returns:
            dataset:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
        """
        sent_list, labels_list = [], []
        for sent in one_sent:
            sent_list.append(self.tokenizer.tokenize(sent))
            labels_list.append(one_label[sent.strip()])

        dataset = self.datatool._convert_sentence_to_bert_dataset(sent_list, labels_list)

        return dataset, labels_list

    def txt2sent(self, filename):
        '''
        copy from Division (S3_sentence_division.py )
        :param filename:
        :return: []
        '''
        sentence = []
        with open(filename, 'r', encoding='utf8') as f:
            paragraph = f.readlines()
            for para in paragraph:
                para2sent = self.sent_split(paragraph=para)
                sentence.extend(para2sent)
        f.close()

        return [x for x in sentence if x != '']

    def sent_split(self, paragraph):
        '''
        copy from Division (S3_sentence_division.py )
        :param paragraph:
        :return:
        '''
        # 段落划分为句子并除去过短元素（如单数字或空格）
        para2sent = re.split(';|\.|\([\s\S]{1,4}\)', paragraph.strip())
        # 保留分割符号，置于句尾，比如标点符号
        seg_word = re.findall(';|\.|\([\s\S]{1,4}\)', paragraph.strip())
        seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
        para2sent = [x + y for x, y in zip(para2sent, seg_word)]  # 顺序可根据需求调换
        # 除去句尾的括号项
        para2sent = [re.sub('\([\s\S]{1,4}\)$', '', sent) for sent in para2sent]

        return [one for one in para2sent if len(one) > 10]


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert-base-uncased/vocab.txt', max_seq_len=200)
    train, _ = data.load_train_and_valid_files(
        train_file='data/batch_one.xlsx', train_sheet='Sheet1', train_txt='data/sent_multi_label/',
    )


if __name__ == '__main__':
    test_data()
