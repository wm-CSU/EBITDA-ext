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
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import random_split
from transformers import BertTokenizer
from tqdm import tqdm
from utils import read_annotation
from S3_sentence_division import Division
import string
from zhon.hanzi import punctuation as chinese_punctuation


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
        self.vocab_file = vocab_file
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def for_word2vec(self, filename: str = 'data/Nonrecurring Items_陈炫衣_20220226.xlsx',
                     sheet_name: str = 'Sheet1',
                     txt_path: str = 'data/sent_label/'):
        ori_data = read_annotation(filename=filename, sheet_name=sheet_name)
        sent_list, label_list = [], []

        division = Division(ori_data)
        for index, one in division.data.iterrows():
            filename = os.path.join(txt_path, index + '.txt')
            f = open(filename, 'r', encoding='utf8')
            for line in f.readlines():
                one = line.split('\t')
                res = re.sub('[{}{}]'.format(string.punctuation, chinese_punctuation), "", one[0])
                sent_list.append(res.split())
                label_list.append(eval(one[1]))

        return sent_list, label_list

    def for_add_tokens(self, filename: str = 'data/Nonrecurring Items_陈炫衣_20220226.xlsx',
                       sheet_name: str = 'Sheet1',
                       txt_path: str = 'data/sent_label/'):
        ori_data = read_annotation(filename=filename, sheet_name=sheet_name)
        sent_list, label_list = [], []

        division = Division(ori_data)
        for index, one in division.data.iterrows():
            filename = os.path.join(txt_path, index + '.txt')
            f = open(filename, 'r', encoding='utf8')
            for line in f.readlines():
                one = line.split('\t')
                # 连字符保留
                res = re.sub('[{}{}]'.format(r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~""", chinese_punctuation), "", one[0])
                sent_list.extend(res.split())

        words = list(set(sent_list))
        words = [one for one in words if not bool(re.search(r'\d', one)) and not bool(re.match('[xvi]+$', one))
                 and len(one) > 3]
        vocab_list = self.read_vocab(self.vocab_file)
        need_add = [one for one in words if one not in vocab_list]
        num_added_toks = self.tokenizer.add_tokens(need_add)  # 返回一个数，表示加入的新词数量，在这里是2
        print(num_added_toks)

        return self.tokenizer

    def read_vocab(self, vocab):
        with open(vocab, 'r', encoding='utf8') as f:
            vocab_list = f.readlines()
        f.close()
        return vocab_list

    def load_file(self, filename: str = 'data/train.xlsx',
                  sheet_name: str = 'Sheet1',
                  txt_path: str = 'data/sent_multi_label/'):
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

        # f1 = open('data/used.txt', 'w+')
        # f2 = open('data/unused.txt', 'w+')
        division = Division(ori_data)
        for index, one in division.data.iterrows():
            filename = os.path.join(txt_path, index + '.txt')
            # filename = os.path.join(txt_path, '8063_95012311081957_2.txt')
            f = open(filename, 'r', encoding='utf8')
            for line in f.readlines():
                one = re.split('[\t\n]', line)
                sent_list.append(self.tokenizer.tokenize(one[0]))
                label_list.append(eval(one[1]))

                # if eval(one[1]) != [0] * division.num_classes:
                #     f1.write('1: ' + one[0] + '\n')
                # else:
                #     f2.write('0: ' + one[0] + '\n')
            # print('{} is loaded.'.format(filename))
        # f1.close()
        # f2.close()
        return sent_list, label_list

    def load_train_and_valid_files(self, train_list, train_sheet, train_txt, split_ratio=0.7):
        """Load train files for task.

        Args:
            train_file: files for sentence classification.

        Returns:
            train_set, valid_set_train
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        sent_list, label_list = [], []
        for name in train_list:
            one_file = os.path.join('data/train_file/', name+'.xlsx')
            one_txt = os.path.join(train_txt, name)

            sents, labels = self.load_file(one_file, sheet_name=train_sheet, txt_path=one_txt)
            sent_list.extend(sents)
            label_list.extend(labels)

        dataset, b2set = self._convert_sentence_to_bert_dataset(sent_list, label_list)

        train_set, valid_set = self.dataset_split(dataset, split_ratio)
        print(len(train_set), 'train records loaded.', len(b2set), 'train for branch 2 records loaded.',
              len(valid_set), 'valid records loaded.')

        return train_set, valid_set, b2set

    def dataset_split(self, dataset, split_ratio=0.7):
        train, valid = random_split(
            dataset,
            lengths=[int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))],
            generator=torch.Generator().manual_seed(2022))

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
        b2_input_ids, b2_input_mask, b2_segment_ids, b2_labels = [], [], [], []
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
            if label_list and sum(label_list[i]) != 0:
                b2_input_ids.append(input_ids)
                b2_input_mask.append(input_mask)
                b2_segment_ids.append(segment_ids)
                b2_labels.append(label_list[i])

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        b2_input_ids = torch.tensor(b2_input_ids, dtype=torch.long)
        b2_input_mask = torch.tensor(b2_input_mask, dtype=torch.long)
        b2_segment_ids = torch.tensor(b2_segment_ids, dtype=torch.long)
        b2_labels = torch.tensor(b2_labels, dtype=torch.long)

        if label_list or (sent_list == [] and label_list == []):  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids), \
                   TensorDataset(b2_input_ids, b2_input_mask, b2_segment_ids, b2_labels)

        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)


class TestData:
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512, ):
        """Initialize data processor.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
        """
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len
        self.datatool = Data(vocab_file, max_seq_len=max_seq_len)

    def for_word2vec(self, one_sent, one_label):
        sent_list, label_list = [], []
        for sent in one_sent:
            res = re.sub('[{}{}]'.format(string.punctuation, chinese_punctuation), "", sent)
            sent_list.append(res.split())

            label_list.append(one_label[sent.strip()])

        return sent_list, label_list

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

        dataset = self.datatool._convert_sentence_to_bert_dataset(sent_list)

        return dataset

    def load_one(self, one_sent, one_label):
        """带标签测试集读取

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

        dataset, _ = self.datatool._convert_sentence_to_bert_dataset(sent_list, labels_list)

        return dataset, labels_list

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
        self.label_map = self._get_label_map()
        ori_data = read_annotation(filename=filename, sheet_name=sheet_name)
        sent_list, label_list = [], []

        from S1_preprocess import Drop_Redundance
        data = Drop_Redundance(ori_data, Train=False)
        division = Division(data)
        for index, one in data.iterrows():
            filename = os.path.join(txt_path, index + '.txt')
            one_sent = division.txt2sent(filename=filename)
            one_label = division.sent_label(one_data=one, one_sent=one_sent,
                                            label_map=self.label_map)
            for sent in one_sent:
                sent_list.append(self.tokenizer.tokenize(sent))
                label_list.append(one_label[sent.strip()])

        dataset = self.datatool._convert_sentence_to_bert_dataset(sent_list, label_list)

        return dataset, label_list

    def _get_label_map(self, filename: str = r'data/label_map.json'):
        import json
        with open(filename, 'r') as f:
            label_map = json.load(f)
            f.close()

        return label_map

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
                mid = self.sent_split(paragraph=para)
                para2sent = self.sent_resplit(paragraph=mid)
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

    def sent_resplit(self, paragraph):
        for para in paragraph:
            if len(para) > 500:  # resplit
                re_sent = re.split(',|\(.{5,}?\)', para.strip())
                # 保留分割符号，置于句尾，比如标点符号
                seg_word = re.findall(',|\(.{5,}?\)', para.strip())
                seg_word.extend(" ")  # 末尾插入一个空字符串，以保持长度和切割成分相同
                re_sent = [x + y for x, y in zip(re_sent, seg_word)]  # 顺序可根据需求调换
                for one in range(len(re_sent) - 1):
                    if re_sent[one].endswith(')'):
                        re_sent[one] = re_sent[one] + re_sent[one + 1]
                        re_sent[one + 1] = ''
                re_sent = [one for one in re_sent if len(one) > 10]

                paragraph.extend(re_sent)
                paragraph.remove(para)
            else:
                continue

        return paragraph


def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('model/bert-base-uncased/vocab.txt', max_seq_len=200)
    train, _ = data.load_train_and_valid_files(
        train_file='data/batch_one.xlsx', train_sheet='Sheet1', train_txt='data/sent_multi_label/',
    )


if __name__ == '__main__':
    test_data()
