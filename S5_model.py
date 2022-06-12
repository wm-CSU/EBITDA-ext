"""BERT model for sentence classification.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

import torch
from torch import nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence


class BertForClassification(nn.Module):
    """BERT with simple linear model."""

    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-uncased'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super(BertForClassification, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.model_path)
        for param in self.bert.parameters():  # freeze bert parameters
            param.requires_grad = True
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size,
                              hidden_size=self.config.hidden_size // 2,
                              num_layers=self.config.bilstm_num_layers,
                              # dropout=self.config.dropout,  # 1-layer not need dropout
                              batch_first=True,
                              bidirectional=True)  # get(batch, seq, feature)
        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes
        self.labels_co_mat = config.labels_co_mat

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, max_seq_len = input_ids.shape[0], input_ids.shape[1]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        if self.config.pooling == 'cls':
            pooled_output = bert_output.last_hidden_state[:, 0]  # [batch, 768]
        elif self.config.pooling == 'pooler':
            pooled_output = bert_output.pooler_output  # [batch, 768]
        elif self.config.pooling == 'last-avg':
            last = bert_output.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            pooled_output = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.config.pooling == 'first-last-avg':
            first = bert_output.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = bert_output.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            pooled_output = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        else:
            pooled_output = bert_output.pooler_output

        # # 送入 bilstm 前，计算 actual_length
        actual_length = []
        for i in range(input_ids.shape[0]):
            one = (input_ids[i, :] == 0).nonzero(as_tuple=True)[0]
            if one == torch.Size([]):
                actual_length.append(one[0])
            else:
                actual_length.append(max_seq_len)
        # ''' Bi-LSTM Computation '''
        # sorted_seq_len, permIdx = actual_length.sort(0, descending=True)
        sorted_seq_len, permIdx = torch.tensor(actual_length).sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_emb = bert_output.last_hidden_state[permIdx]
        pack_input = pack_padded_sequence(sorted_emb, sorted_seq_len, batch_first=True)
        _, char_hidden = self.bilstm(pack_input, None)  # (batch_size, sequence_length, hidden_size)
        # char_hidden = (h_t, c_t);  char_hidden[0] = h_t = (2, batch_size, lstm_hidden_size)
        hidden = char_hidden[0].transpose(1, 0).contiguous().view(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
        # transpose because the first dimension is  num_direction X num-layer .  contiguous is similar to deep copy.
        # before view, the size is (batch_size, 2, hidden_size/2) 2 means 2 direction.
        lstm_res = hidden[recover_idx].squeeze(1)  # (batch_size, hidden_size)

        conjoint_res = torch.cat((pooled_output, lstm_res), dim=1)  # [batch, 2*768]
        dropout_output = self.dropout(conjoint_res)
        logits = self.linear(dropout_output).view(batch_size, self.num_classes)  # logits: (batch_size, num_classes)
        # logits = torch.mm(logits, self.labels_co_mat)  # logits: (batch_size, num_classes)

        return logits, pooled_output
