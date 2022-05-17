"""BERT model for sentence classification.

Author: Min Wang; wangmin0918@csu.edu.cn
"""

import torch
from torch import nn
from transformers import AutoModel


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
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
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

        dropout_output = self.dropout(pooled_output)
        logits = self.linear(dropout_output).view(batch_size, self.num_classes)  # logits: (batch_size, num_classes)
        # logits = nn.functional.softmax(logits, dim=1)

        return logits, pooled_output
