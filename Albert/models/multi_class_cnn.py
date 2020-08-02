#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_class_cnn.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    PreTrainedModel,
    BertModel,
    BertPreTrainedModel,
    AlbertModel,
    AlbertPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
    DistilBertConfig,
    DistilBertModel,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    RobertaConfig,
    RobertaModel,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_roberta import RobertaClassificationHead, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_electra import ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_utils import SequenceSummary


"""
Bert类模型加上CNN，结合multi_label_linear.py，可以轻松定义任意Bert类模型。
"""

class BertCNN(BertPreTrainedModel):
    """bert + cnn, multi class classification."""
    def __init__(self,
                 config,
                 extra_config,
                 freez_pretrained=False,
                 weight=None):
        "weight: 各个label样本中正例的比例，len==num_labels"
        super(BertCNN, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.hidden_size, extra_config.num_filters,
                      kernel_size) for kernel_size in extra_config.kernel_sizes
        ])
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(2 * extra_config.num_filters *
                                len(extra_config.kernel_sizes), self.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.weight = weight
        self.init_weights()
        if freez_pretrained:
            for param in self.albert.parameters():
                param.requires_grad = False

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        #                     (batch_size, sequence_length, hidden_size)
        # pooler_output:      Last layer hidden-state of the first token of the sequence (classification token)
        #                     processed by a Linear layer and a Tanh activation function.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        # cnn
        last_hidden_state = outputs[0]
        cnn_out = torch.cat([conv(last_hidden_state) for conv in self.convs], 1)
        maxpool_out = self.maxpooling(cnn_out)
        avgpool_out = self.avgpooling(cnn_out)
        out = torch.cat([maxpool_out, avgpool_out], 1)
        out = self.linear(out)

        # linear
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        linear_out = self.classifier(pooled_output)

        # res
        logits = linear_out + out

        outputs = (logits, ) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        # (loss), logits, (hidden_states), (attentions)
        return outputs


class AlbertCNN(BertPreTrainedModel):
    """bert + cnn, multi class classification."""
    def __init__(self,
                 config,
                 extra_config,
                 freez_pretrained=False,
                 weight=None):
        "weight: 各个label样本中正例的比例，len==num_labels"
        super(AlbertCNN, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = AlbertModel(config)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.hidden_size, extra_config.num_filters,
                      kernel_size) for kernel_size in extra_config.kernel_sizes
        ])
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(2 * extra_config.num_filters *
                                len(extra_config.kernel_sizes), self.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.weight = weight
        self.init_weights()
        if freez_pretrained:
            for param in self.albert.parameters():
                param.requires_grad = False

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        #                     (batch_size, sequence_length, hidden_size)
        # pooler_output:      Last layer hidden-state of the first token of the sequence (classification token)
        #                     processed by a Linear layer and a Tanh activation function.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.albert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        # cnn
        last_hidden_state = outputs[0]
        cnn_out = torch.cat([conv(last_hidden_state) for conv in self.convs], 1)
        maxpool_out = self.maxpooling(cnn_out)
        avgpool_out = self.avgpooling(cnn_out)
        out = torch.cat([maxpool_out, avgpool_out], 1)
        out = self.linear(out)

        # linear
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        linear_out = self.classifier(pooled_output)

        # res
        logits = linear_out + out

        outputs = (logits, ) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        # (loss), logits, (hidden_states), (attentions)
        return outputs
