#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   custom_classification.py
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
from transformers.modeling_roberta import RobertaClassificationHead
from transformers.modeling_utils import SequenceSummary


class BertForMultiBinaryLabelSeqClassification(BertPreTrainedModel):
    """Bert model adapted for multi-label sequence classification.
    Every label is binary class."""
    def __init__(self, config, freez_pretrained=False, weight=None):
        "weight: 各个label样本中正例的比例，len==num_labels"
        super(BertForMultiBinaryLabelSeqClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.weight = weight
        self.init_weights()
        if freez_pretrained:
            for param in self.bert.parameters():
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
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits, ) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForMultiBinaryLabelSeqClassification(AlbertPreTrainedModel):
    """Alber model adapted for multi-label sequence classification.
    Every label is binary class."""
    def __init__(self, config, freez_pretrained=False, weight=None):
        super(AlbertForMultiBinaryLabelSeqClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()
        print("Freeze参数：", freez_pretrained)
        if freez_pretrained:
            for param in self.albert.parameters():
                param.requires_grad = False
                # print(param.requires_grad)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ):
        "可以直接输入input ids对应的inputs_embeds"
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        # pooler_output:      Last layer hidden-state of the first token of the sequence (classification token)
        #                     processed by a Linear layer and a Tanh activation function.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits, ) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DistilBertConfig
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistilBertForMultiBinaryLabelSeqClassification(DistilBertPreTrainedModel):
    """DistilBert model adapted for multi-label sequence classification
    Every label is binary class.
    """

    def __init__(self, config, freez_pretrained=False, weight=None):
        super(DistilBertForMultiBinaryLabelSeqClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.distilbert = DistilBertModel(config)
        self.linear = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()
        if freez_pretrained:
            for param in self.distilbert.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        "可以直接输入input ids对应的inputs_embeds"
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask)
        hidden_state = distilbert_output[0]           # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]            # (bs, dim)    [CLS]位置
        pooled_output = self.linear(pooled_output)    # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)      # (bs, dim)
        pooled_output = self.dropout(pooled_output)   # (bs, dim)
        logits = self.classifier(pooled_output)       # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLNetForMultiBinaryLabelSeqClassification(XLNetPreTrainedModel):
    """
    XLNet model adapted for multi-label sequence classification.
    Every label is binary class.
    """

    def __init__(self, config, freez_pretrained=False, weight=None):
        super(XLNetForMultiBinaryLabelSeqClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.transformer = XLNetModel(config)
        # Compute a single vector summary of a sequence hidden states according to various possibilities
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)
        self.init_weights()
        if freez_pretrained:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        # mems:               Contains pre-computed hidden-states (key and values in the attention blocks).
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)  # identity or other mapping.
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class RobertaForMultiBinaryLabelSeqClassification(BertPreTrainedModel):
    """
    Roberta model adapted for multi-label sequence classification.
    Every label is binary class.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, freez_pretrained=False, weight=None):
        super(RobertaForMultiBinaryLabelSeqClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        if freez_pretrained:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        # pooler_output:      分类提取特征没有使用pooler_output.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # last_hidden_state
        # RobertaClassificationHead: 取 <s> token作为特征进行分类
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# ELECTRA是一种预训练方法，因此基本模型BERT几乎没有任何变化。
# 唯一的变化是embed size和hidden size的分离->embed size一般较小，而hidden size较大。
# 额外的project(线性)用于将嵌入项从embed size投影到hidden size。
# 在embed size与hidden size相同的情况下，不使用投影层。

# The ELECTRA checkpoints saved using Google Research’s implementation contain both the generator and discriminator.
# The conversion script requires the user to name which model to export into the correct architecture.
# Once converted to the HuggingFace format, these checkpoints may be loaded into all available ELECTRA models, however.
# This means that the discriminator may be loaded in the ElectraForMaskedLM model,
# and the generator may be loaded in the ElectraForPreTraining model.
#   (the classification head will be randomly initialized as it doesn’t exist in the generator)

class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ElectraForSequenceClassification(ElectraPreTrainedModel):
    r"""分类任务只使用了discriminator。但是HuggingFace的实现，同时会载入generator，但未使用。
    相比于BertForSequenceClassification不同之处在于外接了一层Pooler. 因为BertModel的Pooler在内部实现，而ElectraModel没有实现.
    ElectraModel.
        labels: (optional) 'torch.LongTensor' of shape '(batch_size,)':
            Labels for computing the sequence classification/regression loss.
            Indices should be in '[0, ..., config.num_labels - 1]'.
            If 'config.num_labels == 1' a regression loss is computed (Mean-Square loss),
            If 'config.num_labels > 1' a classification loss is computed (Cross-Entropy).
    Outputs: "Tuple" comprising various elements depending on the configuration (config) and inputs:
        loss: (optional, returned when 'labels' is provided) 'torch.FloatTensor' of shape '(1,)':
            Classification (or regression if config.num_labels==1) loss.
        logits: 'torch.FloatTensor' of shape '(batch_size, config.num_labels)'
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states: (optional, returned when 'config.output_hidden_states=True')
            list of 'torch.FloatTensor' (one for the output of each layer + the output of the embeddings)
            of shape '(batch_size, sequence_length, hidden_size)':
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions: (optional, returned when 'config.output_attentions=True')
            list of 'torch.FloatTensor' (one for each layer) of shape '(batch_size, num_heads, sequence_length, sequence_length)':
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config, freez_pretrained=False, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.weight = weight
        if freez_pretrained:
            for param in self.electra.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[1:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ElectraForMultiBinaryLabelSeqClassification(ElectraPreTrainedModel):
    """
    ElectraForSequenceClassification model adapted for multi-label sequence classification.
    Every label is binary class.
    """

    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config, freez_pretrained=False, weight=None):
        super(ElectraForMultiBinaryLabelSeqClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        if freez_pretrained:
            for param in self.electra.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ):
        # outputs的组成：
        # last_hidden_state： Sequence of hidden-states at the output of the last layer of the model.
        # hidden_states：     one for the output of the embeddings + one for the output of each layer.
        #                     each is (batch_size, sequence_length, hidden_size)
        # attentions:         Attentions weights after the attention softmax of each layer.
        #                     each is (batch_size, num_heads, sequence_length, sequence_length)
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(weight=self.weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)