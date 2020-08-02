#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   xbert_cnn_runner.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import logging
import warnings
from multiprocessing import cpu_count
import wandb
import random
import numpy as np

import torch
from base_runner import BaseRunner
from global_config import global_configs

from models.multi_class_cnn import (
    BertCNN,
    AlbertCNN,
)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNConfig:
    num_filters = 512
    kernel_sizes = [3, 4, 5]


class MultiClassCnnRunner(BaseRunner):
    def __init__(
        self,
        model_type,
        model_file,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        freez_pretrained=False,
        **kwargs
    ):
        """
        Args:
            model_type: MODEL_CLASSES中的模型名称，bert，xlnet，roberta，distilbert，albert，electra
            model_file: 预训练的模型文件路径，或者已经fine-tune过的模型保存路径，或者在huggingface注册的模型名称
                        https://huggingface.co/models?search=chinese
            num_labels (optional): 1. 一种label多分类情况下，表示 num of classes. 等于1时使用Mean-Square loss， 大于1时使用Cross-Entropy loss
                                2. 多种label，每个label为二分类，表示 num of labels. 使用Binary Cross Entropy loss.(BCEWithLogitsLoss)
                                3. 多种label，每个label为多分类，表示 num of labels. 需要输入num_label_subclass. 损失函数比较负杂，
                                    需要进一步设计，e.g 每种label进行单独的Cross-Entropy loss，合成总的loss，BP。或者转化为onehot，
                                    使用Binary Cross Entropy loss. label数量变为onehot的总维度。
                                根据实际情况修改模型的loss计算方法部分代码和评价指标的设计代码。
            weight (optional): 不平衡类别时，可选的类别权重list
            args (optional): 模型参数设置dict
            use_cuda (optional): 是否用GPU
            cuda_device (optional): 默认使用第一个识别到的设备
            freez_pretrained: 是否固定预训练模型
            **kwargs (optional): 传入transformers的from_pretrained方法的参数
        """
        MODEL_CLASSES = {
            "bert": (BertConfig, BertCNN, BertTokenizer),
            # "albert": (AlbertConfig, AlbertCNN, AlbertTokenizer),
            # 由于 albert_chinese_* 模型没有用 sentencepiece.
            # 用AlbertTokenizer加载不了词表，因此需要改用BertTokenizer
            # https://huggingface.co/voidful/albert_chinese_xxlarge
            "albert": (AlbertConfig, AlbertCNN, BertTokenizer),
            # "xlnet": (XLNetConfig, XLNetForMultiBinaryLabelSeqClassification, XLNetTokenizer),
            # "roberta": (RobertaConfig, RobertaForMultiBinaryLabelSeqClassification, RobertaTokenizer),
            # "distilbert": (DistilBertConfig, DistilBertForMultiBinaryLabelSeqClassification, DistilBertTokenizer),
            # "electra": (ElectraConfig, ElectraForMultiBinaryLabelSeqClassification, ElectraTokenizer),
        }

        if args and "manual_seed" in args:
            random.seed(args["manual_seed"])
            np.random.seed(args["manual_seed"])
            torch.manual_seed(args["manual_seed"])
            if "n_gpu" in args and args["n_gpu"] > 0:
                torch.cuda.manual_seed_all(args["manual_seed"])

        # ### 参数设置 ###
        # sliding_window：处理长序列时，超过了最长长度限制，使用sliding_window，每次输入 stride * max_seq_length 的长度序列
        # sliding_window，实际效果并不理想，一般不适使用这种方法。MultiLabel情况下没有实现。
        self.args = {
            "sliding_window": False,
            "tie_value": -1,  # sliding_window为True时生效，预测时同一个文档的预测结果占位符
            "stride": 0.8,
            "regression": False,
            "threshold": 0.5
        }
        # NOTE: threshold，在预测predict方法中使用，如果是multi_label的问题，threshold是一个 list。

        self.args.update(global_configs)
        saved_model_args = self._load_model_args(model_file)
        if saved_model_args:
            self.args.update(saved_model_args)
        if args:
            self.args.update(args)
        self.args["model_file"] = model_file
        self.args["model_type"] = model_type

        self.results = {}

        # ### TODO:导入预训练模型、config ###
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(model_file, num_labels=num_labels, **self.args["config"])
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_file, **self.args["config"])
            self.num_labels = self.config.num_labels
        self.config.output_hidden_states = False
        self.config.output_attentions = False

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError("没有可用的CUDA环境.")
        else:
            self.device = "cpu"
        # NOTE: 载入CNN参数
        extra_config = CNNConfig()

        # TODO: 导入预训练模型
        self.weight = weight
        if self.weight:
            self.model = model_class.from_pretrained(
                model_file,
                config=self.config,
                extra_config=extra_config,
                weight=torch.Tensor(self.weight).to(self.device),
                freez_pretrained=freez_pretrained,
                **kwargs)
        else:
            self.model = model_class.from_pretrained(
                model_file,
                config=self.config,
                extra_config=extra_config,
                freez_pretrained=freez_pretrained,
                **kwargs)

        if not use_cuda:
            self.args["fp16"] = False

        # TODO: 初始化预训练模型的tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(
            model_file, do_lower_case=self.args["do_lower_case"], **kwargs
        )


    def train_model(
        self,
        train_df,
        multi_label=False,
        eval_df=None,
        output_dir=None,
        show_running_loss=True,
        args=None,
        verbose=True,
        **kwargs,
    ):
        return super().train_model(
            train_df,
            multi_label=multi_label,
            eval_df=eval_df,
            output_dir=output_dir,
            show_running_loss=show_running_loss,
            verbose=True,
            args=args,
            **kwargs,
        )

    def eval_model(self, eval_df, multi_label=False, output_dir=None, verbose=False, silent=False, **kwargs):
        """**kwargs中可传入metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
        注意函数适用条件。或自定义函数"""
        return super().eval_model(
            eval_df, output_dir=output_dir, multi_label=multi_label, verbose=verbose, silent=silent, **kwargs
        )

    def evaluate(self, eval_df, output_dir, multi_label=False, verbose=True, silent=False, **kwargs):
        return super().evaluate(
            eval_df, output_dir, multi_label=multi_label, verbose=verbose, silent=silent, **kwargs
        )

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(
            examples, evaluate=evaluate, no_cache=no_cache, multi_label=multi_label, verbose=verbose, silent=silent
        )

    def compute_metrics(self, preds, labels, eval_examples, multi_label=False, **kwargs):
        return super().compute_metrics(preds, labels, eval_examples, multi_label=multi_label, **kwargs)

    def predict(self, to_predict, multi_label=False):
        return super().predict(to_predict, multi_label=multi_label)
