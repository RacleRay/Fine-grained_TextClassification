#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_runer.py
'''

import json
import logging
import math
import os
import random
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import confusion_matrix, label_ranking_average_precision_score, matthews_corrcoef
from tqdm.auto import tqdm, trange
import wandb

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tensorboardX import SummaryWriter

from transformers import (
  AlbertForSequenceClassification,
  BertForSequenceClassification,
  DistilBertForSequenceClassification,
  RobertaForSequenceClassification,
  XLNetForSequenceClassification
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

from models.multi_label_linear import ElectraForSequenceClassification

from base_utils import InputExample, convert_examples_to_features
from global_config import global_configs


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseRunner:
  def __init__(
      self,
      model_type,
      model_file,
      num_labels=None,
      weight=None,
      args=None,
      use_cuda=True,
      cuda_device=-1,
      **kwargs,
    ):
    """
    Initializes a BaseRunner model. 参数设置见global_config.py

    Args:
        model_type: MODEL_CLASSES中的模型名称，bert，xlnet，roberta，distilbert，albert，electra
        model_file: 预训练的模型文件路径，或者已经fine-tune过的模型保存路径，或者在huggingface注册的模型名称
                    https://huggingface.co/models?search=chinese
        num_labels (optional): 1. 一种label多分类情况下，表示 num of classes. 等于1时使用Mean-Square loss， 大于1时使用Cross-Entropy loss
                               2. 多种label，每个label为二分类，表示 num of labels. 使用Binary Cross Entropy loss.(BCEWithLogitsLoss)
                               3. 多种label，每个label为多分类，表示 num of labels. 需要输入num_label_subclass. 损失函数比较负杂，
                                  需要进一步设计，e.g 每种label进行单独的Cross-Entropy loss，合成总的loss，BP。或者转化为onehot，
                                  使用Binary Cross Entropy loss.
                               根据实际情况修改模型的loss计算方法部分代码和评价指标的设计代码。
        weight (optional): 不平衡类别时，可选的类别权重list
        args (optional): 模型参数设置dict
        use_cuda (optional): 是否用GPU
        cuda_device (optional): 默认使用第一个识别到的设备
        **kwargs (optional): 传入transformers的from_pretrained方法的参数
    """
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        # "albert": (AlbertConfig, AlbertForMultiBinaryLabelSequenceClassification, AlbertTokenizer),
        # 由于 albert_chinese_* 模型没有用 sentencepiece.
        # 用AlbertTokenizer加载不了词表，因此需要改用BertTokenizer
        # https://huggingface.co/voidful/albert_chinese_xxlarge
        "albert": (AlbertConfig, AlbertForSequenceClassification, BertTokenizer),
        "electra": (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
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

    print("全局设置：", self.args)

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

    print("模型设置：", self.config)
    
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

    # TODO: 导入预训练模型
    self.weight = weight
    if self.weight:
      self.model = model_class.from_pretrained(
          model_file, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs)
    else:
      self.model = model_class.from_pretrained(model_file, config=self.config, **kwargs)

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
      output_dir=None,
      show_running_loss=True,
      args=None,
      eval_df=None,
      verbose=True,
      **kwargs,
    ):
    """
    Args:
        train_df: Pandas Dataframe. 至少含有两列'text'和'labels'，表头可以不指定，或者三列'text_a'、'text_b','labels'.
                                    labels处理成一个list或者array，保存在Dataframe中。
        multi_label：是否是多种不同种类的标签。e.g [是否相信科学，是否吃过早饭, ...].
                     影响模型的loss计算方法，评价指标的设计。multi_label有两种情况，1. 每一种label为二分类， 2. 每一种label为多分类
                     e.g [相信科学的程度(1,2,3,4)，早饭饭量(1,2,3,4),...].
                     multi_label的训练有一点需要注意，不同label的隐含关系，可能有相关性。
                     根据实际情况修改模型的loss计算方法部分代码和评价指标的设计代码。
        output_dir: 模型保存路径，在self.args['output_dir']，可以修改. self.args读取config文件的设置.
        show_running_loss (optional): Defaults True.
        args (optional): 对config中的设置进行修改.
        eval_df (optional): ‘evaluate_during_training‘为True时，传入进行评估运行效果.
        **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
    Returns:
        None
    """
    # 检查参数设置
    if args: self.args.update(args)
    if not output_dir: output_dir = self.args["output_dir"]
    if self.args["silent"]: show_running_loss = False
    if self.args["evaluate_during_training"] and eval_df is None:
      raise ValueError("请指定eval_df")
    if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
      raise ValueError("output_dir ({}) 已存在. 设置 --overwrite_output_dir 覆盖文件.".format(output_dir))

    # 输入数据读取
    if "text" in train_df.columns and "label" in train_df.columns:
      train_examples = [
        InputExample(i, text, None, label) for i, (text, label) in enumerate(
            zip(train_df["text"], train_df["label"]))
        ]
    elif "text_a" in train_df.columns and "text_b" in train_df.columns:
      train_examples = [
        InputExample(i, text_a, text_b, label) for i, (text_a, text_b, label) in enumerate(
            zip(train_df["text_a"], train_df["text_b"], train_df["label"]))
        ]
    else:
      warnings.warn("未指定训练数据标题行, 自动使用第一行作为数据开始.")
      train_examples = [
        InputExample(i, text, None, label)
        for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))
        ]
    train_dataset = self.load_and_cache_examples(train_examples, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)

    self.model.to(self.device)
    global_step, total_running_loss = self.train(
        train_dataset,
        output_dir,
        multi_label=multi_label,
        show_running_loss=show_running_loss,
        eval_df=eval_df,
        verbose=verbose,
        **kwargs,
        )
    self._save_model(model=self.model)  # 可以传入优化器，保存当前优化器状态
    if verbose:
      logger.info(" 训练 {} 结束. Saved to {}.".format(self.args["model_type"], output_dir))
      logger.info(" 当前global step： {}. 运行平局loss： {}.".format(global_step, total_running_loss))

  def train(
      self,
      train_dataset,
      output_dir,
      multi_label=False,
      show_running_loss=True,
      eval_df=None,
      verbose=True,
      **kwargs,
      ):
    # ==============================
    # TODO: settings
    # ==============================
    device = self.device
    model = self.model
    args = self.args

    global_step = 0
    total_running_loss, previous_log_running_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args["silent"], mininterval=0)
    epoch_number = 0
    best_eval_metric = None
    early_stopping_counter = 0
    steps_trained_in_current_epoch = 0
    epochs_trained = 0

    tb_writer = SummaryWriter(logdir=args["tensorboard_dir"])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

    # num_training_steps
    # gradient_accumulation 累积一定数量loss.backward，再更新模型参数optimizer.step()
    t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
       "weight_decay": args["weight_decay"]},
      {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
       "weight_decay": 0.0},
      ]

    # warm up
    warmup_steps = math.ceil(t_total * args["warmup_ratio"])
    args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total)

    # 半精度
    if args["fp16"]:
      try:
        from apex import amp
      except ImportError:
        raise ImportError("需要安装 apex ：https://www.github.com/nvidia/apex to use fp16 training.")
      model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

    # 模型复制，数据并行，梯度累计
    if args["n_gpu"] > 1:
      model = torch.nn.DataParallel(model)

    # model file
    if args["model_file"] and os.path.exists(args["model_file"]):
      try:  # 读取已经fine-tune过的模型文件
        # checkpoint中读取global_step
        checkpoint_suffix = args["model_file"].split("/")[-1].split("-")
        if len(checkpoint_suffix) > 2:
          checkpoint_suffix = checkpoint_suffix[1]
        else:
          checkpoint_suffix = checkpoint_suffix[-1]
        # gradient_accumulation_steps: 是每隔这么多步才更新一次梯度，所以数据遍历的个数要乘上steps
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step * args["gradient_accumulation_steps"] // (len(train_dataloader))
        steps_trained_in_current_epoch = (global_step * args["gradient_accumulation_steps"] % (
            len(train_dataloader))) // args["gradient_accumulation_steps"]

        logger.info(">>> 加载checkpoint，更新global_step.")
        logger.info(">>> 已训练 %d 轮.", epochs_trained)
        logger.info(">>> 当前global_step： %d.", global_step)
        logger.info(">>> 跳过当前epoch %d steps in.", steps_trained_in_current_epoch)
      except ValueError:  # 读取预训练文件，fine-tuning
        logger.info(">>> 开始 fine-tuning.")

    # 设置训练过程中，评价模型的指标
    if args["evaluate_during_training"]:
      training_progress_scores = self._create_training_progress_scores(multi_label, **kwargs)

    # weight & bias：记录模型参数变化
    if args["wandb_project"]:
      wandb.init(project=args["wandb_project"], config={**args}, **args["wandb_kwargs"])
      wandb.watch(self.model)

    # ================================
    # TODO: 训练框架流程
    # ================================
    model.train()
    for _ in train_iterator:
      if epochs_trained > 0:  # 处理加载checkpoint的情况
        epochs_trained -= 1
        continue
      # 迭代一个epoch
      for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args["silent"])):
        if steps_trained_in_current_epoch > 0:
          steps_trained_in_current_epoch -= 1
          continue
        batch = tuple(t.to(device) for t in batch)

        # 运行模型前向计算
        inputs = self._get_inputs_dict(batch)  # 不同模型输入有一些差异
        # ================================
        # TODO: 修改模型代码
        # ================================
        outputs = model(**inputs)

        # ### BP训练设计 ###
        # outputs： ((loss), logits, (hidden_states), (attentions))
        loss = outputs[0]
        if args["n_gpu"] > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training
        current_loss = loss.item()
        if show_running_loss and step % 100 == 0:
          print("\rRunning loss: %f" % loss, end="")

        # Normalize our loss (if averaged)
        if args["gradient_accumulation_steps"] > 1:
          loss = loss / args["gradient_accumulation_steps"]

        if args["fp16"]:
          with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        else:
          loss.backward()

        total_running_loss += loss.item()
        if (step + 1) % args["gradient_accumulation_steps"] == 0:
          if args["fp16"]:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
          else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

          optimizer.step()  # gradient_accumulation_steps 更新一次参数
          scheduler.step()  # Update learning rate schedule
          model.zero_grad()
          global_step += 1

          # 记录参数变化
          if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", (total_running_loss - previous_log_running_loss) / args["logging_steps"], global_step)
            previous_log_running_loss = total_running_loss
            if args["wandb_project"]:
              wandb.log({"Training loss": current_loss, "lr": scheduler.get_lr()[0], "global_step": global_step})

          # Save model checkpoint
          if args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
            output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
            self._save_model(output_dir_current, optimizer, scheduler, model=model)

          # ### Evaluate every xxx steps ###
          if args["evaluate_during_training"] and (args["evaluate_during_training_steps"] > 0
              and global_step % args["evaluate_during_training_steps"] == 0):
            # Evaluate 确保在单GPU环境下运行
            results, model_outputs, wrong_preds = self.eval_model(
                eval_df,
                verbose=verbose and args["evaluate_during_training_verbose"],
                silent=True,
                **kwargs,
                )
            for key, value in results.items():  # results: compute_metrics输出的字典
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)  # tensorboard记录

            # 保存Evaluate时的model状态
            output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))
            if args["save_eval_checkpoints"]:
              self._save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

            # 保存log
            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
              training_progress_scores[key].append(results[key])
            report = pd.DataFrame(training_progress_scores)
            report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

            # weight & bias
            if args["wandb_project"]:
              wandb.log({metric: values[-1] for metric, values in training_progress_scores.items()})

            # ### Early stop every xxx steps ###
            if not best_eval_metric:
              best_eval_metric = results[args["early_stopping_metric"]]  # early_stopping_metric: early_stopping的参照指标
              self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
            # 最小化early_stopping的参照指标为目标
            if best_eval_metric and args["early_stopping_metric_minimize"]:
              if (
                  results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]
              ):
                best_eval_metric = results[args["early_stopping_metric"]]
                self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                early_stopping_counter = 0
              else:
                if args["use_early_stopping"]:
                  if early_stopping_counter < args["early_stopping_patience"]:
                    early_stopping_counter += 1
                    if verbose:
                      logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                      logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                      logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
                  else:
                    if verbose:
                      logger.info(f" 达到Early stop最大容忍轮次.")
                      logger.info(" 停止训练@.")
                      train_iterator.close()
                    return global_step, total_running_loss / global_step
            else:  # 最大化early_stopping的参照指标为目标
              if (results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]):
                best_eval_metric = results[args["early_stopping_metric"]]
                self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
                early_stopping_counter = 0
              else:
                if args["use_early_stopping"]:
                  if early_stopping_counter < args["early_stopping_patience"]:
                    early_stopping_counter += 1
                    if verbose:
                      logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                      logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                      logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
                  else:
                    if verbose:
                      logger.info(f" 达到Early stop最大容忍轮次.")
                      logger.info(" 停止训练@.")
                      train_iterator.close()
                    return global_step, total_running_loss / global_step
      # 完成一个epoch
      epoch_number += 1
      output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

      # save model
      if args["save_model_every_epoch"] or args["evaluate_during_training"]:
        os.makedirs(output_dir_current, exist_ok=True)
      if args["save_model_every_epoch"]:
        self._save_model(output_dir_current, optimizer, scheduler, model=model)

      # ### Evaluate every epoch ###
      # 逻辑和 Evaluate every xxx steps 一样
      if args["evaluate_during_training"]:
        results, _, _ = self.eval_model(
            eval_df, verbose=verbose and args["evaluate_during_training_verbose"], silent=True, **kwargs
            )
        self._save_model(output_dir_current, optimizer, scheduler, results=results)

        training_progress_scores["global_step"].append(global_step)
        training_progress_scores["train_loss"].append(current_loss)
        for key in results:
          training_progress_scores[key].append(results[key])
        report = pd.DataFrame(training_progress_scores)
        report.to_csv(os.path.join(output_dir, "training_progress_scores.csv"), index=False)

        if args["wandb_project"]:
          wandb.log(self._get_last_metrics(training_progress_scores))

        # ### Early stop every epoch ###
        if not best_eval_metric:
          best_eval_metric = results[args["early_stopping_metric"]]
          self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
        # 最小化early_stopping的参照指标为目标
        if best_eval_metric and args["early_stopping_metric_minimize"]:
          if results[args["early_stopping_metric"]] - best_eval_metric < args["early_stopping_delta"]:
            best_eval_metric = results[args["early_stopping_metric"]]
            self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
            early_stopping_counter = 0
          else:
            if args["use_early_stopping"] and args["early_stopping_consider_epochs"]:  # 在epoch维度上进行early_stopping
              if early_stopping_counter < args["early_stopping_patience"]:
                early_stopping_counter += 1
                if verbose:
                  logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                  logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                  logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
              else:
                if verbose:
                  logger.info(f" 达到Early stop最大容忍轮次.")
                  logger.info(" 停止训练@.")
                  train_iterator.close()
                return global_step, total_running_loss / global_step
        else:  # 最大化early_stopping的参照指标为目标
          if results[args["early_stopping_metric"]] - best_eval_metric > args["early_stopping_delta"]:
            best_eval_metric = results[args["early_stopping_metric"]]
            self._save_model(args["best_model_dir"], optimizer, scheduler, model=model, results=results)
            early_stopping_counter = 0
          else:
            if args["use_early_stopping"] and args["early_stopping_consider_epochs"]:
              if early_stopping_counter < args["early_stopping_patience"]:
                early_stopping_counter += 1
                if verbose:
                  logger.info(f" Early stop指标 {args['early_stopping_metric']} 没有提升.")
                  logger.info(f" 当前已执行early stop count: {early_stopping_counter}")
                  logger.info(f" Early stop最大容忍轮次: {args['early_stopping_patience']}")
              else:
                if verbose:
                  logger.info(f" 达到Early stop最大容忍轮次.")
                  logger.info(" 停止训练@.")
                  train_iterator.close()
                return global_step, total_running_loss / global_step

    return global_step, total_running_loss / global_step

  def eval_model(self, eval_df, multi_label=False, output_dir=None, verbose=True, silent=False, **kwargs):
    """
    评估模型，确保在单GPU环境下运行。
    Args:
        eval_df: Pandas Dataframe. 至少含有两列'text'和'label'，表头可以不指定，或者三列'text_a'、‘text_b’,'label'.
        multi_label：是否是多种不同种类的标签。e.g [是否相信科学，是否吃过早饭, ...].
                     影响模型的loss计算方法，评价指标的设计。multi_label有两种情况，1. 每一种label为二分类， 2. 每一种label为多分类
                     e.g [相信科学的程度(1,2,3,4)，早饭饭量(1,2,3,4),...].
                     multi_label的训练有一点需要注意，不同label的隐含关系，可能有相关性。
                     根据实际情况修改模型的loss计算方法部分代码和评价指标的设计代码。
        output_dir: 保存结果的路径
        verbose: 控制台显示
        silent: 显示进度条.
        **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
    Returns:
        result: Dictionary -- evaluation results.
        model_outputs: List -- 输入每一个样例的预测结果.
        wrong_preds: List -- 错误预测样例.
    """
    if not output_dir:
      output_dir = self.args["output_dir"]

    self.model.to(self.device)
    result, preds, model_outputs, wrong_preds = self.evaluate(
        eval_df, output_dir, multi_label=multi_label, verbose=verbose, silent=silent, **kwargs)
    self.results.update(result)

    if verbose:
      logger.info(self.results)

    return result, model_outputs, wrong_preds

  def evaluate(self, eval_df, output_dir, multi_label=False, verbose=True, silent=False, **kwargs):
    """
    **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
    """
    device = self.device
    model = self.model
    args = self.args
    eval_output_dir = output_dir

    results = {}

    # ### 输入数据读取 ###
    if "text" in eval_df.columns and "label" in eval_df.columns:
      eval_examples = [
        InputExample(i, text, None, label)
        for i, (text, label) in enumerate(zip(eval_df["text"], eval_df["label"]))
        ]
    elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
      eval_examples = [
        InputExample(i, text_a, text_b, label)
        for i, (text_a, text_b, label) in enumerate(
            zip(eval_df["text_a"], eval_df["text_b"], eval_df["label"]))
        ]
    else:
      warnings.warn("未指定evaluate数据标题行, 自动使用第一行作为数据开始.")
      eval_examples = [
        InputExample(i, text, None, label)
        for i, (text, label) in enumerate(zip(eval_df.iloc[:, 0], eval_df.iloc[:, 1]))
        ]

    # ### load data ###
    if args["sliding_window"]:
      eval_dataset, window_counts = self.load_and_cache_examples(
          eval_examples, evaluate=True, verbose=verbose, silent=silent)
    else:
      eval_dataset = self.load_and_cache_examples(eval_examples, evaluate=True, verbose=verbose, silent=silent)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    os.makedirs(eval_output_dir, exist_ok=True)

    # ### Evaluate ###
    model.eval()
    for batch in tqdm(eval_dataloader, disable=args["silent"] or silent):
      batch = tuple(t.to(device) for t in batch)

      with torch.no_grad():
        inputs = self._get_inputs_dict(batch)
        outputs = model(**inputs)  # outputs： ((loss), logits, (hidden_states), (attentions))
        tmp_eval_loss, logits = outputs[:2]
        if multi_label:
          logits = logits.sigmoid()  # multi_label：转化为多个二分类，若每个类型的label有2个以上class，根据输出结构修改
        eval_loss += tmp_eval_loss.mean().item()
      nb_eval_steps += 1
      if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
      else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # 不是list的append
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    if args["sliding_window"]:
      count = 0
      window_ranges = []
      for n_windows in window_counts:  # window_counts：属于同一文档的sliding_window数目list，len(window_counts) == len(label)
        window_ranges.append([count, count + n_windows])
        count += n_windows
      # window_counts: [2, 3, 4]
      # window_ranges: [[0, 2], [2, 5], [5, 9]]
      # [window[0] for window in window_ranges]: [0, 2, 5]
      # out_label_ids: [1, 1, 0, 0, 0, 1, 1, 1, 1]
      # preds: ndarray[logits,logits, logits,logits,logits, logits,logits,logits,logits]
      # => preds: [ ndarray[logits,logits], ndarray[logits,logits,logits], ndarray[logits,logits,logits,logits] ]
      # preds： num_of_doc * sliding window * num_of_labels
      # logits: tensor [num_of_labels]
      preds = [preds[window[0]: window[1]] for window in window_ranges]  # 属于同一个文档的多个sliding_window输出
      out_label_ids = [
        out_label_ids[i] for i in range(len(out_label_ids)) if i in [window[0] for window in window_ranges]
        ]  # i: sliding_window起始index，找到label的起始位置
      model_outputs = preds
      preds = [np.argmax(pred, axis=1) for pred in preds]  # 每个sliding_window的预测标签
      final_preds = []
      for pred_row in preds:  # pred_row: sliding_window size，
        mode_pred, counts = sp.stats.mode(pred_row)  # scipy.stats, 返回众数和count
        if len(counts) > 1 and counts[0] == counts[1]:
          final_preds.append(args["tie_value"])  # 占位符
        else:
          final_preds.append(mode_pred[0])
      preds = np.array(final_preds)
    elif not multi_label and args["regression"] is True:
      preds = np.squeeze(preds)
      model_outputs = preds
    else:
      model_outputs = preds
      if not multi_label:
        preds = np.argmax(preds, axis=1)

    # 结果导出
    result, wrong = self.compute_metrics(preds, out_label_ids, eval_examples, **kwargs)
    result["eval_loss"] = eval_loss
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w", encoding="utf-8") as writer:
      for key in sorted(result.keys()):
        writer.write("{} = {}\n".format(key, str(result[key])))

    return results, preds, model_outputs, wrong

  def load_and_cache_examples(
      self, examples, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False
    ):
    """
    将InputExample转化为Dataset，可保存和读取处理后的InputFeatures. train() 和 eval() 的辅助方法.

    Returns:
        dataset -- InputExample的Dataset， (input_ids, input_mask, segment_ids, label_ids).
        window_counts -- 属于同一文档的sliding_window数目，list of Int. 如果设置了sliding_window为True.
    """
    args = self.args
    tokenizer = self.tokenizer

    process_count = args["process_count"]  # 调用的CPU数量
    no_cache = args["no_cache"]  # 是否启用特征缓存

    mode = "eval" if evaluate else "train"
    os.makedirs(self.args["cache_dir"], exist_ok=True)
    # 设置读取数据模式
    if not multi_label and args["regression"]:
      output_mode = "regression"
    else:
      output_mode = "classification"

    # 保存数据特征文件路径
    cached_features_file = os.path.join(
        args["cache_dir"],
        "cached_{}_{}_{}_{}_{}".format(
            mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples))
        )

    # reprocess_input_data：不使用处理好的特征，从输入数据路径再次处理特征。
    if os.path.exists(cached_features_file) and ((not args["reprocess_input_data"] and not no_cache)
        or (mode == "eval" and args["use_cached_eval_features"] and not no_cache)):
      with open(cached_features_file, 'rb') as f:
        features = pickle.load(f, encoding='utf-8')
      if verbose:
        logger.info(f">>> 加载缓存的特征文件 {cached_features_file}")
    else:
      if verbose:
        logger.info(f">>> 生成特征文件，未使用特征缓存文件.")
        if args["sliding_window"]:
          logger.info(">>> 已启用 Sliding window 读取输入文件")
      # 生成bert类模型输入特征格式
      features = convert_examples_to_features(
          examples,
          args["max_seq_length"],
          tokenizer,
          output_mode,
          # XLNet has a CLS token at the end
          cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
          cls_token=tokenizer.cls_token,
          cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
          sep_token=tokenizer.sep_token,
          # RoBERTa uses an extra separator b/w pairs of sentences,
          # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
          sep_token_extra=bool(args["model_type"] in ["roberta", "camembert", "xlmroberta"]),
          # PAD on the left for XLNet
          pad_on_left=bool(args["model_type"] in ["xlnet"]),
          pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
          pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
          process_count=process_count,
          multi_label=multi_label,
          silent=args["silent"] or silent,
          use_multiprocessing=args["use_multiprocessing"],
          sliding_window=args["sliding_window"],
          flatten=not evaluate,
          stride=args["stride"],
          args=args,
          )
      if verbose and args["sliding_window"]:
        logger.info(f" {len(features)} features created from {len(examples)} samples.")
      if not no_cache:
        with open(cached_features_file, 'wb') as f:
          pickle.dump(features, f)

    if args["sliding_window"] and evaluate:
      window_counts = [len(sample) for sample in features]  # window_counts：属于同一文档的sliding_window数目，列表
      features = [feature for feature_set in features for feature in feature_set]  # 一篇文档，多段token

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
      all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
      all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if args["sliding_window"] and evaluate:
      return dataset, window_counts
    else:
      return dataset

  def compute_metrics(self, preds, labels, eval_examples, multi_label=False, **kwargs):
    """
    计算评估指标. 根据实际需要更改.
    TODO: 修改评价指标时，修改该函数。结合实际情况直接重写，相关函数： _create_training_progress_scores
    Args:
        eval_examples: 传入样本，以输出错误预测样例。
        **kwargs: metric 函数对象. 比如传入 f1=sklearn.metrics.f1_score。函数要求接受pretict和true两个值。
    Returns:
        result: Dictionary . (Matthews correlation coefficient, tp, tn, fp, fn, ...)
        wrong: 错误预测样例
    """

    assert len(preds) == len(labels)

    extra_metrics = {}
    for metric, func in kwargs.items():
      extra_metrics[metric] = func(labels, preds)
    # 误分类
    mismatched = labels != preds
    wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]

    # 若多便签，每个label有多个class，每个标签可分别按照多分类进行评价。该函数自定义修改
    if multi_label:  # 多标签
      label_ranking_score = label_ranking_average_precision_score(labels, preds)
      return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong
    elif self.args["regression"]:
      return {**extra_metrics}, wrong

    # 不是多标签的情况
    mcc = matthews_corrcoef(labels, preds)
    if self.model.num_labels == 2:  # 二分类
      tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
      return (
        {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
        wrong,
        )
    elif self.model.num_labels > 2:  # 多分类
      confusion_mat = confusion_matrix(labels, preds)
      return {**{"mcc": mcc}, **extra_metrics, "confusion matrix": confusion_mat}, wrong
    else:  # 只输出一个值的情况
      return {**{"mcc": mcc}, **extra_metrics}, wrong

  def predict(self, to_predict, multi_label=False):
    """
    Inference.
    Args:
        to_predict: 预测文本 list
    Returns:
        preds: list.
        model_outputs: 模型raw outputs.
    """
    device = self.device
    model = self.model
    args = self.args

    self.model.to(self.device)

    if multi_label:
      eval_examples = [
        InputExample(i, text, None, [0 for _ in range(self.num_labels)]) for i, text in enumerate(to_predict)
        ]
    else:
      if isinstance(to_predict[0], list):
        eval_examples = [InputExample(i, text[0], text[1], 0) for i, text in enumerate(to_predict)]
      else:
        eval_examples = [InputExample(i, text, None, 0) for i, text in enumerate(to_predict)]
    if args["sliding_window"]:
      eval_dataset, window_counts = self.load_and_cache_examples(eval_examples, evaluate=True, no_cache=True)
    else:
      eval_dataset = self.load_and_cache_examples(
          eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
          )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # PretrainedConfig对象的属性
    if self.config.output_hidden_states:
      for batch in tqdm(eval_dataloader, disable=args["silent"]):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
          inputs = self._get_inputs_dict(batch)
          outputs = model(**inputs)
          tmp_eval_loss, logits = outputs[:2]  # outputs： ((loss), logits, (hidden_states), (attentions))
          embedding_outputs, layer_hidden_states = outputs[2][0], outputs[2][1:]
          if multi_label:
            logits = logits.sigmoid()
          eval_loss += tmp_eval_loss.mean().item()
        # layer_hidden_states： the output of each layer， each is (batch_size, sequence_length, hidden_size)
        # embedding_outputs： the initial embedding outputs， (batch_size, sequence_length, hidden_size)
        nb_eval_steps += 1
        if preds is None:
          preds = logits.detach().cpu().numpy()
          out_label_ids = inputs["weight"].detach().cpu().numpy()
          all_layer_hidden_states = [state.detach().cpu().numpy() for state in layer_hidden_states]
          all_embedding_outputs = embedding_outputs.detach().cpu().numpy()
        else:
          preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
          out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
          all_layer_hidden_states = np.append(
              [state.detach().cpu().numpy() for state in layer_hidden_states], axis=0
              )
          all_embedding_outputs = np.append(embedding_outputs.detach().cpu().numpy(), axis=0)
    else:
      # 与evaluate函数相似
      for batch in tqdm(eval_dataloader, disable=args["silent"]):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
          inputs = self._get_inputs_dict(batch)
          outputs = model(**inputs)
          tmp_eval_loss, logits = outputs[:2]
          if multi_label:
            logits = logits.sigmoid()
          eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
          preds = logits.detach().cpu().numpy()
          out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
          preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
          out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # 与evaluate函数相似
    if args["sliding_window"]:
      count = 0
      window_ranges = []
      for n_windows in window_counts:
        window_ranges.append([count, count + n_windows])
        count += n_windows
      preds = [preds[window_range[0] : window_range[1]] for window_range in window_ranges]
      model_outputs = preds
      preds = [np.argmax(pred, axis=1) for pred in preds]
      final_preds = []
      for pred_row in preds:
        mode_pred, counts = mode(pred_row)
        if len(counts) > 1 and counts[0] == counts[1]:
          final_preds.append(args["tie_value"])
        else:
          final_preds.append(mode_pred[0])
      preds = np.array(final_preds)
    elif not multi_label and args["regression"] is True:
      preds = np.squeeze(preds)
      model_outputs = preds
    else:
      model_outputs = preds
      if multi_label:
        if isinstance(args["threshold"], list):
          threshold_values = args["threshold"]
          preds = [[self._threshold(pred, threshold_values[i]) for i, pred in enumerate(example)] for example in preds]
        else:
          preds = [[self._threshold(pred, args["threshold"]) for pred in example] for example in preds]
      else:
        preds = np.argmax(preds, axis=1)

    if self.config.output_hidden_states:
      return preds, model_outputs, all_embedding_outputs, all_layer_hidden_states
    else:
      return preds, model_outputs

  def _threshold(self, x, threshold):
    """threshold, float 或者 一个list"""
    if x >= threshold:
      return 1
    return 0

  def _get_inputs_dict(self, batch):
    """根据model_type，处理输入数据结构
    XLM, DistilBERT and RoBERTa 没有使用 segment_ids(token_type_ids)."""
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    if self.args["model_type"] != "distilbert":
      if self.args["model_type"] in ["bert", "xlnet", "albert"]:
        inputs["token_type_ids"] = batch[2]
      else:
        inputs["token_type_ids"] = None
    return inputs

  def _create_training_progress_scores(self, multi_label, **kwargs):
    """TODO: 修改评价指标时，修改该函数。结合实际情况直接重写，相关函数：compute_metrics"""
    extra_metrics = {key: [] for key in kwargs}
    if multi_label:
      training_progress_scores = {
        "global_step": [], "LRAP": [], "train_loss": [], "eval_loss": [], **extra_metrics, }
    else:
      if self.model.num_labels == 2:
        training_progress_scores = {
          "global_step": [],"tp": [],"tn": [],"fp": [],"fn": [],"mcc": [],"train_loss": [],
          "eval_loss": [],**extra_metrics,}
      elif self.model.num_labels > 2:
        training_progress_scores = {
          "global_step": [], "confusion_matrix": [], "mcc": [], "train_loss": [],
          "eval_loss": [],**extra_metrics,}
      elif self.model.num_labels == 1:
        training_progress_scores = {
          "global_step": [], "train_loss": [], "eval_loss": [], **extra_metrics, }
      else:
        training_progress_scores = {
          "global_step": [], "mcc": [], "train_loss": [], "eval_loss": [], **extra_metrics, }
    return training_progress_scores

  def _save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
    """保存传入的对象参数"""
    if not output_dir:
      output_dir = self.args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if model and not self.args["no_save"]:
      # 多GPU训练时，DataParallel包装的模型在保存时，权值参数前面会带有module字符.
      # 为了在单卡环境下可以加载模型，需要以下操作。
      model_to_save = model.module if hasattr(model, "module") else model
      model_to_save.save_pretrained(output_dir)
      self.tokenizer.save_pretrained(output_dir)
      torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
      # 可选择保存优化器状态
      if optimizer and scheduler:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

      # 保存全局参数设置为json，方便查看
      os.makedirs(output_dir, exist_ok=True)
      with open(os.path.join(output_dir, "model_args.json"), "w", encoding='utf-8') as f:
        json.dump(self.args, f, ensure_ascii=False)

    # 保存evaluate的结果
    if results:
      output_eval_file = os.path.join(output_dir, "eval_results.txt")
      with open(output_eval_file, "w", encoding="utf-8") as writer:
        for key in sorted(results.keys()):
          writer.write("{} = {}\n".format(key, str(results[key])))

  def _load_model_args(self, input_dir):
    model_args_file = os.path.join(input_dir, "model_args.json")
    if os.path.isfile(model_args_file):
      with open(model_args_file, "r", encoding='utf-8') as f:
        model_args = json.load(f)
      return model_args
