#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import json
import time
import os

import numpy as np
import tensorflow as tf

from dataset import DataSet
from elmo import Model
from elmo_utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 忽略警告
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # mode
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")

    # data
    parser.add_argument("--data_files", type=str, nargs='+', default=None, help="data file for train or inference")
    parser.add_argument("--eval_files", type=str, nargs='+', default=None, help="eval data file for evaluation")
    parser.add_argument("--label_file", type=str, default=None, help="label file")
    parser.add_argument("--vocab_file", type=str, default=None, help="vocab file")
    parser.add_argument("--embed_file", type=str, default=None, help="embedding file to restore")
    parser.add_argument("--out_file", type=str, default=None, help="output file for inference")
    parser.add_argument("--split_word", type='bool', nargs="?", const=True, default=True, help="Whether to split word when oov")
    parser.add_argument("--reverse", type='bool', nargs="?", const=True, default=False, help="Whether to reverse data")
    parser.add_argument("--weight_file", type=str, nargs="?", default=None, help="class prediction weights.")
    parser.add_argument("--prob", type='bool', nargs="?", const=True, default=False, help="Whether to export prob")

    # model
    parser.add_argument("--max_len", type=int, default=1200, help='max length for doc')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")

    parser.add_argument("--optimizer", type=str, default='RMS', help="Optimizer: RMS or Adam")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. RMS: 0.001 | 0.0001")
    parser.add_argument("--decay_schema", type=str, default='hand', help = 'learning rate decay: exp | hand')
    parser.add_argument("--decay_steps", type=int, default=10000, help="decay steps")

    parser.add_argument("--loss_name", type=str, default='softmax', help="loss type")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="gamma of focal loss")
    parser.add_argument("--max_gradient_norm", type=float, default=2.0, help="Clip gradients to this norm.")
    parser.add_argument("--l2_loss_ratio", type=float, default=0.0, help="l2 loss ratio")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="label smoothing param")

    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="embedding_dropout alone seq len dim")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="drop out keep ratio for decoder, embedding size dim")
    parser.add_argument("--weight_keep_drop", type=float, default=0.8, help="weight keep drop, for WeightDropLSTMCell")
    parser.add_argument("--linear_dropout", type=float, default=0.2, help="weight dropout, for linear clf")

    parser.add_argument("--rnn_cell_name", type=str, default='lstm', help = 'rnn cell name for decoder')
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding_size")
    parser.add_argument("--num_units", type=int, default=300, help="num_units for all rnn cells")

    # clf
    parser.add_argument("--num_classes_each_label", type=int, default=4, help="num_classes_each_label")
    parser.add_argument("--num_labels", type=int, default=20, help="num_labels")

    # train
    parser.add_argument("--fix_embedding", type='bool', nargs="?", const=True, default=False, help="Whether to fix embedding")
    parser.add_argument("--need_early_stop", type='bool', nargs="?", const=True, default=True, help="Whether to early stop")
    parser.add_argument("--patient", type=int, default=5, help="patient of early stop")
    parser.add_argument("--debug", type='bool', nargs="?", const=True, default=False, help="Whether use debug mode")
    parser.add_argument("--num_train_epoch", type=int, default=50, help="training epoches")
    parser.add_argument("--steps_per_stats", type=int, default=20, help="steps to print stats")
    parser.add_argument("--steps_per_summary", type=int, default=50, help="steps to save summary")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="steps to save model")

    parser.add_argument("--checkpoint_dir", type=str, default='/tmp/', help="checkpoint dir to save model")
    parser.add_argument("--checkpoint_load_step", type=int, default=None, help="global step for loading the specific model")
    parser.add_argument("--previous_best_eval", type=float, default=10000.0, help="current best eval score, for special training task")


def convert_to_config(params):
    config = tf.contrib.training.HParams()
    for k,v in params.items():
        config.add_hparam(k,v)
    return config


def train_eval_clf(model, sess, dataset):
    "模型评估"
    from collections import defaultdict
    checkpoint_loss, acc = 0.0, 0.0
    predicts, truths = defaultdict(list), defaultdict(list)
    for i,(source, lengths, targets, _) in enumerate(dataset.get_next(shuffle=False)):
        batch_loss, accuracy, batch_size, predict = model.eval_clf_one_step(sess,
                                                                            source,
                                                                            lengths,
                                                                            targets)
        # predict： batch * 20 * 4
        for i, p in enumerate(predict):
            for j in range(model.config.num_labels):
                label_name = dataset.i2l[j]
                truths[label_name].append(targets[i][j])
                predicts[label_name].append(p[j])
        checkpoint_loss += batch_loss
        acc += accuracy
        if (i+1) % 100 == 0:
            print("=>> batch %d/%d" %(i+1,dataset.num_batches))

    results = {}
    total_f1 = 0.0
    for label_name in dataset.label_names:
        f1, precision, recall = cal_f1(model.config.num_classes_each_label,
                                       np.asarray(predicts[label_name]),
                                       np.asarray(truths[label_name]))
        results[label_name] = f1
        total_f1 += f1
        print("=>> {0} - {1}".format(label_name,f1))

    final_f1 = total_f1 / len(results)

    print("=>> Eval loss %.5f, f1 %.5f" % (checkpoint_loss / i, final_f1))
    return -1 * final_f1, checkpoint_loss / i


def train_clf(flags):
    dataset = DataSet(flags.data_files,
                      flags.vocab_file,
                      flags.label_file,
                      flags.batch_size,
                      reverse=flags.reverse,
                      split_word=flags.split_word,
                      max_len=flags.max_len)
    eval_dataset = DataSet(flags.eval_files,
                           flags.vocab_file,
                           flags.label_file,
                           2 * flags.batch_size,
                           reverse=flags.reverse,
                           split_word=flags.split_word,
                           max_len=flags.max_len)

    params = vars(flags) # equivalent to object.dict
    params['vocab_size'] = len(dataset.w2i)
    config = convert_to_config(params)

    save_config(flags.checkpoint_dir, config)
    print(config)

    # Graph
    train_graph = tf.Graph()
    eval_graph = tf.Graph()
    with train_graph.as_default():
        train_model = Model(config)
        train_model.build()
        initializer = tf.global_variables_initializer()
    with eval_graph.as_default():
        eval_config = load_config(flags.checkpoint_dir,
                                  {"mode":'eval','checkpoint_dir':flags.checkpoint_dir+"/best_eval"})
        eval_model = Model(eval_config)
        eval_model.build()

    # Sess
    train_sess = tf.Session(graph=train_graph,
                            config=get_config_proto(log_device_placement=False))
    train_model.init_model(train_sess, initializer=initializer)
    try:
        if flags.checkpoint_load_step is not None:
            train_model.restore_model(train_sess, flags.checkpoint_load_step)
        else:
            train_model.restore_model(train_sess)  # lastest
    except:
        print("!!! Unable to restore model, train from scratch !!!")

    # start training
    print("=>> Start to train with learning rate {}".format(flags.learning_rate))
    # 手动设置checkpoint继续训练时的初始learning_rate
    def_lr = tf.assign(train_model.learning_rate, flags.learning_rate)
    train_sess.run(def_lr)
    global_step = train_sess.run(train_model.global_step)
    print("=>> Global step", global_step)

    eval_ppls = []  # -final_f1
    best_eval = flags.previous_best_eval
    pre_best_checkpoint = None
    final_learn = 2
    for epoch in range(flags.num_train_epoch):
        step_time, checkpoint_loss, acc, iters = 0.0, 0.0, 0.0, 0
        for i,(source, lengths, targets, _) in enumerate(dataset.get_next()):
            # train
            start_time = time.time()
            add_summary = (global_step % flags.steps_per_summary == 0)
            batch_loss, global_step, accuracy, token_num, batch_size = train_model.train_clf_one_step(
                train_sess,
                source,
                lengths,
                targets,
                add_summary=add_summary,
                run_info=add_summary and flags.debug
            )
            step_time += (time.time() - start_time)
            checkpoint_loss += batch_loss
            acc += accuracy
            iters += token_num

            if global_step == 0:
                continue
            # log
            if global_step % flags.steps_per_stats == 0:
                train_acc = (acc / flags.steps_per_stats) * 100
                acc_summary = tf.Summary()
                acc_summary.value.add(tag='accuracy', simple_value=train_acc)
                train_model.summary_writer.add_summary(acc_summary, global_step=global_step)
                print(
                    "=>> Epoch %d  global step %d loss %.5f batch %d/%d lr %g "
                    "accuracy %.5f wps %.2f step time %.2fs" % (
                            epoch + 1,
                            global_step,
                            checkpoint_loss / flags.steps_per_stats,
                            i + 1,
                            dataset.num_batches,
                            train_model.learning_rate.eval(session=train_sess),
                            train_acc,
                            (iters) / step_time,
                            step_time / (flags.steps_per_stats)
                        )
                    )
                step_time, checkpoint_loss, iters, acc = 0.0, 0.0, 0, 0.0

            # eval
            if global_step % flags.steps_per_eval == 0:
                print("=>> global step {0}, eval result: ".format(global_step))
                checkpoint_path = train_model.save_model(train_sess)
                with tf.Session(graph=eval_graph, config=get_config_proto(log_device_placement=False)) as eval_sess:
                    eval_model.init_model(eval_sess)
                    eval_model.restore_ema_model(eval_sess, checkpoint_path)
                    # eval_model.restore_model(eval_sess)

                    dropout_keep_prob = tf.assign(eval_model.dropout_keep_prob, 1.0)
                    linear_dropout = tf.assign(eval_model.linear_dropout, 0.0)
                    emd_drop = tf.assign(eval_model.embedding_dropout, 0.0)
                    eval_sess.run([dropout_keep_prob, linear_dropout, emd_drop])

                    # 最小化 -f1 为评价标准
                    eval_ppl, eval_loss = train_eval_clf(eval_model, eval_sess, eval_dataset)

                    print("=>> current result {0}, previous best result {1}".format(eval_ppl, best_eval))
                    loss_summary = tf.Summary()
                    loss_summary.value.add(tag='eval_loss', simple_value = eval_loss)
                    train_model.summary_writer.add_summary(loss_summary, global_step=global_step)

                    if eval_ppl < best_eval:
                        pre_best_checkpoint = checkpoint_path
                        eval_model.save_model(eval_sess, global_step)
                        best_eval = eval_ppl
                    eval_ppls.append(eval_ppl)

                if flags.need_early_stop:
                    if early_stop(eval_ppls, flags.patient):
                        print("=>> No loss decrease, restore previous best model and set learning rate to half of previous one")
                        current_lr = train_model.learning_rate.eval(session=train_sess)

                        if final_learn > 0:
                            final_learn -= 1
                        else:
                            print("=>> Early stop, exit")
                            exit(0)

                        # 当early_stop时，若final_learn不为0，继续减小学习率，从前一个最优epoch继续训练
                        train_model.saver.restore(train_sess, pre_best_checkpoint)
                        lr = tf.assign(train_model.learning_rate, current_lr / 10)
                        # 最后一轮final_learn不设置dropout
                        if final_learn == 0:
                            dropout_keep_prob = tf.assign(train_model.dropout_keep_prob, 1.0)
                            linear_dropout = tf.assign(train_model.linear_dropout, 0.0)
                            emd_drop = tf.assign(train_model.embedding_dropout, 0.0)
                            train_sess.run([dropout_keep_prob, linear_dropout, emd_drop])
                        train_sess.run(lr)
                        eval_ppls = [best_eval]
                        continue
        print("=>> Finsh epoch {1}, global step {0}".format(global_step, epoch+1))
    print("=>> Best accuracy {0}".format(best_eval))


def inference(flags):
    print("inference data file {0}".format(flags.data_files))
    dataset = DataSet(flags.data_files,
                      flags.vocab_file,
                      flags.label_file,
                      flags.batch_size,
                      reverse=flags.reverse,
                      split_word=flags.split_word,
                      max_len=flags.max_len)
    config = load_config(flags.checkpoint_dir,
                         {
                             'mode': 'inference',
                             'checkpoint_dir': flags.checkpoint_dir+"/best_eval",
                             'embed_file': None
                         })
    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = Model(config)
        model.build()

        try:
            if flags.checkpoint_load_step is not None:
                model.restore_model(sess, flags.checkpoint_load_step)
            else:
                model.restore_model(sess)  # lastest
        except Exception as e:
            print("unable to restore model with exception",e)
            exit(1)

        scalars = model.scalars.eval(session=sess)
        print("Scalars:", scalars)
        weight = model.weight.eval(session=sess)
        print("Weight:",weight)
        count = 0
        for (source, lengths, _, _) in dataset.get_next(shuffle=False):
            predict, logits = model.inference_clf_one_batch(sess, source, lengths)
            probs = tf.nn.softmax(logits)
            for i, (p, l) in enumerate(zip(predict, probs)):
                for j in range(flags.num_labels):
                    label_name = dataset.i2l[j]
                    if flags.prob:
                        tag =  [float(v) for v in l[j]]
                    else:
                        tag = dataset.tag_i2l[np.argmax(p[j])]
                    dataset.items[count + i][label_name] = tag
            count += len(lengths)
            print("\r# process {0:.2%}".format(count / dataset.data_size), new_line=False)

    print("=>> Write result to file ...")
    with open(flags.out_file,'w') as f:
        for item in dataset.items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("=>> Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    if flags.mode == 'train':
        train_clf(flags)
    elif flags.mode == 'inference':
        inference(flags)