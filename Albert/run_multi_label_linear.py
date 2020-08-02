#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from xbert_multi_label_linear_runner import MultiBinaryClaRunner


def process_tsv(file):
    "根据数据的储存方式修改"
    data = pd.read_table(file, names=['label', 'text'], encoding='utf-8')
    label = data.label.apply(lambda x: x.split('@'))
    label = np.array(label.to_list())
    enc = OneHotEncoder()
    # '0_-1', '0_-2', '0_0', '0_1' => 1000, 0100, 0010, 0001
    label_ont_hot = enc.fit_transform(label)
    for i, row_label in enumerate(label_ont_hot.toarray()):
        data.iloc[i]['label'] = row_label
    return data


def main():
    train_df = process_tsv('train.tsv')
    eval_df = process_tsv('val.tsv')

    # num_labels = kind of labels * classes of each label
    model = MultiBinaryClaRunner('albert',
                                 './albert_large/',
                                 num_labels=80,
                                 freez_pretrained=False,
                                 args={
                                     "reprocess_input_data": False,
                                     "overwrite_output_dir": False,
                                     "num_train_epochs": 5
                                 })

    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)
    print(model_outputs)

    predictions, raw_outputs = model.predict(
        ["This thing is entirely different from the other thing. "])
    print(predictions)
    print(raw_outputs)


if __name__ == "__main__":
    main()
