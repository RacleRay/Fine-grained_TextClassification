#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_multi_class_cnn.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import pandas as pd
from xbert_cnn_runner import MultiClassCnnRunner


def main():
    train_csv = "path"
    eval_csv = "path"

    # label is the class representing Int
    train_df = pd.read_csv(train_csv, names=['label', 'text'], encoding='utf-8')
    eval_df = process_tsv(eval_csv, names=['label', 'text'], encoding='utf-8')

    # num_labels = number of classes
    model = MultiBinaryClaRunner('albert', './albert_tiny/', num_labels=5, freez_pretrained=False,
        args={"reprocess_input_data": True, "overwrite_output_dir": True, "num_train_epochs": 5})

    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(result)
    print(model_outputs)

    predictions, raw_outputs = model.predict(["This thing is entirely different from the other thing. "])
    print(predictions)
    print(raw_outputs)


if __name__=="__main__":
    main()