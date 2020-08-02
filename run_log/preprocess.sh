#!/bin/bash

# Modify the following values depend on your environment
# Path to the csv files
TRAIN_FILE=data/data_aug.csv
VALIDATION_FILE=../../data/validationset/sentiment_analysis_validationset.csv
TESTA_FILE=../../data/testa/sentiment_analysis_testa.csv
TESTB_FILE=../../data/testb/sentiment_analysis_testb.csv

# Path to pretrained embedding file
EMBEDDING_FILE=../fsauor2018/wordvec/sgns.sogou.word

VOCAB_SIZE=50000

# Create a folder to save training files
mkdir -p data

echo 'Process training file ...'
python data_process.py \
    --data_file=$TRAIN_FILE \
    --output_file=data/train.json \
    --vocab_file=data/vocab.txt \
	--is_trian_file=True \
    --vocab_size=$VOCAB_SIZE

echo 'Process validation file ...'
python data_process.py \
    --data_file=$VALIDATION_FILE \
    --output_file=data/validation.json

echo 'Process testa file ...'
python data_process.py \
    --data_file=$TESTA_FILE \
    --output_file=data/testa.json

# Uncomment following code to get testb file
# echo 'Process testb file ...'
# python data_preprocess.py \
#     --data_file=$TESTB_FILE \
#     --output_file=data/testb.json

echo 'Get pretrained embedding ...'
python data_process.py \
    --data_file=$EMBEDDING_FILE \
    --output_file=data/embedding.txt \
    --vocab_file=data/vocab.txt \
    --embedding=True

echo "Get label file ..."
cp ../labels.txt data/labels.txt