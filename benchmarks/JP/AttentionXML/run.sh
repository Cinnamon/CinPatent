#!/bin/sh

raw_path=$1

echo "Installing dependencies..."
pip install -r requirements.txt
pip install mecab-python3
pip install unidic-lite

python prepare_data.py --data_path $raw_path --max_len 512 --storage_dir ./data/EUR-Lex
echo ""
echo "Preprocessing data..."
python preprocess.py --text-path ./data/EUR-Lex/train_texts.txt --label-path ./data/EUR-Lex/train_labels.txt --vocab-path ./data/EUR-Lex/vocab.npy --emb-path ./data/EUR-Lex/emb_init.npy --w2v-model data/glove.840B.300d.gensim
python preprocess.py --text-path ./data/EUR-Lex/test_texts.txt --label-path ./data/EUR-Lex/test_labels.txt --vocab-path ./data/EUR-Lex/vocab.npy
echo ""
echo "Training model..."
python main.py --data-cnf configure/datasets/EUR-Lex.yaml --model-cnf configure/models/AttentionXML-EUR-Lex.yaml
echo ""
echo "Evaluating model..."
python evaluation.py --results results/AttentionXML-EUR-Lex-labels.npy --targets ./data/EUR-Lex/test_labels.npy
