#!/bin/sh

# pip install mxnet
DIR=$1
STORAGE_DIR="./data"

for path in "$DIR"/*.ndjson
do
    echo "Preparing data: $path"
    python prepare_data.py --raw_data_path $path --max_len 768 --storage_dir $STORAGE_DIR
    echo ""
    echo "Preprocessing data..."
    python ./data/make_valid.py ./data/train_org.txt
    echo ""
    echo "Training and evaluating model.."
    python train.py
    echo ""
done
