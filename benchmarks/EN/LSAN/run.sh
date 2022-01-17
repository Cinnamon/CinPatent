#!/bin/sh

# pip install mxnet
DIR=$1
STORAGE_DIR="./data"

for path in "$DIR"/*.ndjson
do
    echo "$path"
    python prepare_data.py --data_path $path --max_len 768 --storage_dir $STORAGE_DIR
    echo ""
    echo "Preprocessing data..."
    python convert_to_npy.py --root_dir $STORAGE_DIR --max_seq_len 512
    echo ""
    echo "Training and evaluating model.."
    python classification.py
    echo ""
done