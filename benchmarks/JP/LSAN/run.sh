#!/bin/sh

# pip install mxnet
path=$1

echo "$path"
python prepare_data.py --data_path $path --max_len 512 --storage_dir ./data/
echo ""
echo "Preprocessing data..."
python convert_to_seq.py --root_dir ./data --max_seq_len 512
echo ""
echo "Training and evaluating model.."
python classification.py
