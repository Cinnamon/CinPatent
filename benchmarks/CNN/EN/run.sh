#!/bin/sh

echo "Installing dependencies..."
pip install -r requirements.txt

raw_path=$1
storage_path="./data"

echo "Raw data path: $path"
    python3 prepare_data.py --data_path $path --max_len 512 --storage_dir $storage_path
echo ""
echo "Training and evaluating model.."
python train.py --train_data $storage_path/train_data.json --val_data $storage_path/val_data.json --log ./log --batch_size 32 --epochs 20 --max_length 512 --checkpoint ./checkpoint --lr 3e-3 --fasttext ./data/fasttext.vec --config ./config/
