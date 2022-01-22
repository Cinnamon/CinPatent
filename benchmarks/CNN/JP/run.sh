#!/bin/sh

# DIR=$1
path=$1
STORAGE_DIR="./data"

# for path in "$DIR"/*.ndjson
# do
echo "$path"
    python3 prepare_data.py --data_path $path --max_len 768 --storage_dir $STORAGE_DIR
echo ""
echo "Training and evaluating model.."
python train.py --train_data $STORAGE_DIR/train_data.json --val_data $STORAGE_DIR/val_data.json --log ./log --batch_size 32 --epochs 20 --max_length 512 --checkpoint ./checkpoint --lr 2e-3 --vocab_file ./data/w2idx.json --config ./config/
echo ""
# done