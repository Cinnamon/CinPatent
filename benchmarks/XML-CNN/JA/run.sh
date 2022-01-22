#!/bin/sh
path=$1

echo "Preparing data: $path"
python3 prepare_data.py --raw_data_path $path --max_len 512 --storage_dir ./data --split_val 1
# python3 $STORAGE_DIR/make_valid.py $STORAGE_DIR/train_org.txt
echo ""
echo "Training and evaluating model.."
python3 train.py
echo ""
done
