## CNN

### How to run

0. Install dependencies
```bash
pip install -r requirements.txt
```

1. Change directory to target language (`EN` for English or `JA` for Japanse)
```bash
cd EN/
```

2. Run `prepare_data.sh` to prepare data for training and
```bash
python prepare_data.py \
    --data_path ../../../../data/$DATA_PREFIX.ndjson \
    --max_len 512 \
    --storage_dir datasets/$DATA_PREFIX
```

3. Run `run.sh` to start training on specified dataset
```bash
python main.py \
    --data_dir $DATA_DIR \
    --train_file train_data.json \
    --val_file val_data.json \
    --test_file test_data.json \
    --do_predict \
    --batch_size 32 \
    --epochs 20 \
    --max_length 512 \
    --lr 3e-3 \
    --config_dir ./config \
    --model_dir model \
    --log_dir tmp/log/
```