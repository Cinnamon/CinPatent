export CUDA_VISBLE_DEVICES=0

DATA_PREFIX='CinPatent-JA/ja_0.05'
DATA_DIR=datasets/$DATA_PREFIX

python main.py \
    --data_dir $DATA_DIR \
    --train_file train_data.json \
    --val_file val_data.json \
    --test_file test_data.json \
    --vocab_file w2idx.json \
    --embed_dim 300 \
    --do_train \
    --do_predict \
    --batch_size 32 \
    --epochs 20 \
    --max_length 512 \
    --lr 2e-3 \
    --config_dir ./config \
    --model_dir model \
    --log_dir tmp/log/