DATA_PREFIX='CinPatent-EN/en_0.05'

python prepare_data.py \
    --data_path ../../../../data/$DATA_PREFIX.ndjson \
    --max_len 512 \
    --storage_dir datasets/$DATA_PREFIX