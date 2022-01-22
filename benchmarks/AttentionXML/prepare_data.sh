DATA_PREFIX='CinPatent-EN/en_0.05'

python prepare_data.py \
    --data_path ../../../data/${DATA_PREFIX}.ndjson \
    --output_dir datasets/${DATA_PREFIX} \
    --lang en
