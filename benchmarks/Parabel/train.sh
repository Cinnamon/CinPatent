DATA_PREFIX='CinPatent-EN/en_0.05'

python train.py \
    --data_dir datasets/${DATA_PREFIX} \
    --model_dir model/${DATA_PREFIX}