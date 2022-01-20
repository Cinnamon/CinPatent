DATA_PREFIX='CinPatent-EN/en_0.05'

python train.py \
    --model fastxml \
    --data_dir datasets/${DATA_PREFIX} \
    --model_dir model/fastxml/${DATA_PREFIX}
