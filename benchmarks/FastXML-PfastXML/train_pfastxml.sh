DATA_PREFIX='CinPatent-EN/en_0.05'

python train.py \
    --model pfastxml \
    --data_dir datasets/${DATA_PREFIX}/ \
    --model_dir model/pfastxml/${DATA_PREFIX}
