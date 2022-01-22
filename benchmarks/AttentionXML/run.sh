export CUDA_VISIBLE_DEVICES=3

python main.py \
    --config_file config_ja.yaml \
    --do_train \
    --do_predict
