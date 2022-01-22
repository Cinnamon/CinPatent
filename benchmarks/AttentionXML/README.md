# AttentionXML

Original implementation: https://github.com/yourh/AttentionXML

## How to run

0. Install dependencies
```
pip install -r requirements.txt
```

1. Run `prepare_data.sh` to prepare training datasets in texts
```bash
python prepare_data.py \
    --data_path ../../../data/${DATA_PREFIX}.ndjson \
    --output_dir datasets/${DATA_PREFIX} \
    --lang en
```

2. Run `create_features.sh` to generate training features and encode labels
```bash
python preprocess.py \
    --text-path $DATA_DIR/train_texts.txt \
    --label-path $DATA_DIR/train_labels.txt \
    --vocab-path $DATA_DIR/vocab.npy \
    --emb-path $DATA_DIR/emb_init.npy \
    --w2v-model tmp/glove.840B.300d.kv

python preprocess.py \
    --text-path $DATA_DIR/test_texts.txt \
    --label-path $DATA_DIR/test_labels.txt \
    --vocab-path $DATA_DIR/vocab.npy
```

3. Modify configurations in `config.yaml` and run `train.sh` to start training on specified data
```bash
python main.py \
    --config_file config_en.yaml \
    --do_train \
    --do_predict
```