# Parabel

Original implementation: https://github.com/tomtung/omikuji

## How to run

0. Install package
```bash
pip install -e .
```

1. Run `prepare_data.sh` to create dataset
```bash
DATA_PREFIX='CinPatent-EN/en_0.05'

python prepare_data.py \
    --data_path ../../../data/${DATA_PREFIX}.ndjson \
    --output_dir datasets/${DATA_PREFIX} \
    --lang en
```

2. Run `train.sh` to train Parabel on specified dataset
```
DATA_PREFIX='CinPatent-EN/en_0.05'

python train.py \
    --data_dir datasets/${DATA_PREFIX} \
    --model_dir model/${DATA_PREFIX}
```

**Note**: You can change default hyperparameters in `train.py`.
