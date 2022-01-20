# FastXML/PfastXML

Original implementation: https://github.com/Refefer/fastxml

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

2. Run `train_fastxml.sh` to train FastXML on specified dataset or `train_pfastxml.sh` for PfastXML.
```bash
DATA_PREFIX='CinPatent-EN/en_0.05'

python train.py \
    --model fastxml \
    --data_dir datasets/${DATA_PREFIX} \
    --model_dir fastxml/${DATA_PREFIX}
```

Classification results are saved in provided `model_dir`.

**Note**: You can change default hyperparameters in `train.py`.