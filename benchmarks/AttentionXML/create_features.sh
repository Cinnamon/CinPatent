DATA_PREFIX='CinPatent-JA/ja_0.05'
DATA_DIR=datasets/$DATA_PREFIX

echo 'Checking word embedding...'
mkdir -p tmp/
cd tmp/
if [ ! -f glove.840B.300d.kv ]; then
    echo 'Downloading GloVe...'
    wget -nc https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip
    unzip -n glove.840B.300d.zip
    echo 'Converting GloVe to Gensim KeyVectored format...'
    python ../glove2kv.py -i glove.840B.300d.txt -o glove.840B.300d.kv && rm glove.840B.300d.txt
fi
cd ../

echo 'Creating training features...'
python preprocess.py \
    --text-path $DATA_DIR/train_texts.txt \
    --label-path $DATA_DIR/train_labels.txt \
    --vocab-path $DATA_DIR/vocab.npy \
    --emb-path $DATA_DIR/emb_init.npy \
    --w2v-model tmp/glove.840B.300d.kv

echo 'Creating validation features...'
python preprocess.py \
    --text-path $DATA_DIR/val_texts.txt \
    --label-path $DATA_DIR/val_labels.txt \
    --vocab-path $DATA_DIR/vocab.npy

echo 'Creating testing features...'
python preprocess.py \
    --text-path $DATA_DIR/test_texts.txt \
    --label-path $DATA_DIR/test_labels.txt \
    --vocab-path $DATA_DIR/vocab.npy
