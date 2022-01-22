from ntpath import join
import numpy as np
# from gensim.models import KeyedVectors
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Dict
from tqdm import tqdm
import os, json
import nltk
import argparse

def read_data(root_dir: str):
    with open(os.path.join(root_dir, 'train_texts.txt'), 'r') as f:
        train_text = [r.strip() for r in f.readlines()]
    with open(os.path.join(root_dir, 'test_texts.txt'), 'r') as f:
        test_text = [r.strip() for r in f.readlines()]

    with open(os.path.join(root_dir, 'train_labels.txt'), 'r') as f:
        train_label = [r.strip().split() for r in f.readlines()]
    with open(os.path.join(root_dir, 'test_labels.txt'), 'r') as f:
        test_label = [r.strip().split() for r in f.readlines()]
    
    with open(os.path.join(root_dir, 'w2idx.json'), 'r') as f:
        w2idx = json.load(f)
    
    return train_text, test_text, train_label, test_label, w2idx

def text_to_seq(dataset: List[str], word2idx: Dict[str, int], unk_idx: int):
    seqs = []
    for sentence in tqdm(dataset, desc='text2seq', ncols=100):
        seqs.append([word2idx[word] if word in word2idx else unk_idx for word in nltk.word_tokenize(sentence)])
    return seqs

def pad_and_truncate_seqs(seqs, max_seq_len, pad_idx):
    seq_pads = np.zeros((len(seqs), max_seq_len))
    for i, seq in tqdm(enumerate(seqs), desc='Padding and truncating', ncols=100):
        pad_len = max_seq_len - len(seq)
        if pad_len > 0:
            seq_pads[i] = np.pad(seq, (0, pad_len), 'constant', constant_values=(pad_idx))
        else:
            seq_pads[i] =  seq[:max_seq_len]
    return seq_pads

# def load_glove(w2v_path: str = './data/glove.840B.300d.gensim'):
#     w2v_model = KeyedVectors.load(w2v_path)
#     w2idx = {v:k for k, v in enumerate(w2v_model.vocab.keys())}
    
#     return w2idx

def save_npy(tensor, path: str):
    np.save(path, tensor.astype(np.int64))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, default='./data', help="data dir")
    parser.add_argument("--max_seq_len", type=int, default=512, help='maximum sequence length')

    args = parser.parse_args()


    train_texts, test_texts, train_labels, test_labels, w2idx = read_data(root_dir=args.root_dir)
    unk_idx = w2idx['<unk>']
    pad_idx = w2idx['<pad>']
    
    train_seqs = text_to_seq(dataset=train_texts, word2idx=w2idx, unk_idx=unk_idx)
    test_seqs = text_to_seq(dataset=test_texts, word2idx=w2idx, unk_idx=unk_idx)

    train_pad = pad_and_truncate_seqs(train_seqs, max_seq_len=args.max_seq_len, pad_idx=pad_idx)
    test_pad = pad_and_truncate_seqs(test_seqs, max_seq_len=args.max_seq_len, pad_idx=pad_idx)

    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_labels)
    test_labels = mlb.transform(test_labels)

    print(f"saving npy data to {args.root_dir}")
    save_npy(train_pad, os.path.join(args.root_dir, 'X_train.npy'))
    save_npy(test_pad, os.path.join(args.root_dir, 'X_test.npy'))
    save_npy(train_labels, os.path.join(args.root_dir, 'y_train.npy'))
    save_npy(test_labels, os.path.join(args.root_dir, 'y_test.npy'))
    save_npy(np.random.randn(train_labels.shape[-1], 300), os.path.join(args.root_dir, 'label_embed.npy'))
    print("Done!")

if __name__=="__main__":
    main()


