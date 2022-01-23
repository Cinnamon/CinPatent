import json
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from cnn_utils.utils import logger
import MeCab
wakati = MeCab.Tagger("-Owakati")

def read_json_data(train_path: str, valid_path: str, vocab_file: str):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(valid_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(vocab_file, 'r', encoding='utf-8') as f:
        w2idx = json.load(f)

    return train_data, val_data, w2idx


def text_to_seqs(dataset: List, w2idx: Dict, unk_idx: int):
    seqs = []
    for sentence in tqdm(dataset, desc='Converting text to seq', ncols=100, nrows=1):
        seqs.append([w2idx[word] if word in w2idx else unk_idx for word in wakati.parse(sentence).split()])
    return seqs

def pad_and_truncate_seqs(seqs: List, max_seq_len: int, pad_idx: int):
    seq_pads = np.zeros((len(seqs), max_seq_len))
    for i, seq in tqdm(enumerate(seqs), desc='Padding and truncating seqs', ncols=100, nrows=1):
        pad_len = max_seq_len - len(seq)
        if pad_len > 0:
            seq_pads[i] = np.pad(seq, (0, pad_len), 'constant', constant_values=(pad_idx))
        else:
            seq_pads[i] =  seq[:max_seq_len]
    return seq_pads

def dataset(train_path: str, val_path: str, vocab_file: str, batch_size: int, max_seq_len: int = 512):
    train_data, valid_data, w2idx = read_json_data(train_path=train_path, valid_path=val_path, vocab_file=vocab_file)

    pad_idx, unk_idx = w2idx['<pad>'], w2idx['<unk>']

    train_seq = text_to_seqs(dataset=train_data['text'], w2idx=w2idx, unk_idx=unk_idx)
    val_seq = text_to_seqs(dataset=valid_data['text'], w2idx=w2idx, unk_idx=unk_idx)

    train_pad = pad_and_truncate_seqs(seqs=train_seq, max_seq_len=max_seq_len, pad_idx=pad_idx)
    val_pad = pad_and_truncate_seqs(seqs=val_seq, max_seq_len=max_seq_len, pad_idx=pad_idx)
    
    lb = MultiLabelBinarizer()
    train_label = lb.fit_transform(train_data['label'])
    val_label = lb.transform(valid_data['label'])
    num_labels = train_label.shape[1]
    
    train_tensor = TensorDataset(torch.tensor(train_pad, dtype=torch.long), torch.tensor(train_label, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(val_pad, dtype=torch.long), torch.tensor(val_label, dtype=torch.long))

    logger.info('Creating data loader...')
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, num_labels, w2idx

if __name__=="__main__":
    pass