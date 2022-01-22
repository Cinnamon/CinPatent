import json
from typing import Dict, List
import numpy as np
import torchtext
from tqdm import tqdm
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import logging
from cnn_utils.utils import logger

def read_json_data(train_path: str, valid_path: str):
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(valid_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    return train_data, val_data

def text_to_seqs(dataset: List, w2idx: Dict, unk_idx: int):
    seqs = []
    for sentence in tqdm(dataset, desc='Converting text to seq', ncols=100, nrows=1):
        seqs.append([w2idx[word] if word in w2idx else unk_idx for word in nltk.word_tokenize(sentence)])
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

def dump_word_embeddings(file_path: str):
    ftext = torchtext.vocab.FastText('simple')
    ftext.stoi[ftext.itos[0]], ftext.stoi[ftext.itos[1]] = len(ftext.itos), len(ftext.itos)+1

    ftext.stoi['<unk>'] = 0
    ftext.stoi['<pad>'] = 1
    ftext.vectors = torch.cat((ftext.vectors, ftext.vectors[:2]), dim=0)
    ftext.vectors[:2] = torch.rand(2, 300)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(len(ftext.stoi))+' '+str(ftext.dim)+'\n')
        for k, v in tqdm(zip(ftext.stoi, ftext.vectors), desc=f'Saving FastText embedding to : {file_path}', ncols=100, nrows=1):
            f.write(k + ' ' + ' '.join(v.numpy().astype(str)) + '\n')

def dataset(train_path: str, val_path: str, fasttext_path: str, batch_size: int, max_seq_len: int = 512):
    train_data, valid_data = read_json_data(train_path=train_path, valid_path=val_path)

    if not os.path.exists(fasttext_path):
        dump_word_embeddings(fasttext_path)
    fasttext = torchtext.vocab.Vectors(name=fasttext_path)
    w2idx = fasttext.stoi

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
    
    return train_loader, val_loader, num_labels, fasttext

if __name__=="__main__":
    pass