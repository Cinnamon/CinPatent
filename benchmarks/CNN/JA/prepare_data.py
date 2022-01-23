import json
from typing import Dict, List
from tqdm import tqdm
import os, argparse, re
import unicodedata

import MeCab

from cnn_utils.utils import logger


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^ぁ-んァ-ン一-龥()`!?\'A-Za-z0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower().strip()

def read_jsonl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(r.strip()) for r in tqdm(
            f.readlines(), desc='Reading dataset')]
    return data

def split_data(data: List[Dict], split_val: bool = False):
    test = [r for r in data if r['is_test']]
    if split_val:
        train = [r for r in data if r['is_train']]
        val = [r for r in data if r['is_dev']]
        return train, val, test
    else:
        train = [r for r in data if r['is_train'] or r['is_dev']]
        return train, test

def concat_fields(data: List[Dict]):
    texts = [row['abstract']+row['title'] +
             row['claim_1'] + row['description'] for row in data]
    labels = [row['labels'] for row in data]
    return texts, labels

def preprocess_data(data: List[str], max_seq_len: int = 768):
    text_cleaned = [clean_text(r) for r in tqdm(data, desc="Cleaning text")]
    truncated_text = []
    for row in tqdm(text_cleaned, desc="Truncating data"):
        tokens = row.split(" ")
        if len(tokens) > max_seq_len:
            truncated_text.append(' '.join(tokens[:max_seq_len]))
        else:
            truncated_text.append(row)

    return truncated_text

def save_processed_data(root_dir: str, texts: List[str], labels: List[str], file_type: str = "train"):
    labels = [r for r in labels]
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, file_type + ".json"), 'w', encoding='utf-8') as f:
        json.dump({'text': texts, 'label': labels}, f, indent=4)

def create_vocab(data: List[str], storage_dir: str = './data'):
    logger.info("Creating vocab...")
    wakati = MeCab.Tagger("-Owakati")
    tokens = set()
    for text in data:
        tokens.update(wakati.parse(text).split())
    vocab = ['<unk>', '<pad>'] + list(tokens)
    w2idx = {w:i for i, w in enumerate(vocab)}

    logger.info(f"Saving vocab to {os.path.join(storage_dir, 'w2idx.json')}")
    with open(os.path.join(storage_dir, 'w2idx.json'), 'w', encoding='utf-8') as f:
        json.dump(w2idx, f, indent=4)

def main():
    parser = argparse.ArgumentParser("Prepare data")
    parser.add_argument("--data_path", type=str, default="./cinpatent_ja.ndjson", help="data file path")
    parser.add_argument("--max_len", type=int, default=768, help="max sequence length")
    parser.add_argument("--storage_dir", type=str, default='./data', help="Path to save preprocessed data")
    args = parser.parse_args()

    data = read_jsonl_data(file_path=args.data_path)
    train, val, test = split_data(data=data, split_val=True)
    train_texts, train_labels = concat_fields(data=train)
    val_texts, val_labels = concat_fields(data=val)
    test_texts, test_labels = concat_fields(data=test)
    train_texts, val_texts, test_texts = preprocess_data(
        data=train_texts), preprocess_data(val_texts), preprocess_data(data=test_texts)

    create_vocab(data=train_texts, storage_dir=args.storage_dir)
    save_processed_data(root_dir=args.storage_dir, texts=train_texts, labels=train_labels, file_type='train_data')
    save_processed_data(root_dir=args.storage_dir, texts=val_texts, labels=val_labels, file_type='val_data')
    save_processed_data(root_dir=args.storage_dir, texts=test_texts, labels=test_labels, file_type='test_data')

    logger.info("Done!")
    logger.info(f"Data saved to folder: {args.storage_dir}")

if __name__=="__main__":
    main()