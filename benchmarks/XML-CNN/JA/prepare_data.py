import json
import logging
import re
from typing import List, Dict
from tqdm import tqdm
import argparse
import os, time
import logging
import unicodedata
import MeCab
wakati = MeCab.Tagger("-Owakati")


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s/%(funcName)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG,
                        datefmt="%m/%d/%Y %I:%M:%S %p")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^ぁ-んァ-ン一-龥A-Za-z0-9(),!?'`]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower().strip()


def read_jsonl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(r.strip()) for r in tqdm(
            f.readlines(), desc='Reading dataset', ncols=100)]

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
    texts = [row['abstract'] + row['title'] + row['claim_1'] + row['description'] for row in data]
    labels = [row['labels'] for row in data]

    return texts, labels


def preprocess_data(data: List[str], max_seq_len: int = 768):
    text_cleaned = [clean_text(r) for r in tqdm(data, desc="Cleaning text", ncols=100)]
    truncated_text = []
    for row in tqdm(text_cleaned, desc="Truncating data", ncols=100):
        tokens = wakati.parse(row[:1536]).split()
        if len(tokens) > max_seq_len:
            truncated_text.append(" ".join(tokens[:max_seq_len]))
        else:
            truncated_text.append(" ".join(tokens))
    
    return truncated_text


def save_file(texts: List[str], labels: List[List[str]], root_dir: str, file_name: str):
    with open(os.path.join(root_dir, file_name), 'w', encoding='utf-8') as f:
        for idx, (text, label) in enumerate(zip(texts, labels)):
            f.write(str(idx) + '\t' + ' '.join(label) + '\t' + text + '\n')

def main():
    parser = argparse.ArgumentParser("Prepare data")

    parser.add_argument("--raw_data_path", type=str, default="./cinpatent_en.ndjson", help="data file path")
    parser.add_argument("--max_len", type=int, default=768, help="max sequence length")
    parser.add_argument("--split_val", type=bool, default=False, help="split val data?")
    parser.add_argument("--storage_dir", type=str, default='./data', help="Path to save preprocessed data")

    args = parser.parse_args()

    logger = init_logger()

    data = read_jsonl_data(file_path=args.raw_data_path)

    train, val, test = None, None, None
    if args.split_val:
        train, val, test = split_data(data=data, split_val=True)
    else:
        train, test = split_data(data=data, split_val=False)

    train_texts, train_labels = concat_fields(data=train)
    test_texts, test_labels = concat_fields(data=test)
    
    train_texts, test_texts = preprocess_data(data=train_texts), preprocess_data(data=test_texts)

    save_file(texts=train_texts, labels=train_labels, root_dir=args.storage_dir, file_name='train.txt')
    save_file(texts=test_texts, labels=test_labels, root_dir=args.storage_dir, file_name='test.txt')

    if val is not None:
        val_texts, val_labels = concat_fields(data=val)
        val_texts, val_labels = concat_fields(data=val)
        val_texts = preprocess_data(data=val_texts)

        save_file(texts=val_texts, labels=val_labels, root_dir=args.storage_dir, file_name='valid.txt')

    logger.info("Done!")
    logger.info(f"Data saved to folder: {args.storage_dir}")

if __name__=="__main__":
    main()
    time.sleep(7)
