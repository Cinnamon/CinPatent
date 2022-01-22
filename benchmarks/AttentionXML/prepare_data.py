import logging
import re
import os
import sys
import unicodedata
from typing import List, Dict
sys.path.append('../')

from utils import make_args, read_dataset, get_text


def clean_text(text: str, lang: str):
    if lang == 'en':
        text = re.sub(r"[^A-Za-z0-9()`!?\']", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
    elif lang == 'ja':
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^ぁ-んァ-ン一-龥ーA-Za-z0-9(),!?%&$+='`―]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
    else:
        raise NotImplementedError
    return text.lower().strip()

def concat_fields(data: List[Dict]):
    texts = [get_text(row) for row in data]
    labels = [row['labels'] for row in data]
    return texts, labels

def save_file(data: List[str], root_dir: str, file_type: str, is_label: bool = False):
    if is_label:
        with open(os.path.join(root_dir, file_type + '_labels.txt'), 'w', encoding='utf-8') as f:
            for r in data:
                f.write(' '.join(r)+'\n')
    else:
        with open(os.path.join(root_dir, file_type + '_texts.txt'), 'w', encoding='utf-8') as f:
            for r in data:
                f.write(r + '\n')


if __name__ == '__main__':
    args = make_args()
    data, classes = read_dataset(args.data_path)
    for data_set in ('train', 'val', 'test'):
        logging.info(f'Output {data_set}')
        text, labels = concat_fields(data=data[data_set])
        cleaned_text = [clean_text(r, lang=args.lang) for r in text]
        save_file(data=cleaned_text, root_dir=args.output_dir, file_type=data_set, is_label=False)
        save_file(data=labels, root_dir=args.output_dir, file_type=data_set, is_label=True)