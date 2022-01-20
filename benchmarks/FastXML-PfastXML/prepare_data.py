import os
import scipy
import sys
import logging
from typing import List
sys.path.append('../')

from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords

from utils import get_text, get_tfidf_vectorizer, make_args, read_dataset


def create_data(data, classes: List, output_dir: str, lang: str = None):
    stop_words = stopwords.words('english') if lang == 'en' else None
    vectorizer = get_tfidf_vectorizer(stop_words)
    label_encoder = MultiLabelBinarizer(classes=classes, sparse_output=True)

    for data_set in ('train', 'val', 'test'):
        logging.info(f'Output {data_set} data')
        text_list = [get_text(sample) for sample in data[data_set]]
        label_list = [sample['labels'] for sample in data[data_set]]
        if data_set == 'train':
            X = vectorizer.fit_transform(text_list)
            y = label_encoder.fit_transform(label_list)
        else:
            X = vectorizer.transform(text_list)
            y = label_encoder.transform(label_list)

        scipy.sparse.save_npz(os.path.join(output_dir, f'X.{data_set}.npz'), X)
        scipy.sparse.save_npz(os.path.join(output_dir, f'Y.{data_set}.npz'), y)

if __name__ == '__main__':
    args = make_args()
    raw_data, classes = read_dataset(args.data_path)
    create_data(raw_data, classes, output_dir=args.output_dir, lang=args.lang)