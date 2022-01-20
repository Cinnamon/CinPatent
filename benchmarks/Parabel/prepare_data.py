import sys
import os
import logging
from typing import Dict, List
sys.path.append('../')

from nltk.corpus import stopwords

from utils import make_args, read_dataset, get_text, get_tfidf_vectorizer


def get_labels(sample, classes):
    encoded_labels = [str(classes[x]) for x in sample['labels']]
    return ','.join(encoded_labels)

def create_data(data: Dict, classes: List, output_dir: str, lang: str):
    stop_words = stopwords.words('english') if lang == 'en' else None
    vectorizer = get_tfidf_vectorizer(stop_words)
    classes_mapper = {x: i for i, x in enumerate(classes)}

    X, y = {}, {}
    X['train'] = vectorizer.fit_transform([get_text(sample) for sample in data['train']])
    y['train'] = [get_labels(sample, classes_mapper) for sample in data['train']]
    X['val'] = vectorizer.transform([get_text(sample) for sample in data['val']])
    y['val'] = [get_labels(sample, classes_mapper) for sample in data['val']]
    X['test'] = vectorizer.transform([get_text(sample) for sample in data['test']])
    y['test'] = [get_labels(sample, classes_mapper) for sample in data['test']]
    n_features = len(vectorizer.vocabulary_)

    for data_set in ('train', 'val', 'test'):
        logging.info(f'Output {data_set}')
        with open(os.path.join(output_dir, f'{data_set}.txt'), 'w') as f:
            n_samples = len(data[data_set])
            f.write('{} {} {}\n'.format(n_samples, n_features, len(classes)))
            for i, ey in enumerate(y[data_set]):
                f.write(ey)
                for feat_id, feat_val in zip(
                    X[data_set].indices[X[data_set].indptr[i]:X[data_set].indptr[i+1]], 
                    X[data_set].data[X[data_set].indptr[i]:X[data_set].indptr[i+1]]
                ):
                    f.write(" {}:{}".format(feat_id, feat_val))
                f.write('\n')

if __name__ == '__main__':
    args = make_args()
    raw_data, classes = read_dataset(args.data_path)
    create_data(raw_data, classes, output_dir=args.output_dir, lang=args.lang)