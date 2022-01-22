"""
Created on 2018/12/9
@author yrh
"""

import os
import sys
from typing import Tuple
from argparse import ArgumentParser
sys.path.append('../')

from pathlib import Path
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from logzero import logger

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb
from deepxml.models import Model
from deepxml.networks import AttentionRNN
from deepxml.evaluation import get_ndcg, get_precision

from utils import output_scores

def make_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Path of configuration in yaml format')
    parser.add_argument('--do_train', action='store_true', help='Whether to train')
    parser.add_argument('--do_predict', action='store_true', help='Whether to inference')
    parser.add_argument('--tree_id', type=str, default=None)
    args = parser.parse_args()
    return args

def get_data_path(data_dir: str, type: str = "train") -> Tuple[str, str]:
    text_path = os.path.join(data_dir, f'{type}_texts.npy')
    label_path = os.path.join(data_dir, f'{type}_labels.npy')
    return (text_path, label_path)

def main(args):
    tree_id = F'-Tree-{args.tree_id}' if args.tree_id is not None else ''
    yaml = YAML(typ='safe')
    config = yaml.load(Path(args.config_file))
    data_cnf, model_cnf = config['data'], config['model']

    # Init model
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}{tree_id}')
    logger.info(F'Model Name: {model_name}')

    # Load embedding
    emb_path = os.path.join(data_cnf['data_dir'], 'emb_init.npy')
    emb_init = get_word_emb(emb_path)
    label_encoder_path = os.path.join(data_cnf['data_dir'], 'labels_binarizer')

    if args.do_train:
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(*get_data_path(data_cnf['data_dir'], type='train'))
        valid_x, valid_labels = get_data(*get_data_path(data_cnf['data_dir'], type='val'))
        mlb = get_mlb(label_encoder_path, labels=train_labels)
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_x)}')
        logger.info(F'Size of Validation Set: {len(valid_x)}')

        logger.info('Training')
        train_loader = DataLoader(MultiLabelDataset(train_x, train_y), model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
        valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=False), model_cnf['valid']['batch_size'], num_workers=4)
        model = Model(
            network=AttentionRNN, 
            labels_num=labels_num, 
            model_path=model_path,
            lang=data_cnf['lang'], 
            emb_init=emb_init, 
            **data_cnf['model'],
            **model_cnf['model']
        )
        model.train(train_loader, valid_loader, **model_cnf['train'])
        logger.info('Finish Training')

    if args.do_predict:
        logger.info('Loading Test Set')
        mlb = get_mlb(label_encoder_path)
        labels_num = len(mlb.classes_)
        test_x, test_labels = get_data(*get_data_path(data_cnf['data_dir'], type='test'))
        test_y = mlb.transform(test_labels)
        logger.info(F'Size of Test Set: {len(test_x)}')

        logger.info('Predicting')
        test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'], num_workers=4)
        t_loader = DataLoader(MultiLabelDataset(test_x, test_y), model_cnf['predict']['batch_size'], num_workers=4)
        model = Model(
            network=AttentionRNN, 
            labels_num=labels_num, 
            model_path=model_path, 
            lang=data_cnf['lang'],
            emb_init=emb_init,
            **data_cnf['model'], 
            **model_cnf['model']
        )
        scores = {}
        scores['f1'] = model.predict_(t_loader)
        sorted_probs, sorted_pred_labels = model.predict(test_loader, k=model_cnf['predict'].get('k', labels_num))
        sorted_pred_labels = mlb.classes_[sorted_pred_labels]
        for k in (1, 3, 5):
            scores[f'p@{k}'] = get_precision(sorted_pred_labels, test_labels, mlb, top=k)
            scores[f'ndcg@{k}'] = get_ndcg(sorted_pred_labels, test_labels, mlb, top=k)
        output_scores(scores, output_dir=model_cnf['path'])

if __name__ == '__main__':
    args = make_args()
    main(args)