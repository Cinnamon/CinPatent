import yaml
import numpy as np
from sklearn.metrics import f1_score, ndcg_score
import logging

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s/%(funcName)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG,
                        datefmt="%m/%d/%Y %I:%M:%S %p")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

logger = init_logger()

def read_config_file(file_path: str):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    
def compute_measures(target, pred):
    micro_f1 = f1_score(target, pred.round(), average='micro')          
    ndcg1 = ndcg_score(target, pred, k=1)
    ndcg3 = ndcg_score(target, pred, k=3)
    ndcg5 = ndcg_score(target, pred, k=5)

    p1 = precision_k(target, pred, 1)
    p3 = precision_k(target, pred, 3)
    p5 = precision_k(target, pred, 5)

    return micro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5

def precision_k(true_mat, score_mat, k):
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    score_mat = np.copy(backup)
    for i in range(rank_mat.shape[0]):
        score_mat[i][rank_mat[i, :-k]] = 0
    score_mat = np.ceil(score_mat)
    mat = np.multiply(score_mat, true_mat)
    num = np.sum(mat, axis=1)
    p = np.mean(num / k).item()
    return p