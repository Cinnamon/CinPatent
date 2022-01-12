import os
import torch
import scipy.sparse as sp
import numpy as np
from fastxml import Trainer, Inferencer
from torchmetrics.classification import F1, Precision
from sklearn.metrics import ndcg_score
from argparse import ArgumentParser
from fastxml.weights import propensity

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="fastxml", help="Model name in (fastxml, pfastxml, pfastrexml")
parser.add_argument("--data_dir", type=str, help="Data directory")
parser.add_argument("--model_dir", type=str, help="Model directory")

def load_X(path):
    X = sp.load_npz(path).astype(np.float32)
    return [X[i, :] for i in range(X.shape[0])]

def load_y(path, return_sparse=False):
    y = sp.load_npz(path)
    if return_sparse:
        return y
    res = [[] for _ in range(y.shape[0])]
    rows, cols = y.nonzero()
    for r, c in zip(rows, cols):
        res[r].append(c) 
    return res

def get_data(data_dir, dtype: str = "train", return_sparse=False):
    X = load_X(os.path.join(args.data_dir, f"X.{dtype}.npz"))
    y = load_y(os.path.join(args.data_dir, f"Y.{dtype}.npz"), return_sparse=return_sparse)
    return X, y

def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = torch.from_numpy(y_true)
    y_prob = torch.from_numpy(y_prob)
    metrics = {}
    metrics["micro_f1"] = F1(threshold=threshold, average="micro")(y_prob, y_true).item()
    for k in (1, 3, 5):
        metrics[f"p@{k}"] = Precision(threshold=threshold, average="micro", top_k=k)(y_prob, y_true).item()
        metrics[f"ndcg@{k}"] = ndcg_score(y_true, y_prob, k=k)
    return metrics


# Config
args = parser.parse_args()
os.makedirs(args.model_dir, exist_ok=True)
save_model_path = os.path.join(args.model_dir, "model")
trainer_config = {
    "n_trees": 5,
    "n_jobs": 4,
    "n_epochs": 20,
}

# Train
print("Getting train data")
X, y = get_data(args.data_dir, "train")
weights = propensity(y)
if args.model == "fastxml":
    trainer = Trainer(**trainer_config)
    trainer.fit(X, y)
elif args.model == "pfastxml":
    trainer = Trainer(**trainer_config)
    trainer.fit(X, y, weights)
elif args.model == "pfastrexml":
    trainer = Trainer(leaf_classifiers=True, **trainer_config)
    trainer.fit(X, y, weights)
else:
    raise NotImplementedError
trainer.save(save_model_path)

# Predict
print("Getting test data")
X, y = get_data(args.data_dir, "test", return_sparse=True)
clf = Inferencer(save_model_path)
y_prob = clf.predict(X, fmt="sparse")
scores = compute_metrics(y_true=y.toarray(), y_prob=y_prob.toarray())
for k, v in scores.items():
    print('{}: {:.2f}'.format(k, v), end=' ')