import omikuji
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os

from torchmetrics.classification import F1, Precision
from sklearn.metrics import ndcg_score
import torch

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

# Train
# data_dir = "./datasets/jp_patents_bert/"
hyper_param = omikuji.Model.default_hyper_param()
# Adjust hyper-parameters as needed
hyper_param.n_trees = 5
model = omikuji.Model.train_on_data(os.path.join(args.data_dir, "train.txt"), hyper_param)
model.save(args.output_dir)

model = omikuji.Model.load(args.output_dir)

# Serialize & de-serialize
# model.save("./model_jp_bert")
# model = omikuji.Model.load("./model_jp_bert")
# Optionally densify model weights to trade off between prediction speed and memory usage
model.densify_weights(0.05)


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = torch.from_numpy(y_true)
    y_prob = torch.from_numpy(y_prob)
    metrics = {}
    metrics["micro_f1"] = F1(threshold=threshold, average="micro")(y_prob, y_true).item()
    for k in (1, 3, 5):
        metrics[f"p@{k}"] = Precision(threshold=threshold, average="micro", top_k=k)(y_prob, y_true).item()
        metrics[f"ndcg@{k}"] = ndcg_score(y_true, y_prob, k=k)
    return metrics

with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
    num_samples, num_feats, num_labels = map(int, f.readline().split())
    y_pred = np.zeros(shape=(num_samples, num_labels), dtype=float)
    y_true = np.zeros(shape=(num_samples, num_labels), dtype=bool)
    for i in tqdm(range(num_samples), desc="Predicting"):
        labels, *feature_value_str = f.readline().split()
        for label in labels.split(","):
            y_true[i][int(label)] = 1
        feature_value_pairs = []
        for x in feature_value_str:
            feat_id, feat_val = x.split(":")
            feature_value_pairs.append((int(feat_id), float(feat_val)))

        # import pdb; pdb.set_trace()
        label_score_pairs = model.predict(feature_value_pairs, top_k=num_labels)
        for label, score in label_score_pairs:
            y_pred[i][label] = score

score = compute_metrics(y_true, y_pred)
for k, v in score.items():
    print('{}: {:.2f}'.format(k, v * 100), end=' ')
