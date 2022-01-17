import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, fasttext_embedding, kernel_sizes = [5, 4, 3, 2], num_filters = 128, p = 0.5, num_classes = 1):
        super(TextCNN, self).__init__()
        self.num_classes = num_classes
        self.embeddings = nn.Embedding.from_pretrained(embeddings=fasttext_embedding, freeze=False)
        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=self.embeddings.embedding_dim, out_channels=num_filters, kernel_size=kernel_sizes[i])
            for i in range(len(kernel_sizes))])
        
        self.fc = nn.Linear(len(kernel_sizes)*num_filters, num_classes)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x: Tensor):
        x = self.embeddings(x).permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.conv1d]

        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]

        x = self.dropout(torch.cat(x, 1))
        output =  self.fc(x)
        return torch.sigmoid(output)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0