from attention.model import StructuredSelfAttention
from attention.train import train
import torch
import utils
import data_got
import json
import os

config = utils.read_config("config.yml")
if config.GPU:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print('loading data...\n')
label_num = 523
train_loader, test_loader, label_embed, X_tst, Y_tst, Y_trn = data_got.load_data(
    batch_size=config.batch_size)
label_embed = torch.from_numpy(label_embed).float()  # [L*300]

with open('./data/w2idx.json', 'r', encoding='utf-8') as f:
    w2idx = json.load(f)

print("load done")


def multilabel_classification(attention_model, train_loader, test_loader, epochs, GPU=True):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adam(attention_model.parameters(),
                           lr=0.001, betas=(0.9, 0.99))
    train(attention_model, train_loader, test_loader, loss, opt, epochs, GPU)


attention_model = StructuredSelfAttention(batch_size=config.batch_size, lstm_hid_dim=config['lstm_hidden_dimension'],
                                          d_a=config["d_a"], n_classes=label_num, label_embed=label_embed, w2idx=w2idx)
if config.use_cuda:
    attention_model.cuda()
# attention_model.to(DEVICE)
multilabel_classification(attention_model, train_loader,
                          test_loader, epochs=config["epochs"])
