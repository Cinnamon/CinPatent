import random
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import torchtext
import argparse

from utils.model import *
from utils.data_utils import dataset
from utils.utils import logger, compute_measures

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f'using seed: {seed}')

def initialize_model(w2idx: Dict[str, int],
                    device: str, 
                    num_classes: int,
                    train_iter_len: int, 
                    epochs: int, lr: float, 
                    p=0.2, embed_dim: int = 300):
    logger.info('Initializing model...')
    # config = read_config_file(file_path=config_file)

    model = TextCNN(w2idx=w2idx, embed_dim=embed_dim, num_classes=num_classes, p=p).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    n_steps = train_iter_len * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=n_steps, num_warmup_steps=100)
    criterion = nn.BCELoss()

    return model, criterion, optimizer, scheduler, epochs

def step(model, criterion, optimizer, scheduler, batch, device):
    x, y_train = tuple(t.to(device) for t in batch)

    optimizer.zero_grad()
    
    y_pre = model(x)
    loss = criterion(y_pre, y_train.float())

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()

def validate(model, criterion, val_iterator, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_labels, targets = list(), list()
        for batch in val_iterator:
            x, y_true = tuple(t.to(device) for t in batch)

            output = model(x)
            loss = criterion(output, y_true.float())
            
            running_loss+= loss.item()
            pred_labels.extend(output.detach().cpu().numpy())
            targets.extend(y_true.detach().cpu().numpy())
        val_loss = running_loss/(len(val_iterator))
        
        pred_labels, targets = np.array(pred_labels), np.array(targets)
        micro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = compute_measures(targets, pred_labels)
    
    return val_loss, micro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5


def train(model, optimizer, scheduler, criterion, train_loader, val_loader, checkpoint, log, epochs, device):

    early_stopping = EarlyStopping(patience=10, delta=1e-5)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, batch in enumerate(train_loader):
            loss = step(model, criterion, optimizer, scheduler, batch, device)
            running_loss += loss 
            if (i + 1) % 100 == 0 or i == 0:
                print("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch+1, epochs, i + 1, len(train_loader), running_loss/(i + 1)))
        else:
            train_loss = running_loss/len(train_loader)
            print("Epoch: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch+1, epochs, i + 1, len(train_loader), train_loss))
            print("Evaluating...")
            # _, _, _, _, _, tp1, _, _ = validate(model, criterion, train_loader, device)
            # print("Train --- loss: {} - train-p1: {}".format(train_loss, tp1))
            val_loss, micro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = validate(model, criterion, val_loader, device)

            train_losses.append(train_loss), val_losses.append(val_loss)
            print("  Val --- loss: {} - micro-f1: {}".format(val_loss, micro_f1))
            print("      --- ndcg1: {} - ndcg3: {} - ndcg5: {} - p1: {} - p3: {} - p5: {}\n".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))
            
            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
            }, os.path.join(checkpoint, 'cp' + str(epoch+1) + '.pt'))

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print('Early stoppping. Previous model saved in: ', checkpoint)
                break

    logger.info(f'save log to: {checkpoint}')
    train_losses, val_losses = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1)
    np.savetxt(log +'loss_log.txt', np.hstack((train_losses, val_losses)), delimiter='#')

def main():
    parser = argparse.ArgumentParser(description='TextCNN')

    parser.add_argument('--train_data', type=str, default='./data/train.json', help='path to train dataset')
    parser.add_argument('--val_data', type=str, default='./data/val.json', help='path to val dataset')
    parser.add_argument('--log', type=str, default='./checkpoint/', help='path to log directory')
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument('--batch_size', type=int, default= 16, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number epochs')
    parser.add_argument('--max_length', type=int, default= 512, help='max sequence length')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to check point directory')
    # parser.add_argument('--resume', type=int, default=0, help='resume train model from checkpoint')
    parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
    parser.add_argument('--vocab_file', type=str, default='./data/w2idx.json', help='path to word embedding')
    parser.add_argument('--config', type=str, default='./config/', help='path to config directory')
    args = parser.parse_args()

    TRAIN_PATH = args.train_data
    VAL_PATH = args.val_data
    LR = args.lr
    LOG = args.log
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    CHECK_POINT = args.checkpoint
    # RESUME = args.resume
    EPOCHS = args.epochs
    VOCAB_FILE = args.vocab_file
    CONFIG = args.config
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    set_seed()
    # read dataset
    logger.info('reading dataset...')
    train_loader, val_loader, num_classes, w2idx = dataset(train_path=TRAIN_PATH, val_path=VAL_PATH, vocab_file=VOCAB_FILE, batch_size=BATCH_SIZE, max_seq_len=MAX_LENGTH)
    logger.info(f'Number of label: {num_classes}')

    print('\nsave config to: ', os.path.join(CONFIG, 'config.pt'))
    torch.save({'train_iter_len': len(train_loader),
                'embed_dim': args.embed_dim,
                'lr' : LR,
                'epochs': EPOCHS,
                'batch_size':BATCH_SIZE,
                'max_seq_len': MAX_LENGTH}, os.path.join(CONFIG, 'config.pt'))

    model, criterion, optimizer, scheduler, epochs = initialize_model(w2idx=w2idx, embed_dim=args.embed_dim,
                                                                      device=DEVICE, train_iter_len=len(train_loader),
                                                                      epochs=EPOCHS, lr=LR, p=0.2, num_classes=num_classes)
                                                    
    # if RESUME:
    #     checkpoint = torch.load(CHECK_POINT)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    print('')
    logger.info('training model...')
    train(model = model, optimizer = optimizer, scheduler = scheduler, criterion = criterion, train_loader = train_loader, 
          val_loader = val_loader, checkpoint = CHECK_POINT, log = LOG, epochs=epochs, device = DEVICE)

if __name__ == '__main__':
    main()