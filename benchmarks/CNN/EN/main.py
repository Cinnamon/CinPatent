import random
import os
import sys
import warnings
import glob
sys.path.append('../../')
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import torchtext
import argparse

from cnn_utils.model import *
from cnn_utils.data_utils import dataset
from cnn_utils.utils import logger, compute_measures
from utils import output_scores


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f'using seed: {seed}')

def initialize_model(fasttext_embedding: torchtext.vocab.Vectors, num_classes: int, device: str, train_iter_len: int,  epochs: int, lr: float, p=0.2):
    logger.info('Initializing model...')
    # config = read_config_file(file_path=config_file)
    model = TextCNN(fasttext_embedding=fasttext_embedding, num_classes=num_classes, p=p).to(device)
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
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--train_file', type=str, default='train.json', help='path to train file')
    parser.add_argument('--val_file', type=str, default='val.json', help='path to val file')
    parser.add_argument('--test_file', type=str, default='test.json', help='path to test file')
    parser.add_argument('--do_train', action='store_true', help='whether to do training')
    parser.add_argument('--do_predict', action='store_true', help='whether to do testing')
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to log directory')
    parser.add_argument('--batch_size', type=int, default= 16, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number epochs')
    parser.add_argument('--epoch_predict', type=int, default=None, help='epoch to predict')
    parser.add_argument('--max_length', type=int, default= 512, help='max sequence length')
    # parser.add_argument('--resume', type=int, default=0, help='resume train model from checkpoint')
    parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
    parser.add_argument('--config_dir', type=str, default='./config/', help='path to config output directory')
    parser.add_argument('--model_dir', type=str, default='./model/', help='path to model directory')
    args = parser.parse_args()

    TRAIN_PATH = os.path.join(args.data_dir, args.train_file)
    VAL_PATH = os.path.join(args.data_dir, args.val_file)
    TEST_PATH = os.path.join(args.data_dir, args.test_file)
    WE = os.path.join(args.data_dir, 'fasttext.vec')
    CHECKPOINT_DIR = os.path.join(args.model_dir, 'ckpt')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    CONFIG_PATH = os.path.join(args.config_dir, 'config.pt')
    os.makedirs(args.log_dir, exist_ok=True)
    LOG = args.log_dir

    LR = args.lr
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    EPOCHS = args.epochs
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()
    if args.do_train:
        # read dataset
        logger.info('Reading dataset...')
        train_loader, val_loader, num_classes, fasttext_embedding = dataset(
            train_path=TRAIN_PATH, 
            val_path=VAL_PATH, 
            fasttext_path=WE, 
            batch_size=BATCH_SIZE, 
            max_seq_len=MAX_LENGTH
        )
        logger.info(f'Number of labels: {num_classes}')
        print('Save config to:', CONFIG_PATH)
        torch.save({
            'train_iter_len': len(train_loader),
            'we': WE, 'num_classes': num_classes,
            'lr': LR, 'epochs': EPOCHS,
            'batch_size':BATCH_SIZE, 'max_seq_len': MAX_LENGTH
            }, CONFIG_PATH)

        model, criterion, optimizer, scheduler, epochs = initialize_model(
            fasttext_embedding=fasttext_embedding.vectors,
            device=DEVICE, 
            train_iter_len=len(train_loader),
            epochs=EPOCHS, 
            lr=LR, p=0.2, 
            num_classes=num_classes,
        )
        logger.info('Training...')
        train(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, train_loader=train_loader, 
            val_loader=val_loader, checkpoint=CHECKPOINT_DIR, log=LOG, epochs=epochs, device=DEVICE)
    
    if args.do_predict:
        print('loading config...')
        config = torch.load(CONFIG_PATH)
        train_iter_len = config['train_iter_len']
        epochs = config['epochs']
        we = config['we']
        lr = config['lr']
        num_classes = config['num_classes']
        batch_size = config['batch_size']
        max_seq_len = config['max_seq_len']

        _, test_loader, _, fasttext_embedding = dataset(train_path=TRAIN_PATH, val_path=TEST_PATH, fasttext_path=we, batch_size=batch_size, max_seq_len=max_seq_len)
        model, criterion, _, _, _ = initialize_model(fasttext_embedding=fasttext_embedding.vectors, device=DEVICE, train_iter_len=train_iter_len, epochs=epochs, lr=lr, num_classes=num_classes)
        # load weigth to model
        epoch_predict = args.epochs if args.epoch_predict is None else args.epoch_predict
        ckpt_path = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{epoch_predict}.pt"))[0]  # checkpoint with matched suffix (epoch id)
        print("Testing with checkpoint", ckpt_path)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Val loss: {}\n'.format(checkpoint['val_loss']))

        # validate model with test set
        print('testing model with test set...')
        test_loss, micro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = validate(model = model, criterion = criterion, val_iterator = test_loader, device = DEVICE)
        scores = {
            "f1": micro_f1,
            "p@1": p1, "p@3": p3, "p@5": p5,
            "ndcg@1": ndcg1, "ndcg@3": ndcg3, "ndcg@5": ndcg5,
        }
        output_scores(scores, output_dir=args.model_dir)

if __name__ == '__main__':
    main()