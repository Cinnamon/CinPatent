import torch
import argparse
from utils.data_utils import *
from train import initialize_model, validate
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='TextCNN')
    
    parser.add_argument('--train_data', type=str, default='./data/train.json', help='path to train dataset')
    parser.add_argument('--test_data', type=str, default='./data/test.json', help='path to test dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='check point')
    parser.add_argument('--config', type=str, default='./config/', help='config file')

    args = parser.parse_args()

    TRAIN_PATH = args.train_data
    TEST_PATH =args.test_data
    CHECK_POINT = args.checkpoint
    CONFIG = args.config
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('loading config...')
    config = torch.load(CONFIG)
    train_iter_len = config['train_iter_len']
    epochs = config['epochs']
    we = config['we']
    lr = config['lr']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    max_seq_len = config['max_seq_len']

    # read dataset
    print('reading dataset...')
    _, test_loader, _, fasttext_embedding = dataset(train_path=TRAIN_PATH, val_path=TEST_PATH, fasttext_path=we, batch_size=batch_size, max_seq_len=max_seq_len)

    model, criterion, _, _, _ = initialize_model(fasttext_embedding=fasttext_embedding.vectors, device=DEVICE, train_iter_len=train_iter_len, epochs=epochs, lr=lr, num_classes=num_classes)

    # load weigth to model
    print('loading checkpoint from: ', CHECK_POINT)
    checkpoint = torch.load(CHECK_POINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Val loss: {}\n'.format(checkpoint['val_loss']))

    # validate model with test set
    print('testing model with test set...')
    test_loss, micro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5 = validate(model = model, criterion = criterion, val_iterator = test_loader, device = DEVICE)
    print("  Test --- loss: {} - micro-f1: {}".format(test_loss, micro_f1))
    print("       --- ndcg1: {} - ndcg3: {} - ndcg5: {} - p1: {} - p3: {} - p5: {}\n".format(ndcg1, ndcg3, ndcg5, p1, p3, p5))
if __name__ == '__main__':
    main()