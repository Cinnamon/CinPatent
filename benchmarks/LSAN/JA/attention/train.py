import numpy as np
from tqdm import tqdm
from sklearn import metrics

import logging


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s/%(funcName)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG,
                        datefmt="%m/%d/%Y %I:%M:%S %p")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

logger = init_logger()
def train(attention_model,train_loader,test_loader,criterion,opt,epochs = 5,GPU=True):
    if GPU:
        attention_model.cuda()
    for i in range(epochs):
        logger.info(f"Running EPOCH {i+1}")
        train_loss = []
        prec_k = []
        ndcg_k = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            # print(train[0].size(), train[1].size())
            # x, y = train[0], train[1]
            x, y = train[0].cuda(), train[1].cuda()
            y_pred= attention_model(x)
            loss = criterion(y_pred, y.float())/train_loader.batch_size
            loss.backward()
            opt.step()
            labels_cpu = y.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg_k.append(ndcg)
            train_loss.append(float(loss))
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        logger.info(f"epoch {i+1} train end : avg_loss = {avg_loss}")
        logger.info(f"precision@1 : {epoch_prec[0]} , precision@3 : {epoch_prec[2]} , precision@5 : {epoch_prec[4]}")
        logger.info(f"ndcg@1 : {epoch_ndcg[0]} , ndcg@3 : {epoch_ndcg[2]} , ndcg@5 : {epoch_ndcg[4]}")
        test_acc_k = []
        test_loss = []
        test_ndcg_k = []
        micro_f1 = 0
        for batch_idx, test in enumerate(tqdm(test_loader)):
            # x, y = test[0], test[1]
            x, y = test[0].cuda(), test[1].cuda()
            val_y= attention_model(x)
            loss = criterion(val_y, y.float()) /train_loader.batch_size
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_acc_k.append(prec)

            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_ndcg_k.append(ndcg)
            test_loss.append(float(loss))

            micro_f1 = metrics.f1_score(labels_cpu.numpy(), pred_cpu.round().numpy(), average='micro')
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)

        with open('./data/result.txt', 'w', encoding='utf-8') as f:
            f.write(str(micro_f1)+ '\n')
            f.write(str(test_prec[0])+str(test_prec[2])+str(test_prec[4]))
            f.write(str(test_ndcg[0])+str(test_ndcg[2])+str(test_ndcg[4]))
            
        logger.info(f"epoch {i+1} test end : avg_loss = {avg_test_loss}")
        logger.info(f"micro-f1: {micro_f1}")
        logger.info(f"precision@1 : {test_prec[0]} , precision@3 : {test_prec[2]} , precision@5 : {test_prec[4]}")
        logger.info(f"ndcg@1 : {test_ndcg[0]} , ndcg@3 : {test_ndcg[2]} , ndcg@5 : {test_ndcg[4]}")


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)
def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)