import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.optim as optim
import time

from data import Data
from SC_model import BattleNet
from Massive import *
from utils import Result

SEED = 128
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_name = 'Mass'  # Mass, BattleNet
n_epochs = 60
batch_size = 64
learning_rate = 0.001
hidden_dim = 20


indi = True
coop = True
supp = True
att = False
ensure_mono = True
print(f'exp setting, coop:{coop}, supp:{supp}, need att:{att}, mono:{ensure_mono}')

file = 'bush'  # ['corrid', 'plain', 'bush']
save_model = False


dataset = Data(file)
n_unit = dataset.n_unit
min_t_size, max_t_size = dataset.min_team_size, dataset.max_team_size


if model_name == "Mass":
    model = MassNE(n_unit, hidden_dim, indi=indi, coop=coop, supp=supp, need_att=att, device=device)
else:
    model = BattleNet(n_unit, 0, device=device)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)


def evaluate(pred, label):
    if type(pred) != np.ndarray:
        pred = pred.cpu().detach().numpy()

    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss



criterion = nn.BCELoss()
record = Result(cols=['auc', 'acc', 'logloss'])
total_step = len(dataset.train) // batch_size + 1

start_time = time.time()
print('training begin')
for epoch in range(n_epochs):
    pass_time = (time.time() - start_time) / 60
    print('Epoch {}, minutes elapsed {:.1f}'.format(epoch + 1, pass_time))
    model.train()
    batch_gen = dataset.get_batch(batch_size)
    for i, (X, y) in enumerate(batch_gen):
        y_tensor = torch.Tensor(y).to(device)
        pred = model(X)
        loss = criterion(pred, y_tensor)

        if model_name == 'Mass' and ensure_mono:
            mono_loss = model.get_mono_loss()
            if np.random.rand() < 0.005:
                print(mono_loss)
            loss = loss + mono_loss * 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            print('Epoch [{}/{}], Step [{}/{}]'.format(epoch + 1, n_epochs, i + 1, total_step))

    model.eval()
    preds, ys = [], []
    batch_gen = dataset.get_batch(10000, 'train', shuffle=False)
    for X, y in batch_gen:
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.reshape(-1))
            ys.append(y.reshape(-1))

    auc, acc, logloss = evaluate(torch.cat(preds), np.concatenate(ys))
    print('Epoch [{}], train set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(epoch + 1, auc, acc, logloss))
    record.train_log.append([auc, acc, logloss])

    preds, ys = [], []
    batch_gen = dataset.get_batch(10000, 'valid', shuffle=False)
    for X, y in batch_gen:
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.reshape(-1))
            ys.append(y.reshape(-1))

    auc, acc, logloss = evaluate(torch.cat(preds), np.concatenate(ys))
    print('Epoch [{}], valid set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(epoch + 1, auc, acc, logloss))
    record.valid_log.append([auc, acc, logloss])

    preds, ys = [], []
    batch_gen = dataset.get_batch(10000, 'test', shuffle=False)
    for X, y in batch_gen:
        with torch.no_grad():
            pred = model(X)
            preds.append(pred.reshape(-1))
            ys.append(y.reshape(-1))

    auc, acc, logloss = evaluate(torch.cat(preds), np.concatenate(ys))
    print('Epoch [{}], test set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(epoch + 1, auc, acc, logloss))
    record.test_log.append([auc, acc, logloss])

    if save_model and epoch >= 50:
        command = input('save model?')
        if command == '1':
            torch.save(model.state_dict(), f'./mas_{file}')

# record.plot()
print('result:', record.cols)
print('best_valid on test:', record.best_valid())
