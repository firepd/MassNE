import numpy as np
import trueskill
from trueskill import Rating, rate
from trueskill import TrueSkill, setup
from pprint import pprint
from itertools import chain
import math
from utils import evaluate, Result
from data import Data


class TSmodel:
    def __init__(self, n_unit, mu=25.0, sigma=8.333):
        self.n_unit = n_unit
        self.database = [Rating() for i in range(n_unit)]
        setup(backend='scipy')
        
    def play(self, data, outcomes):
        cnt = 0
        for match, y in zip(data, outcomes):
            team_A = []
            team_B = []
            index_A = {}
            index_B = {}
            for unit_id, num in enumerate(match[:self.n_unit]):
                if num == 0:
                    continue
                index_A[unit_id] = len(team_A)
                for _ in range(num):
                    team_A.append(self.database[unit_id])

            for unit_id, num in enumerate(match[self.n_unit:]):
                if num == 0:
                    continue
                index_B[unit_id] = len(team_B)
                for _ in range(num):
                    team_B.append(self.database[unit_id])

            if y == 1:
                y = [0, 1]
            elif y == 0:
                y = [1, 0]
            else:
                raise ValueError

            updated_A, updated_B = rate([team_A, team_B], ranks=y)
            for unit_id, pos in index_A.items():
                self.database[unit_id] = updated_A[pos]

            for unit_id, pos in index_B.items():
                self.database[unit_id] = updated_B[pos]

            cnt += 1
    
    def predict(self, data):
        outcomes = []
        for match in data:
            team_A = []
            team_B = []
            for unit_id, num in enumerate(match[:self.n_unit]):
                if num == 0:
                    continue
                team_A.append((self.database[unit_id], num))

            for unit_id, num in enumerate(match[self.n_unit:]):
                if num == 0:
                    continue
                team_B.append((self.database[unit_id], num))

            y = self.win_probability(team_A, team_B)
            outcomes.append(y)
        return np.array(outcomes)

    def win_probability(self, team1, team2):
        ts = trueskill.global_env()
        BETA = ts.beta
        delta_mu = sum(r.mu * num for r, num in team1) - sum(r.mu * num for r, num in team2)
        sum_sigma = sum(r.sigma ** 2 * num for r, num in chain(team1, team2))
        size = sum(num for r, num in team1) + sum(num for r, num in team2)
        denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
        return ts.cdf(delta_mu / denom)
        

SEED = 128
np.random.seed(SEED)
setup(backend='mpmath')  # install mpmath

epoch = 1
file = 'bush'  # ['corrid', 'plain', 'bush']

dataset = Data(file)
n_unit = dataset.n_unit

train_x, train_y = dataset.get_all('train')
valid_x, valid_y = dataset.get_all('valid')
test_x, test_y = dataset.get_all('test')

record = Result(cols=['auc', 'acc', 'logloss'], minimize=False)
model = TSmodel(n_unit)
for i in range(epoch):
    print(f'epoch:{i}')
    batch_gen = dataset.get_batch(5000)
    for j, (X, y) in enumerate(batch_gen):
        print(f'batch:{j}')
        model.play(X, y)

        # pred = model.predict(train_x)
        # auc, acc, logloss = evaluate(pred, train_y)
        # print('train auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))
        record.train_log.append([0.5, 0.5, 0.5])

        pred = model.predict(valid_x)
        auc, acc, logloss = evaluate(pred, valid_y)
        print('valid auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))
        record.valid_log.append([auc, acc, logloss])

        pred = model.predict(test_x)
        auc, acc, logloss = evaluate(pred, test_y)
        print('test auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))
        record.test_log.append([auc, acc, logloss])

print('result:', record.cols)
print('best_valid on test:', record.best_valid(col='acc'))
