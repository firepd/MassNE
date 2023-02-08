import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import KFold, train_test_split


def squad_num_vec(squad, unit2int):
    vec_num = np.zeros(len(unit2int))
    for key, num in squad.items():
        vec_num[unit2int[key]] = num
    # print(sum(vec_num))
    return vec_num


def make_dataframe(path):
    data = torch.load(path)
    id_cnt = set()
    for squad_a, squad_b, _ in data:
        id_cnt = id_cnt | squad_a.keys()
        id_cnt = id_cnt | squad_b.keys()

    unit2int = {name: i for i, name in enumerate(id_cnt)}
    table = []
    for squad_a, squad_b, win_rate in data:
        if win_rate > 0.5:
            win = 1
        elif win_rate < 0.5:
            win = 0
        else:
            continue
        
        squad_a_num = squad_num_vec(squad_a, unit2int)
        squad_b_num = squad_num_vec(squad_b, unit2int)
        if np.random.rand() < 0.5:
            match = np.concatenate([squad_a_num, squad_b_num, [win]])
        else:
            match = np.concatenate([squad_b_num, squad_a_num, [1 - win]])
        table.append(match)
    df = pd.DataFrame(table)
    return df, unit2int


class Data:
    def __init__(self, file='corrid', seed=None):
        assert file in ['corrid', 'plain', 'bush', 'plain_6000']
        path = '../data/' + file
        self.seed = seed
        df, unit2int = make_dataframe(path)
        # print(df.head())
        int2unit = {unit2int[name]: name for name in unit2int}
        self.unit2int, self.int2unit = unit2int, int2unit
        self.n_unit = len(self.unit2int)
        print('n_unit:', self.n_unit)
        self.data = df.to_numpy().astype(int)

        team1_size = self.data[:, :self.n_unit].sum(axis=1)
        team2_size = self.data[:, self.n_unit:-1].sum(axis=1)
        team_size = np.concatenate([team1_size, team2_size])
        print('min and max team size:', team_size.min(), team_size.max())
        print('mean team size:', team_size.mean())
        self.min_team_size, self.max_team_size = team_size.min(), team_size.max()

        unit_size = self.data.max()
        print('max unit size:', unit_size)
        self.max_unit_size = unit_size
        self.each_unit_size = {}
        for i in range(self.n_unit):
            self.each_unit_size[i] = max(self.data[:, i].max(), self.data[:, i+self.n_unit].max())

        unit_type1 = (self.data[:, :self.n_unit] >= 1).sum(axis=1)
        unit_type2 = (self.data[:, self.n_unit:-1] >= 1).sum(axis=1)
        unit_type = np.concatenate([unit_type1, unit_type2])
        print('min and max unit kind:', unit_type.min(), unit_type.max())
        self.min_unit_type, self.max_unit_type = unit_type.min(), unit_type.max()

        print('Overall win rate:', self.data[:, -1].mean())
        print('whole dataset size:', self.data.shape)
        self.train, self.valid, self.test = None, None, None
        self.split()

    def split(self):
        n_sample = len(self.data)
        index = list(range(n_sample))
        random.shuffle(index)
        train_idx = index[:int(n_sample*0.8)]
        valid_idx = index[int(n_sample*0.8): int(n_sample*0.9)]
        test_idx = index[int(n_sample*0.9):]
        self.train = self.data[train_idx]
        self.valid = self.data[valid_idx]
        self.test = self.data[test_idx]
        assert len(set(train_idx) & set(valid_idx) & set(test_idx)) == 0
        print('train shape:', self.train.shape)
        print('valid shape:', self.valid.shape)
        print('test shape:', self.test.shape)

    def encode(self, data_X):
        t = self.n_unit
        A = data_X[:, :t]
        B = data_X[:, t:]

        return A + B * -1

    def get_all(self, type='train', encoding=False):
        if type == 'train':
            data = self.train
        elif type == 'valid':
            data = self.valid
        elif type == 'test':
            data = self.test
        else:
            raise Warning('wrong type!')

        y = data[:, -1]
        data_X = self.select(data)

        if encoding:  # BT/LR
            return self.encode(data_X), y
        else:
            return data_X, y

    def get_batch(self, batch_size=32, type='train', shuffle=True):
        if type == 'train':
            data = self.train
        elif type == 'valid':
            data = self.valid
        elif type == 'test':
            data = self.test
        else:
            raise Warning('wrong type!')

        y = data[:, -1]
        data_X = self.select(data)
        length = len(data)
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)

        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = data_X[excerpt]
            yield X, y[excerpt]
            start_idx += batch_size

    def select(self, data):
        t = self.n_unit
        data = data[:, :t*2]
        return data


if __name__ == '__main__':
    dataset = Data()
