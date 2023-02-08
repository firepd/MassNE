import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, name, dir='./log/', record=False):
        self.name = dir + name
        self.record = record
        
    def printf(self, string):
        print(string)
        if self.record:
            file = open(self.name, 'a')
            file.write(string)
            file.write('\n')
            file.close()


def evaluate(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    assert len(pred) == len(label)

    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss


class Result:
    def __init__(self, cols=None, minimize=True):
        if cols is None:
            cols = ['logloss', 'auc', 'acc']
        self.train_log = []
        self.valid_log = []
        self.test_log = []
        self.cols = cols
        self.minimize = minimize

    def to_df(self):
        self.train_log = pd.DataFrame(np.round(self.train_log, 4))
        self.valid_log = pd.DataFrame(np.round(self.valid_log, 4))
        self.test_log = pd.DataFrame(np.round(self.test_log, 4))
        assert self.train_log.shape == self.valid_log.shape == self.test_log.shape
        self.train_log.columns = self.cols
        self.valid_log.columns = self.cols
        self.test_log.columns = self.cols

    def check_df(self):
        if type(self.train_log) is not pd.DataFrame:
            self.to_df()

    def plot(self, col='logloss', name=None):
        self.check_df()
        log_len = len(self.train_log)
        plt.plot(np.arange(log_len), self.train_log[col], label='train')
        plt.plot(np.arange(log_len), self.valid_log[col], label='valid')
        plt.plot(np.arange(log_len), self.test_log[col], label='test')
        plt.title(col)
        plt.legend()
        if name is not None:
            plt.savefig(name)
        plt.show()
    
    def best_valid(self, col='logloss'):
        self.check_df()
        if self.minimize:
            index = self.valid_log[col].idxmin()
        else:
            index = self.valid_log[col].idxmax()
        
        return index+1, self.test_log.iloc[index].tolist()

    def best_test(self, col='logloss'):
        self.check_df()
        if self.minimize:
            index = self.test_log[col].idxmin()
        else:
            index = self.test_log[col].idxmax()
        return index + 1, self.test_log.iloc[index].tolist()


class Querey_tool:
    def __init__(self, data):
        self.unitid2name = {
            48: 'TERRAN_MARINE', 49: 'TERRAN_REAPER', 51: 'TERRAN_MARAUDER', 50: 'TERRAN_GHOST',
            53: 'TERRAN_HELLION', 484: 'TERRAN_HELLIONTANK', 692: 'TERRAN_CYCLONE',
            33: 'TERRAN_SIEGETANK', 52: 'TERRAN_THOR', 691: 'TERRAN_THORAP',
            34: 'TERRAN_VIKINGASSAULT', 35: 'TERRAN_VIKINGFIGHTER', 54: 'TERRAN_MEDIVAC',
            689: 'TERRAN_LIBERATOR', 55: 'TERRAN_BANSHEE', 57: 'TERRAN_BATTLECRUISER',
            73: 'PROTOSS_ZEALOT', 77: 'PROTOSS_SENTRY', 74: 'PROTOSS_STALKER',
            311: 'PROTOSS_ADEPT', 75: 'PROTOSS_HIGHTEMPLAR', 141: 'PROTOSS_ARCHON',
            83: 'PROTOSS_IMMORTAL', 4: 'PROTOSS_COLOSSUS', 78: 'PROTOSS_PHOENIX',
            80: 'PROTOSS_VOIDRAY', 496: 'PROTOSS_TEMPEST', 79: 'PROTOSS_CARRIER',
            105: 'ZERG_ZERGLING', 9: 'ZERG_BANELING', 110: 'ZERG_ROACH',
            688: 'ZERG_RAVAGER', 107: 'ZERG_HYDRALISK', 126: 'ZERG_QUEEN',
            111: 'ZERG_INFESTOR', 109: 'ZERG_ULTRALISK', 108: 'ZERG_MUTALISK',
            112: 'ZERG_CORRUPTOR', 114: 'ZERG_BROODLORD'
        }
        unit2embed = data.unit2int
        self.embed2name = {unit2embed[unit_id]:self.unitid2name[unit_id].split('_')[1] for unit_id in self.unitid2name}
        print(self.embed2name)
        self.name2embed = {self.embed2name[embed_id]: embed_id for embed_id in self.embed2name}

    def query(self, a):
        if type(a) is int:
            return self.embed2name[a]
        else:
            return [self.embed2name[i] for i in a]

    def query_id(self, a):
        if type(a) is str:
            return self.name2embed[a]
        else:
            return [self.name2embed[i] for i in a]


if __name__ == '__main__':
    result = Result(minimize=True)
    result.train_log = np.random.randn(10,3)
    result.test_log = np.random.randn(10,3)
    result.valid_log = np.random.randn(10,3)
    result.plot()
    print(result.best_valid())
    print(result.best_test())