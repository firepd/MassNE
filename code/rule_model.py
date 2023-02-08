import numpy as np
import sys
import sklearn.metrics as metrics

sys.path.append('./statistics/')

from data import Data
from metadata import T_units, P_units, Z_units
from HP_Attack import HP_damage


model_name = 'LTD'  # Lanchester
file = 'bush'  # ['corrid', 'plain', 'bush']
num_to_run = 5


class LTD:
    def __init__(self, data):
        self.n_unit = data.n_unit
        int2unit = data.int2unit
        unit_ids= [int2unit[i] for i in range(data.n_unit)]
        self.HPs = np.array([HP_damage[i][0] for i in unit_ids])
        self.HPs_square_root = self.HPs ** 0.5
        self.to_ground_damage = np.array([HP_damage[i][1] for i in unit_ids])
        self.to_ground_freq = np.array([HP_damage[i][2] for i in unit_ids])
        self.to_ground_dps = self.to_ground_damage / self.to_ground_freq
        self.to_air_damage = np.array([HP_damage[i][3] for i in unit_ids])
        self.to_air_freq = np.array([HP_damage[i][4] for i in unit_ids])
        self.to_air_dps = self.to_air_damage / self.to_air_freq
        # self.unit_dps = (self.to_air_dps + self.to_ground_dps) / 2
        self.unit_dps = self.to_ground_dps  # to_ground_dps performs better

    def predict(self, X_test):
        team_A = X_test[:, :self.n_unit]
        team_B = X_test[:, self.n_unit:]
        A_score = (team_A * self.HPs_square_root * self.unit_dps).sum(axis=-1)
        B_score = (team_B * self.HPs_square_root * self.unit_dps).sum(axis=-1)
        diff = A_score - B_score
        return 1 / (1 + np.exp(-diff))


class TS_Lanchester2:
    def __init__(self, data):
        self.n_unit = data.n_unit
        int2unit = data.int2unit
        unit_ids = [int2unit[i] for i in range(self.n_unit)]
        self.HPs = np.array([HP_damage[i][0] for i in unit_ids])
        self.to_ground_damage = np.array([HP_damage[i][1] for i in unit_ids])
        self.to_ground_freq = np.array([HP_damage[i][2] for i in unit_ids])
        self.to_ground_dps = self.to_ground_damage / self.to_ground_freq
        self.to_air_damage = np.array([HP_damage[i][3] for i in unit_ids])
        self.to_air_freq = np.array([HP_damage[i][4] for i in unit_ids])
        self.to_air_dps = self.to_air_damage / self.to_air_freq

        ground_unit = T_units.get('ground', []) + P_units.get('ground', []) + Z_units.get('ground', [])
        ground_unit = ground_unit + T_units.get('ground_to_all', []) + \
                      P_units.get('ground_to_all', []) + Z_units.get('ground_to_all', [])
        air_to_all = T_units.get('air_to_all', []) + P_units.get('air_to_all', []) + Z_units.get('air_to_all', [])
        air_to_air = T_units.get('air_to_air', []) + P_units.get('air_to_air', []) + Z_units.get('air_to_air', [])
        air_to_ground = T_units.get('air_to_ground', []) + P_units.get('air_to_ground', []) + Z_units.get('air_to_ground', [])
        air_unit = air_to_all + air_to_air + air_to_ground
        ground_ind = [1 if i in set(ground_unit) else 0 for i in unit_ids]
        air_ind = [1 if i in set(air_unit) else 0 for i in unit_ids]
        assert sum(ground_ind) + sum(air_ind) == len(ground_unit) + len(air_unit)
        self.ground_indi = np.array(ground_ind)
        self.air_indi = np.array(air_ind)

    def predict(self, X_test):
        team_A = X_test[:, :self.n_unit]
        team_B = X_test[:, self.n_unit:]
        A_size = team_A.sum(axis=-1)
        B_size = team_B.sum(axis=-1)
        A_mean_HP = (team_A * self.HPs).sum(axis=-1) / A_size
        B_mean_HP = (team_B * self.HPs).sum(axis=-1) / B_size
        A_ground_sum_HP = (team_A * self.HPs * self.ground_indi).sum(axis=-1)
        A_air_sum_HP = (team_A * self.HPs * self.air_indi).sum(axis=-1)
        B_ground_sum_HP = (team_B * self.HPs * self.ground_indi).sum(axis=-1)
        B_air_sum_HP = (team_B * self.HPs * self.air_indi).sum(axis=-1)
        A_mean_to_ground_dps = (team_A * self.to_ground_dps).sum(axis=-1) / A_size
        A_mean_to_air_dps = (team_A * self.to_air_dps).sum(axis=-1) / A_size
        B_mean_to_ground_dps = (team_B * self.to_ground_dps).sum(axis=-1) / B_size
        B_mean_to_air_dps = (team_B * self.to_air_dps).sum(axis=-1) / B_size
        DPF_BA = (B_mean_to_air_dps * A_air_sum_HP + B_mean_to_ground_dps * A_ground_sum_HP) / (A_ground_sum_HP+A_air_sum_HP)
        DPF_AB = (A_mean_to_air_dps * B_air_sum_HP + A_mean_to_ground_dps * B_ground_sum_HP) / (B_ground_sum_HP+B_air_sum_HP)

        alpha = DPF_BA / A_mean_HP
        beta = DPF_AB / B_mean_HP
        diff = A_size * (beta ** 0.5) - B_size * (alpha ** 0.5)
        return 1 / (1 + np.exp(-diff))


def evaluate(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss


auc_all = []
acc_all = []
for _ in range(num_to_run):
    dataset = Data(file)
    if model_name == 'LTD':
        model = LTD(dataset)
    else:
        model = TS_Lanchester2(dataset)
    data_X, y = dataset.get_all('test')
    pred = model.predict(data_X)
    auc, acc, logloss = evaluate(pred, y)
    print('test set auc: {:.4f}, acc: {:.4f}, logloss: {:.4f}'.format(auc, acc, logloss))
    auc_all.append(auc)
    acc_all.append(acc)

mean_auc = np.mean(auc_all).round(4)
std_auc = np.std(auc_all).round(4)
mean_acc = np.mean(acc_all).round(4)
std_acc = np.std(acc_all).round(4)
print(f'mean auc {mean_auc}, std auc {std_auc}; mean acc {mean_acc}, std acc {std_acc}')

# print('mean auc: {:.4f}, and mean acc: {:.4f}'.format(np.mean(auc_all), np.mean(acc_all)))

