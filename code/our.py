# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

all_mode = 3
indiv_mode = all_mode
coop_mode = all_mode
attack_mode = all_mode
defend_mode = all_mode


def combinations(team_size):
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            if i == j:
                continue
            index1.append(i)
            index2.append(j)
    #
    return index1, index2


def get_shift_index(n_unit, max_unit_size):
    idx1 = []
    idx2 = []
    for i in range(n_unit):
        for j in range(max_unit_size):
            if j == max_unit_size - 1:
                continue
            prev = i*max_unit_size + j
            later = i*max_unit_size + j + 1
            idx1.append(prev)
            idx2.append(later)
    return idx1, idx2


class HeadcountEffect(nn.Module):  # Marginal effect
    def __init__(self, n_unit, max_unit_size=800, mode=0):
        super(HeadcountEffect, self).__init__()
        self.n_unit = n_unit
        self.max_unit_size = max_unit_size
        assert mode in [0, 1, 2, 3]
        self.mode = mode
        self.embed = nn.Embedding(max_unit_size*n_unit, 1)
        self.embed.weight.data[:] = 0.5
        # size = max_unit_size  todo
        
        shift = [i*max_unit_size for i in range(n_unit)]
        self.shift = torch.LongTensor(shift)
        self.idx1, self.idx2 = get_shift_index(n_unit, max_unit_size)

    def forward(self, unit_nums):  # (32, 10)
        if self.mode == 0:
            unit_nums = unit_nums.clone()
            unit_nums[unit_nums > 1] = 1
            return unit_nums
        elif self.mode == 1:
            return unit_nums
        elif self.mode == 3:
            mask = (unit_nums != 0) * 1
            batch = unit_nums.size(0)
            shift = self.shift.unsqueeze(0).expand(batch, -1).to(unit_nums.device)
            unit_nums = unit_nums + shift
            num_effect = self.embed(unit_nums).squeeze(-1)  # [batch, n_unit]
            num_effect = torch.relu(num_effect) * mask
            return num_effect

    def non_negative(self, func):
        for layer in func:
            if type(layer) == torch.nn.modules.linear.Linear:
                layer.weight.data[layer.weight.data < 0] = 0

    def get_mono_loss(self):  # monotonicity loss
        if self.mode != 3:
            return 0
        diff = self.embed.weight[self.idx1].detach() - self.embed.weight[self.idx2]
        diff = torch.relu(diff)
        loss = diff.sum()
        return loss



class ANFM(nn.Module):
    def __init__(self, n_unit, hidden_dim, indi, coop, need_att):
        super(ANFM, self).__init__()
        assert n_unit > 1
        self.n_unit = n_unit
        self.skill = nn.Sequential(nn.Embedding(n_unit, 1), nn.Softplus())
        self.embedding = nn.Embedding(n_unit, hidden_dim)
        self.indi, self.need_att, self.coop = indi, need_att, coop

        self.indi_effect = HeadcountEffect(n_unit, mode=indiv_mode)
        self.coop_effect = HeadcountEffect(n_unit, mode=coop_mode)
        self.att_W = nn.Linear(hidden_dim, hidden_dim)
        self.index1, self.index2 = combinations(n_unit)
        dropout = nn.Dropout(0.2)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), dropout,
            nn.Linear(64, 1, bias=True), nn.ReLU(),
        )

    def forward(self, unit_nums):
        n_match = len(unit_nums)
        if self.indi:
            all_unit = torch.arange(self.n_unit).to(unit_nums.device)
            unit_skill = self.skill(all_unit)
            unit_skill = unit_skill.view(1, -1).expand(n_match, -1)  # [64, 10]
            team_skill = (unit_skill * self.indi_effect(unit_nums)).sum(dim=1)   # [64]
        else:
            team_skill = 0
        if self.coop:
            order2 = self.interact(unit_nums)
        else:
            order2 = 0
        return team_skill + order2

    def interact(self, unit_nums):
        batch = len(unit_nums)
        all_unit = torch.arange(self.n_unit).to(unit_nums.device)  # (10)
        all_embedding = self.embedding(all_unit)  # (10, 20)
        a = all_embedding[self.index1]  # [10*9, 20]
        b = all_embedding[self.index2]  # [10*9, 20]

        pair_wise_score = self.MLP(a * b).squeeze(dim=-1)  # [10*9]
        pair_wise_score = pair_wise_score.unsqueeze(0).expand(batch, -1)  # [32, 10*9]
        num_effect = self.coop_effect(unit_nums)
        order2 = pair_wise_score * num_effect[:, self.index1] * num_effect[:, self.index2]  # [32, 10*9]
        order2 = order2.sum(-1)  # [32]
        return order2

    def get_mono_loss(self):
        loss1 = self.indi_effect.get_mono_loss()
        loss2 = self.coop_effect.get_mono_loss()
        return loss1 + loss2


class Blade_chest(nn.Module):
    def __init__(self, n_unit, hidden_dim, method='inner', need_att=True):
        super(Blade_chest, self).__init__()
        assert n_unit > 1
        assert method in ['inner', 'dist']
        self.n_unit = n_unit
        self.chest = nn.Embedding(n_unit, hidden_dim)
        self.blade = nn.Embedding(n_unit, hidden_dim)

        self.index1, self.index2 = self.get_index(n_unit, n_unit)
        self.method, self.need_att = method, need_att
        self.attack_effect = HeadcountEffect(n_unit, mode=attack_mode)
        self.defend_effect = HeadcountEffect(n_unit, mode=defend_mode)
        self.att_W = nn.Linear(hidden_dim, hidden_dim)
        dropout = nn.Dropout(0.2)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), dropout,
            nn.Linear(64, 1, bias=True), nn.ReLU(),
        )

    def get_index(self, size1, size2):
        # size1, size2 = 3, 4
        index1 = np.repeat([i for i in range(size1)], size2)  # [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        index2 = np.tile([i for i in range(size2)], size1)  # [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        return index1, index2

    def forward(self, team_A, team_B):
        batch = len(team_A)
        attack_num_effect = self.attack_effect(team_A)  # [batch, 10]
        defend_num_effect = self.defend_effect(team_B)  # [batch, 10]
        all_unit = torch.arange(self.n_unit).to(team_A.device)  # [10]
        a_blade = self.blade(all_unit)  # (10, 20)
        b_chest = self.chest(all_unit)  # (10, 20)

        attack = a_blade[self.index1]  # (10*9, 20)
        defense = b_chest[self.index2]  # (10*9, 20)
        a_beat_b = self.get_interaction(attack, defense)  # (10*9)

        a_beat_b = a_beat_b.unsqueeze(0).expand(batch, -1)  # (64, 10*9)
        a_beat_b = a_beat_b * attack_num_effect[:, self.index1] * defend_num_effect[:, self.index2]
        a_beat_b = a_beat_b.sum(-1)
        return a_beat_b

    def get_interaction(self, a_blade, b_chest):  # (10*9, hidden_dim)
        if self.method == 'inner':
            interact = a_blade * b_chest
        else:  # dist
            interact = (a_blade - b_chest)
        a_beat_b = self.MLP(interact).squeeze(dim=-1)
        return a_beat_b

    def get_mono_loss(self):
        loss1 = self.attack_effect.get_mono_loss()
        loss2 = self.defend_effect.get_mono_loss()
        return loss1 + loss2


class Mass(nn.Module):
    def __init__(self, n_unit, hidden_dim=10, indi=True, coop=True, supp=True, need_att=True, device='cpu'):
        super(Mass, self).__init__()
        assert n_unit > 1
        self.n_unit = n_unit
        self.hidden_dim = hidden_dim
        self.device = device
        self.cooperate = ANFM(n_unit, hidden_dim, indi=indi, coop=coop, need_att=need_att)
        self.compete = Blade_chest(n_unit, hidden_dim, need_att=need_att)
        self.supp = supp

    def forward(self, data):
        if all_mode == 3:
            data = torch.LongTensor(data).to(self.device)
        else:
            data = torch.Tensor(data).to(self.device)
        team_A = data[:, :self.n_unit]
        team_B = data[:, self.n_unit:]

        Aability = self.cooperate(team_A)
        Bability = self.cooperate(team_B)

        if self.supp:
            comp_A = self.compete(team_A, team_B)
            comp_B = self.compete(team_B, team_A)
            adv = comp_A - comp_B
        else:
            adv = 0

        probs = torch.sigmoid(Aability - Bability + adv).view(-1)
        return probs

    def get_mono_loss(self):
        loss1 = self.cooperate.get_mono_loss()
        loss2 = self.compete.get_mono_loss()
        return loss1 + loss2


if __name__ == '__main__':
    max_unit_num = 100
    n_unit = 20
    n_match = 32
    match = np.random.randint(0, max_unit_num, (n_match, n_unit*2))

    model = Mass(n_unit, 20, need_att=False)
    preds = model(match)
    print(preds.shape)

