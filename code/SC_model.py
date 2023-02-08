import torch
import torch.nn as nn
import torch.nn.functional as F


class NetDNN(nn.Module):
    def __init__(self, *n_layers, dropout=0.0):
        super(NetDNN, self).__init__()
        assert len(n_layers) > 1
        self.nstacks = len(n_layers) - 1
        self.layers = nn.ModuleList([nn.Linear(n_layers[i], n_layers[i+1]) for i in range(self.nstacks)])
        self.dropout = dropout

    def forward(self, x):
        y = x
        for i in range(self.nstacks - 1):
            y = self.layers[i](y)
            y = F.dropout(y, p=self.dropout)
            y = F.relu(y)
        # last layer has no relu
        y = self.layers[-1](y)
        y = F.dropout(y, p=self.dropout)
        return y


class BattleNet(nn.Module):
    def __init__(self, n_unit, n_terrain=0, device='cpu'):
        super(BattleNet, self).__init__()
        self.n_unit = n_unit
        self.n_terrain = n_terrain
        self.device = device
        dropout = 0.1
        # self.encoder = NetDNN(n_unit, 24, 24, dropout=dropout)
        # self.out_layer = NetDNN(24*2 + n_terrain, 24, 24, 1, dropout=dropout)
        self.encoder = NetDNN(n_unit, 32, 32, dropout=dropout)
        self.out_layer = NetDNN(32*2 + n_terrain, 32, 32, 1, dropout=dropout)

    def forward(self, x, map_info=None):
        x = torch.Tensor(x).to(self.device)
        team_A, team_B = x[:, :self.n_unit], x[:, self.n_unit:]
        team_vec_A = self.encoder(team_A)
        team_vec_B = self.encoder(team_B)
        if map_info is None:
            all_info = torch.cat([team_vec_A, team_vec_B], dim=-1)
        else:
            all_info = torch.cat([team_vec_A, team_vec_B, map_info], dim=-1)
        out = self.out_layer(all_info)
        out = torch.sigmoid(out).view(-1)
        return out


