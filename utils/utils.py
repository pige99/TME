import numpy as np
import torch
from torch import nn


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

class MergeLayer_2(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4, dim5, drop=0.3):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2 + dim3, dim4)
    self.fc2 = torch.nn.Linear(dim4, dim5)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2, x3):
    x = torch.cat([x1, x2, x3], dim=1)
    h = self.act(self.fc1(x))
    h = self.dropout(h)
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.1):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.dropout(x)
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class MLP_1(torch.nn.Module):
  def __init__(self, in_dim, output_dim, drop=0.1):
    super().__init__()
    self.fc_1 = torch.nn.Linear(in_dim, output_dim)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    # x = dropout(x)
    # x = self.act(self.fc_2(x))
    # x = dropout(x)

    return x


class MLP_2(torch.nn.Module):
  def __init__(self, in_dim, hidden_dim, output_dim, drop=0.1):
    super().__init__()
    self.fc_1 = torch.nn.Linear(in_dim, hidden_dim)
    self.fc_2 = torch.nn.Linear(hidden_dim, output_dim)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    # x = dropout(x)
    x = self.act(self.fc_2(x))
    x = dropout(x)

    return x

class NodeClassLayer(nn.Module):
    def __init__(self, in_dim, class_dim, dropout=0.1):
        super().__init__()
        hidden_dim = in_dim//2
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, class_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, h):
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc3(h).squeeze(dim=1)

        return h

class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)