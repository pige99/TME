import sys
import time
import bisect
import argparse
import dgl
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from dgl.data.utils import save_graphs
from bipartite_graph_motif import count_bipartite_graph
from homogeneous_graph_motif import count_homogeneous_graph

def feature_normalize(data):
  mu = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  return (data - mu) / std

def count_motif(args):
  df = pd.read_csv("./data/ml_{}.csv".format(args.data), usecols=['u', 'i', 'ts'])
  num_edges = len(df.index)
  u, v, timestamp = df.u.values[:num_edges], df.i.values[:num_edges], df.ts.values[:num_edges]

  g = dgl.graph((torch.tensor(u), torch.tensor(v)))
  g.edata['timestamp'] = torch.tensor(timestamp, dtype=torch.int32)
  threshold_time = args.threshold_time

  if args.bipartite:
    count_bipartite_graph(g, threshold_time)
  else:
    count_homogeneous_graph(g, threshold_time)  

  graph_labels = {"glabel": torch.tensor([0])}
  save_graphs("./data/{}/{}_{}.dgl".format(args.data, args.data, args.threshold_time), [g], graph_labels)
  outfile = "./data/{}/{}_{}.npy".format(args.data, args.data, args.threshold_time)  
  data = g.edata['motif_count'].numpy()
  normalize_data = feature_normalize(data)
  null_edge = np.zeros((1, normalize_data.shape[1]))
  normalize_data = np.vstack([null_edge, normalize_data])
  np.save(outfile, normalize_data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser('Interface for motif counting')
  parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or Email_Eu)',
                      default='wikipedia')
  parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')
  parser.add_argument('--threshold_time', type=int, default=86400,
                      help='Time difference(seconds) between largest time and smallest time in motif')

  args = parser.parse_args()
  print(args)
  count_motif(args)


