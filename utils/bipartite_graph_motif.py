import time
import bisect
import dgl
import torch
import numpy as np

from tqdm import tqdm

# Count motif 1
def count_motif_1_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_out_ngr == dst)
  src_out_ngr_2, src_out_timestamp_2 = src_out_ngr[mask_2], src_out_timestamp[mask_2]

  for i, node_i in enumerate(src_out_ngr_2):
    left_margin_3, right_margin_3 = src_out_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr == dst)
    g.edata['motif_count'][eid][0] += torch.sum(mask_3)

def count_motif_1_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr == dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_out_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr == dst)
    g.edata['motif_count'][eid][1] += torch.sum(mask_3)

def count_motif_1_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr == dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr == dst)
    g.edata['motif_count'][eid][2] += torch.sum(mask_2)


# Count motif 2
def count_motif_2_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_out_ngr == dst)
  src_out_ngr_2, src_out_timestamp_2 = src_out_ngr[mask_2], src_out_timestamp[mask_2]

  for i, node_i in enumerate(src_out_ngr_2):
    left_margin_3, right_margin_3 = src_out_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr != node_i)
    g.edata['motif_count'][eid][3] += torch.sum(mask_3)

def count_motif_2_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr == dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_out_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr != node_i)
    g.edata['motif_count'][eid][4] += torch.sum(mask_3)
        

def count_motif_2_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr == node_i)
    g.edata['motif_count'][eid][5] += torch.sum(mask_2)


# Count motif 3
def count_motif_3_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_out_ngr == dst)
  src_out_ngr_3, src_out_timestamp_3 = src_out_ngr[mask_3], src_out_timestamp[mask_3]

  for i, node_i in enumerate(src_out_ngr_3):
    left_margin_2, right_margin_2 = pivot, src_out_timestamp_3[i]
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr != node_i)
    g.edata['motif_count'][eid][6] += torch.sum(mask_2)

def count_motif_3_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_out_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr == node_i)
    g.edata['motif_count'][eid][7] += torch.sum(mask_3)

def count_motif_3_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr == dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr != node_i)
    g.edata['motif_count'][eid][8] += torch.sum(mask_2)


# Count motif 4
def count_motif_4_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_out_ngr != dst)
  src_out_ngr_2, src_out_timestamp_2 = src_out_ngr[mask_2], src_out_timestamp[mask_2]

  for i, node_i in enumerate(src_out_ngr_2):
    left_margin_3, right_margin_3 = src_out_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr == node_i)
    g.edata['motif_count'][eid][9] += torch.sum(mask_3)
  
def count_motif_4_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_out_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr == dst)
    g.edata['motif_count'][eid][10] += torch.sum(mask_3)

def count_motif_4_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr == dst)
    src_out_ngr_2 = src_out_ngr[mask_2]
    g.edata['motif_count'][eid][11] += torch.sum(mask_2)


# Count motif 5
def count_motif_5_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot, pivot + threshold_time
  mask = torch.logical_and(src_out_timestamp > left_margin, src_out_timestamp <= right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]

  for i, node_i in enumerate(src_out_ngr):   
    ngr2, _, eids2 = g.in_edges(node_i, form='all')
    ngr2_timestamp = g.edata['timestamp'][eids2]
    left_margin_2, right_margin_2 = src_out_timestamp[i], pivot + threshold_time
    mask_2 = torch.logical_and(ngr2_timestamp > left_margin_2, ngr2_timestamp <= right_margin_2)
    mask_2 = torch.logical_and(mask_2, ngr2 != src)
    ngr2, ngr2_timestamp = ngr2[mask_2], ngr2_timestamp[mask_2]

    for j, node_j in enumerate(ngr2):
      ngr1, _, eids1 = g.in_edges(dst, form='all')
      ngr1_timestamp = g.edata['timestamp'][eids1]
      left_mragin_1, right_margin_1 = ngr2_timestamp[j], pivot + threshold_time
      mask_1 = torch.logical_and(ngr1_timestamp > left_mragin_1, ngr1_timestamp <= right_margin_1)
      mask_1 = torch.logical_and(mask_1, ngr1 == node_j)
      g.edata['motif_count'][eid][12] += torch.sum(mask_1)

def count_motif_5_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot - threshold_time, pivot
  mask = torch.logical_and(src_out_timestamp >= left_margin, src_out_timestamp < right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr2, _, eids2 = g.in_edges(dst, form='all')
    ngr2_timestamp = g.edata['timestamp'][eids2]
    left_margin_2, right_margin_2 = pivot, src_out_timestamp[i] + threshold_time
    mask_2 = torch.logical_and(ngr2_timestamp > left_margin_2, ngr2_timestamp <= right_margin_2)
    mask_2 = torch.logical_and(mask_2, ngr2 != src)
    ngr2, ngr2_timestamp = ngr2[mask_2], ngr2_timestamp[mask_2]

    for j, node_j in enumerate(ngr2):
      ngr1, _, eids1 = g.in_edges(node_i, form='all')
      ngr1_timestamp = g.edata['timestamp'][eids1]
      left_mragin_1, right_margin_1 = ngr2_timestamp[j], src_out_timestamp[i] + threshold_time
      mask_1 = torch.logical_and(ngr1_timestamp > left_mragin_1, ngr1_timestamp <= right_margin_1)
      mask_1 = torch.logical_and(mask_1, ngr1 == node_j)
      g.edata['motif_count'][eid][13] += torch.sum(mask_1)

def count_motif_5_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot, pivot + threshold_time
  mask = torch.logical_and(src_out_timestamp > left_margin, src_out_timestamp <= right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr3, _, eids3 = g.in_edges(dst, form='all')
    ngr3_timestamp = g.edata['timestamp'][eids3]
    left_margin_3, right_margin_3 = src_out_timestamp[i] - threshold_time, pivot
    mask_3 = torch.logical_and(ngr3_timestamp >= left_margin_3, ngr3_timestamp < right_margin_3)
    mask_3 = torch.logical_and(mask_3, ngr3 != src)
    ngr3, ngr3_timestamp = ngr3[mask_3], ngr3_timestamp[mask_3]

    for j, node_j in enumerate(ngr3):
      ngr4, _, eids4 = g.in_edges(node_i, form='all')
      ngr4_timestamp = g.edata['timestamp'][eids4]
      left_mragin_4, right_margin_4 = src_out_timestamp[i] - threshold_time, ngr3_timestamp[j]
      mask_4 = torch.logical_and(ngr4_timestamp >= left_mragin_4, ngr4_timestamp < right_margin_4)
      mask_4 = torch.logical_and(mask_4, ngr4 == node_j)
      g.edata['motif_count'][eid][14] += torch.sum(mask_4)

def count_motif_5_4(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot - threshold_time, pivot
  mask = torch.logical_and(src_out_timestamp >= left_margin, src_out_timestamp < right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr3, _, eids3 = g.in_edges(node_i, form='all')
    ngr3_timestamp = g.edata['timestamp'][eids3]
    left_margin_3, right_margin_3 = pivot - threshold_time, src_out_timestamp[i]
    mask_3 = torch.logical_and(ngr3_timestamp >= left_margin_3, ngr3_timestamp < right_margin_3)
    mask_3 = torch.logical_and(mask_3, ngr3 != src)
    ngr3, ngr3_timestamp = ngr3[mask_3], ngr3_timestamp[mask_3]

    for j, node_j in enumerate(ngr3):
      ngr4, _, eids4 = g.in_edges(dst, form='all')
      ngr4_timestamp = g.edata['timestamp'][eids4]
      left_mragin_4, right_margin_4 = pivot - threshold_time, ngr3_timestamp[j]
      mask_4 = torch.logical_and(ngr4_timestamp >= left_mragin_4, ngr4_timestamp < right_margin_4)
      mask_4 = torch.logical_and(mask_4, ngr4 == node_j)
      g.edata['motif_count'][eid][15] += torch.sum(mask_4)


# Count motif 6
def count_motif_6_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot, pivot + threshold_time
  mask = torch.logical_and(src_out_timestamp > left_margin, src_out_timestamp <= right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]

  for i, node_i in enumerate(src_out_ngr):   
    ngr3, _, eids3 = g.in_edges(node_i, form='all')
    ngr3_timestamp = g.edata['timestamp'][eids3]
    left_margin_3, right_margin_3 = pivot, src_out_timestamp[i]
    mask_3 = torch.logical_and(ngr3_timestamp > left_margin_3, ngr3_timestamp < right_margin_3)
    mask_3 = torch.logical_and(mask_3, ngr3 != src)
    ngr3, ngr3_timestamp = ngr3[mask_3], ngr3_timestamp[mask_3]

    for j, node_j in enumerate(ngr3):
      ngr1, _, eids1 = g.in_edges(dst, form='all')
      ngr1_timestamp = g.edata['timestamp'][eids1]
      left_mragin_1, right_margin_1 = src_out_timestamp[i], pivot + threshold_time
      mask_1 = torch.logical_and(ngr1_timestamp > left_mragin_1, ngr1_timestamp <= right_margin_1)
      mask_1 = torch.logical_and(mask_1, ngr1 == node_j)
      g.edata['motif_count'][eid][16] += torch.sum(mask_1)

def count_motif_6_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot, pivot + threshold_time
  mask = torch.logical_and(src_out_timestamp > left_margin, src_out_timestamp <= right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr2, _, eids2 = g.in_edges(dst, form='all')
    ngr2_timestamp = g.edata['timestamp'][eids2]
    left_margin_2, right_margin_2 = pivot, src_out_timestamp[i]
    mask_2 = torch.logical_and(ngr2_timestamp > left_margin_2, ngr2_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, ngr2 != src)
    ngr2, ngr2_timestamp = ngr2[mask_2], ngr2_timestamp[mask_2]

    for j, node_j in enumerate(ngr2):
      ngr4, _, eids4 = g.in_edges(node_i, form='all')
      ngr4_timestamp = g.edata['timestamp'][eids4]
      left_mragin_4, right_margin_4 = src_out_timestamp[i] - threshold_time, pivot
      mask_4 = torch.logical_and(ngr4_timestamp >= left_mragin_4, ngr4_timestamp < right_margin_4)
      mask_4 = torch.logical_and(mask_4, ngr4 == node_j)
      g.edata['motif_count'][eid][17] += torch.sum(mask_4)

def count_motif_6_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot - threshold_time, pivot
  mask = torch.logical_and(src_out_timestamp >= left_margin, src_out_timestamp < right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]

  for i, node_i in enumerate(src_out_ngr):   
    ngr3, _, eids3 = g.in_edges(dst, form='all')
    ngr3_timestamp = g.edata['timestamp'][eids3]
    left_margin_3, right_margin_3 = src_out_timestamp[i], pivot
    mask_3 = torch.logical_and(ngr3_timestamp > left_margin_3, ngr3_timestamp < right_margin_3)
    mask_3 = torch.logical_and(mask_3, ngr3 != src)
    ngr3, ngr3_timestamp = ngr3[mask_3], ngr3_timestamp[mask_3]

    for j, node_j in enumerate(ngr3):
      ngr1, _, eids1 = g.in_edges(node_i, form='all')
      ngr1_timestamp = g.edata['timestamp'][eids1]
      left_mragin_1, right_margin_1 = pivot, src_out_timestamp[i] + threshold_time
      mask_1 = torch.logical_and(ngr1_timestamp > left_mragin_1, ngr1_timestamp <= right_margin_1)
      mask_1 = torch.logical_and(mask_1, ngr1 == node_j)
      g.edata['motif_count'][eid][18] += torch.sum(mask_1)

def count_motif_6_4(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot - threshold_time, pivot
  mask = torch.logical_and(src_out_timestamp >= left_margin, src_out_timestamp < right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr2, _, eids2 = g.in_edges(node_i, form='all')
    ngr2_timestamp = g.edata['timestamp'][eids2]
    left_margin_2, right_margin_2 = src_out_timestamp[i], pivot
    mask_2 = torch.logical_and(ngr2_timestamp > left_margin_2, ngr2_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, ngr2 != src)
    ngr2, ngr2_timestamp = ngr2[mask_2], ngr2_timestamp[mask_2]

    for j, node_j in enumerate(ngr2):
      ngr4, _, eids4 = g.in_edges(dst, form='all')
      ngr4_timestamp = g.edata['timestamp'][eids4]
      left_mragin_4, right_margin_4 = pivot - threshold_time, src_out_timestamp[i]
      mask_4 = torch.logical_and(ngr4_timestamp >= left_mragin_4, ngr4_timestamp < right_margin_4)
      mask_4 = torch.logical_and(mask_4, ngr4 == node_j)
      g.edata['motif_count'][eid][19] += torch.sum(mask_4)


# Count motif 7
def count_motif_7_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot, pivot + threshold_time
  mask = torch.logical_and(src_out_timestamp > left_margin, src_out_timestamp <= right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]

  for i, node_i in enumerate(src_out_ngr):   
    ngr3, _, eids3 = g.in_edges(node_i, form='all')
    ngr3_timestamp = g.edata['timestamp'][eids3]
    left_margin_3, right_margin_3 = src_out_timestamp[i], pivot + threshold_time
    mask_3 = torch.logical_and(ngr3_timestamp > left_margin_3, ngr3_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, ngr3 != src)
    ngr3, ngr3_timestamp = ngr3[mask_3], ngr3_timestamp[mask_3]

    for j, node_j in enumerate(ngr3):
      ngr1, _, eids1 = g.in_edges(dst, form='all')
      ngr1_timestamp = g.edata['timestamp'][eids1]
      left_mragin_1, right_margin_1 = pivot, src_out_timestamp[i]
      mask_1 = torch.logical_and(ngr1_timestamp > left_mragin_1, ngr1_timestamp < right_margin_1)
      mask_1 = torch.logical_and(mask_1, ngr1 == node_j)
      g.edata['motif_count'][eid][20] += torch.sum(mask_1)

def count_motif_7_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot, pivot + threshold_time
  mask = torch.logical_and(src_out_timestamp > left_margin, src_out_timestamp <= right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr2, _, eids2 = g.in_edges(dst, form='all')
    ngr2_timestamp = g.edata['timestamp'][eids2]
    left_margin_2, right_margin_2 = src_out_timestamp[i] - threshold_time, pivot
    mask_2 = torch.logical_and(ngr2_timestamp >= left_margin_2, ngr2_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, ngr2 != src)
    ngr2, ngr2_timestamp = ngr2[mask_2], ngr2_timestamp[mask_2]

    for j, node_j in enumerate(ngr2):
      ngr4, _, eids4 = g.in_edges(node_i, form='all')
      ngr4_timestamp = g.edata['timestamp'][eids4]
      left_mragin_4, right_margin_4 =  pivot, src_out_timestamp[i]
      mask_4 = torch.logical_and(ngr4_timestamp > left_mragin_4, ngr4_timestamp < right_margin_4)
      mask_4 = torch.logical_and(mask_4, ngr4 == node_j)
      g.edata['motif_count'][eid][21] += torch.sum(mask_4)

def count_motif_7_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot - threshold_time, pivot
  mask = torch.logical_and(src_out_timestamp >= left_margin, src_out_timestamp < right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]

  for i, node_i in enumerate(src_out_ngr):   
    ngr3, _, eids3 = g.in_edges(dst, form='all')
    ngr3_timestamp = g.edata['timestamp'][eids3]
    left_margin_3, right_margin_3 = pivot, src_out_timestamp[i] + threshold_time
    mask_3 = torch.logical_and(ngr3_timestamp > left_margin_3, ngr3_timestamp < right_margin_3)
    mask_3 = torch.logical_and(mask_3, ngr3 != src)
    ngr3, ngr3_timestamp = ngr3[mask_3], ngr3_timestamp[mask_3]

    for j, node_j in enumerate(ngr3):
      ngr1, _, eids1 = g.in_edges(node_i, form='all')
      ngr1_timestamp = g.edata['timestamp'][eids1]
      left_mragin_1, right_margin_1 = src_out_timestamp[i], pivot
      mask_1 = torch.logical_and(ngr1_timestamp > left_mragin_1, ngr1_timestamp < right_margin_1)
      mask_1 = torch.logical_and(mask_1, ngr1 == node_j)
      g.edata['motif_count'][eid][22] += torch.sum(mask_1)

def count_motif_7_4(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  pivot = g.edata['timestamp'][eid]
  left_margin, right_margin = pivot - threshold_time, pivot
  mask = torch.logical_and(src_out_timestamp >= left_margin, src_out_timestamp < right_margin)
  mask = torch.logical_and(mask, src_out_ngr != dst)
  src_out_ngr, src_out_timestamp = src_out_ngr[mask], src_out_timestamp[mask]
  
  for i, node_i in enumerate(src_out_ngr):   
    ngr2, _, eids2 = g.in_edges(node_i, form='all')
    ngr2_timestamp = g.edata['timestamp'][eids2]
    left_margin_2, right_margin_2 = pivot - threshold_time, src_out_timestamp[i]
    mask_2 = torch.logical_and(ngr2_timestamp >= left_margin_2, ngr2_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, ngr2 != src)
    ngr2, ngr2_timestamp = ngr2[mask_2], ngr2_timestamp[mask_2]

    for j, node_j in enumerate(ngr2):
      ngr4, _, eids4 = g.in_edges(dst, form='all')
      ngr4_timestamp = g.edata['timestamp'][eids4]
      left_mragin_4, right_margin_4 = src_out_timestamp[i], pivot
      mask_4 = torch.logical_and(ngr4_timestamp > left_mragin_4, ngr4_timestamp < right_margin_4)
      mask_4 = torch.logical_and(mask_4, ngr4 == node_j)
      g.edata['motif_count'][eid][23] += torch.sum(mask_4)


def count_bipartite_graph(g, threshold_time, device='cpu'):
  func_ls = [count_motif_1_1, count_motif_1_2, count_motif_1_3,
              count_motif_2_1, count_motif_2_2, count_motif_2_3,
              count_motif_3_1, count_motif_3_2, count_motif_3_3,
              count_motif_4_1, count_motif_4_2, count_motif_4_3,
              count_motif_5_1, count_motif_5_2, count_motif_5_3, count_motif_5_4,
              count_motif_6_1, count_motif_6_2, count_motif_6_3, count_motif_6_4,
              count_motif_7_1, count_motif_7_2, count_motif_7_3, count_motif_7_4]

  
  # func_ls = [count_motif_1_1, count_motif_1_2, count_motif_1_3]

  g.edata['motif_count'] = torch.zeros(g.number_of_edges(), 24)
  for eid in tqdm(range(g.num_edges())):
    for f in func_ls:
      f(g, threshold_time, eid)
