import time
import bisect
import dgl
import torch
import numpy as np

from tqdm import tqdm

# Count motif 1
def count_motif_1_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return

  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_out_ngr == dst)
  src_out_ngr_2, src_out_timestamp_2 = src_out_ngr[mask_2], src_out_timestamp[mask_2]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  
  for i, node_i in enumerate(src_out_ngr_2):        
    left_margin_3, right_margin_3 = src_out_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_in_ngr == dst)
    g.edata['motif_count'][eid][0] += torch.sum(mask_3)

def count_motif_1_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return

  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr == dst)   
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  
  for i, node_i in enumerate(src_out_ngr_1):        
    left_margin_3, right_margin_3 = pivot, src_out_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_in_ngr == dst)
    g.edata['motif_count'][eid][1] += torch.sum(mask_3)

def count_motif_1_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr == dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]
  
  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_2, right_margin_2 = src_in_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_in_ngr == dst)
    g.edata['motif_count'][eid][2] += torch.sum(mask_2)


# Count motif 2
def count_motif_2_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_out_ngr == dst)
  src_out_ngr_3, src_out_timestamp_3 = src_out_ngr[mask_3], src_out_timestamp[mask_3]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  
  for i, node_i in enumerate(src_out_ngr_3):
    left_margin_2, right_margin_2 = pivot, src_out_timestamp_3[i]
    mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_in_ngr == dst)
    g.edata['motif_count'][eid][3] += torch.sum(mask_2)

def count_motif_2_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid].item()
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr == dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]
  
  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_in_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_in_ngr == dst)
    g.edata['motif_count'][eid][4] += torch.sum(mask_3)

def count_motif_2_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  pivot = g.edata['timestamp'][eid].item()
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr == dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_in_ngr == dst)
    g.edata['motif_count'][eid][5] += torch.sum(mask_2)


# Count motif 3
def count_motif_3_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_in_ngr == dst)
  src_in_ngr_2, src_in_timestamp_2 = src_in_ngr[mask_2], src_in_timestamp[mask_2]
  
  for i, node_i in enumerate(src_in_ngr_2):
    left_margin_3, right_margin_3 = src_in_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_in_ngr == dst)
    g.edata['motif_count'][eid][6] += torch.sum(mask_3)

def count_motif_3_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr == dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]
  
  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_in_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > pivot, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr == dst) 
    g.edata['motif_count'][eid][7] += torch.sum(mask_3)

def count_motif_3_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr == dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]
  
  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_2, right_margin_2 = src_in_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr == dst)
    g.edata['motif_count'][eid][8] += torch.sum(mask_2)


# Count motif 4
def count_motif_4_1(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][9] += torch.sum(mask_3)

def count_motif_4_2(g, threshold_time, eid):
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
    mask_2 = torch.logical_and(mask_2, src_out_ngr == node_i)
    g.edata['motif_count'][eid][11] += torch.sum(mask_2)


# Count motif 5
def count_motif_5_1(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][12] += torch.sum(mask_2)

def count_motif_5_2(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][13] += torch.sum(mask_3)

def count_motif_5_3(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][14] += torch.sum(mask_2)


# Count motif 6
def count_motif_6_1(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][15] += torch.sum(mask_3)
  
def count_motif_6_2(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][16] += torch.sum(mask_3)

def count_motif_6_3(g, threshold_time, eid):
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
    g.edata['motif_count'][eid][17] += torch.sum(mask_2)


# Count motif 7
def count_motif_7_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_in_ngr == dst)
  src_in_ngr_2, src_in_timestamp_2 = src_in_ngr[mask_2], src_in_timestamp[mask_2]

  for i, node_i in enumerate(src_in_ngr_2):
    left_margin_3, right_margin_3 = src_in_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_out_ngr != dst)
    g.edata['motif_count'][eid][18] += torch.sum(mask_3)


def count_motif_7_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr == dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]

  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1]

  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_in_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(dst_out_timestamp > left_margin_3, dst_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, dst_out_ngr != src)
    g.edata['motif_count'][eid][19] += torch.sum(mask_3)


def count_motif_7_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]

  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_in_ngr == node_i)
    g.edata['motif_count'][eid][20] += torch.sum(mask_2)


# Count motif 8
def count_motif_8_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_in_ngr == dst)
  src_in_ngr_3, src_in_timestamp_3 = src_in_ngr[mask_3], src_in_timestamp[mask_3]

  for i, node_i in enumerate(src_in_ngr_3):
    left_margin_2, right_margin_2 = pivot, src_in_timestamp_3[i],
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr != dst)
    g.edata['motif_count'][eid][21] += torch.sum(mask_2)


def count_motif_8_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_in_ngr != dst)
  src_in_ngr_3, src_in_timestamp_3 = src_in_ngr[mask_3], src_in_timestamp[mask_3]

  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  for i, node_i in enumerate(src_in_ngr_3):
    if not g.has_edges_between(src, node_i):
      continue
    left_margin_1, right_margin_1 = src_in_timestamp_3[i] - threshold_time, pivot
    mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
    mask_1 = torch.logical_and(mask_1, src_out_ngr == node_i)
    g.edata['motif_count'][eid][22] += torch.sum(mask_1)


def count_motif_8_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return

  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1] 
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr == dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]

  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_2, right_margin_2 = src_in_timestamp_1[i], pivot
    mask_2 = torch.logical_and(dst_out_timestamp > left_margin_2, dst_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, dst_out_ngr != src)
    g.edata['motif_count'][eid][23] += torch.sum(mask_2)


# Count motif 9
def count_motif_9_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_in_ngr != dst)
  src_in_ngr_3, src_in_timestamp_3 = src_in_ngr[mask_3], src_in_timestamp[mask_3]

  for i, node_i in enumerate(src_in_ngr_3):
    if not g.has_edges_between(src, node_i):
      continue     
    left_margin_2, right_margin_2 = pivot, src_in_timestamp_3[i],
    mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_out_ngr == node_i)
    g.edata['motif_count'][eid][24] += torch.sum(mask_2)


def count_motif_9_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_in_ngr == dst)
  src_in_ngr_3, src_in_timestamp_3 = src_in_ngr[mask_3], src_in_timestamp[mask_3]

  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]

  for i, node_i in enumerate(src_in_ngr_3):
    left_margin_1, right_margin_1 = src_in_timestamp_3[i] - threshold_time, pivot
    mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
    mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
    g.edata['motif_count'][eid][25] += torch.sum(mask_1)


def count_motif_9_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  if not g.has_edges_between(dst, src):
    return

  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1] 
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot - threshold_time, pivot
  mask_2 = torch.logical_and(src_in_timestamp >= left_margin_2, src_in_timestamp < right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_in_ngr == dst)
  src_in_ngr_2, src_in_timestamp_2 = src_in_ngr[mask_2], src_in_timestamp[mask_2]

  for i, node_i in enumerate(src_in_ngr_2):
    left_margin_1, right_margin_1 = pivot - threshold_time, src_in_timestamp_2[i]
    mask_1 = torch.logical_and(dst_out_timestamp >= left_margin_1, dst_out_timestamp < right_margin_1)
    mask_1 = torch.logical_and(mask_1, dst_out_ngr != src)
    g.edata['motif_count'][eid][26] += torch.sum(mask_1)


# Count motif 10
def count_motif_10_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1]
  
  if len(src_in_ngr) == 0 or len(dst_out_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(dst_out_timestamp > left_margin_2, dst_out_timestamp < right_margin_2)
  mask_2 = torch.logical_and(mask_2, dst_out_ngr != src)
  dst_out_ngr_2, dst_out_timestamp_2 = dst_out_ngr[mask_2], dst_out_timestamp[mask_2]

  for i, node_i in enumerate(dst_out_ngr_2):
    left_margin_3, right_margin_3 = dst_out_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(src_in_timestamp > left_margin_3, src_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, src_in_ngr == node_i)
    g.edata['motif_count'][eid][27] += torch.sum(mask_3)

def count_motif_10_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1]
  
  if len(src_in_ngr) == 0 or len(dst_out_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(dst_out_timestamp > left_margin_3, dst_out_timestamp < right_margin_3)
  mask_3 = torch.logical_and(mask_3, dst_out_ngr != src)
  dst_out_ngr_3, dst_out_timestamp_3 = dst_out_ngr[mask_3], dst_out_timestamp[mask_3]

  for i, node_i in enumerate(dst_out_ngr_3):
    left_margin_1, right_margin_1 = dst_out_timestamp_3[i] - threshold_time, pivot
    mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
    mask_1 = torch.logical_and(mask_1, src_in_ngr == node_i)
    g.edata['motif_count'][eid][28] += torch.sum(mask_1)  

def count_motif_10_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1]
  
  if len(src_in_ngr) == 0 or len(dst_out_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(dst_out_timestamp >= left_margin_1, dst_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, dst_out_ngr != src)
  dst_out_ngr_1, dst_out_timestamp_1 = dst_out_ngr[mask_1], dst_out_timestamp[mask_1]

  for i, node_i in enumerate(dst_out_ngr_1):
    left_margin_2, right_margin_2 = dst_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(src_in_timestamp > left_margin_2, src_in_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, src_in_ngr == node_i)
    g.edata['motif_count'][eid][29] += torch.sum(mask_2)


# Count motif 11
def count_motif_11_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1]
  
  if len(src_out_ngr) == 0 or len(dst_out_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_2, right_margin_2 = pivot, pivot + threshold_time
  mask_2 = torch.logical_and(src_out_timestamp > left_margin_2, src_out_timestamp <= right_margin_2)
  mask_2 = torch.logical_and(mask_2, src_out_ngr != dst)
  src_out_ngr_2, src_out_timestamp_2 = src_out_ngr[mask_2], src_out_timestamp[mask_2]

  for i, node_i in enumerate(src_out_ngr_2):
    left_margin_3, right_margin_3 = src_out_timestamp_2[i], pivot + threshold_time
    mask_3 = torch.logical_and(dst_out_timestamp > left_margin_3, dst_out_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, dst_out_ngr == node_i)
    g.edata['motif_count'][eid][30] += torch.sum(mask_3)   

def count_motif_11_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  dst_in_timestamp = g.edata['timestamp'][g.in_edges(dst, form='eid')]
  dst_in_ngr = g.in_edges(dst)[0]
  
  if len(src_out_ngr) == 0 or len(dst_in_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_out_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(dst_in_timestamp > left_margin_3, dst_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, dst_in_ngr == node_i)
    g.edata['motif_count'][eid][31] += torch.sum(mask_3)

def count_motif_11_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  dst_in_timestamp = g.edata['timestamp'][g.in_edges(dst, form='eid')]
  dst_in_ngr = g.in_edges(dst)[0]
  
  if len(src_in_ngr) == 0 or len(dst_in_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr != dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]

  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_2, right_margin_2 = src_in_timestamp_1[i], pivot
    mask_2 = torch.logical_and(dst_in_timestamp > left_margin_2, dst_in_timestamp <= right_margin_2)
    mask_2 = torch.logical_and(mask_2, dst_in_ngr == node_i)
    g.edata['motif_count'][eid][32] += torch.sum(mask_2)  


# Count motif 12
def count_motif_12_1(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  dst_out_timestamp = g.edata['timestamp'][g.out_edges(dst, form='eid')]
  dst_out_ngr = g.out_edges(dst)[1]
  
  if len(src_out_ngr) == 0 or len(dst_out_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_3, right_margin_3 = pivot, pivot + threshold_time
  mask_3 = torch.logical_and(src_out_timestamp > left_margin_3, src_out_timestamp <= right_margin_3)
  mask_3 = torch.logical_and(mask_3, src_out_ngr != dst)
  src_out_ngr_3, src_out_timestamp_3 = src_out_ngr[mask_3], src_out_timestamp[mask_3]

  for i, node_i in enumerate(src_out_ngr_3):
    left_margin_2, right_margin_2 = pivot, src_out_timestamp_3[i]
    mask_2 = torch.logical_and(dst_out_timestamp > left_margin_2, dst_out_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, dst_out_ngr == node_i)
    g.edata['motif_count'][eid][33] += torch.sum(mask_2)    

def count_motif_12_2(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_in_timestamp = g.edata['timestamp'][g.in_edges(src, form='eid')]
  src_in_ngr = g.in_edges(src)[0]
  dst_in_timestamp = g.edata['timestamp'][g.in_edges(dst, form='eid')]
  dst_in_ngr = g.in_edges(dst)[0]
  
  if len(src_in_ngr) == 0 or len(dst_in_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_in_timestamp >= left_margin_1, src_in_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_in_ngr != dst)
  src_in_ngr_1, src_in_timestamp_1 = src_in_ngr[mask_1], src_in_timestamp[mask_1]

  for i, node_i in enumerate(src_in_ngr_1):
    left_margin_3, right_margin_3 = pivot, src_in_timestamp_1[i] + threshold_time
    mask_3 = torch.logical_and(dst_in_timestamp > left_margin_3, dst_in_timestamp <= right_margin_3)
    mask_3 = torch.logical_and(mask_3, dst_in_ngr == node_i)
    g.edata['motif_count'][eid][34] += torch.sum(mask_3)  

def count_motif_12_3(g, threshold_time, eid):
  src, dst = g.find_edges(eid)
  
  src_out_timestamp = g.edata['timestamp'][g.out_edges(src, form='eid')]
  src_out_ngr = g.out_edges(src)[1]
  dst_in_timestamp = g.edata['timestamp'][g.in_edges(dst, form='eid')]
  dst_in_ngr = g.in_edges(dst)[0]
  
  if len(src_out_ngr) == 0 or len(dst_in_ngr) == 0:
    return
  
  pivot = g.edata['timestamp'][eid]
  left_margin_1, right_margin_1 = pivot - threshold_time, pivot
  mask_1 = torch.logical_and(src_out_timestamp >= left_margin_1, src_out_timestamp < right_margin_1)
  mask_1 = torch.logical_and(mask_1, src_out_ngr != dst)
  src_out_ngr_1, src_out_timestamp_1 = src_out_ngr[mask_1], src_out_timestamp[mask_1]

  for i, node_i in enumerate(src_out_ngr_1):
    left_margin_2, right_margin_2 = src_out_timestamp_1[i], pivot
    mask_2 = torch.logical_and(dst_in_timestamp > left_margin_2, dst_in_timestamp < right_margin_2)
    mask_2 = torch.logical_and(mask_2, dst_in_ngr == node_i)
    g.edata['motif_count'][eid][35] += torch.sum(mask_2)


def count_homogeneous_graph(g, threshold_time):
  func_ls = [count_motif_1_1, count_motif_1_2, count_motif_1_3,
              count_motif_2_1, count_motif_2_2, count_motif_2_3,
              count_motif_3_1, count_motif_3_2, count_motif_3_3,
              count_motif_4_1, count_motif_4_2, count_motif_4_3,
              count_motif_5_1, count_motif_5_2, count_motif_5_3,
              count_motif_6_1, count_motif_6_2, count_motif_6_3,
              count_motif_7_1, count_motif_7_2, count_motif_7_3,
              count_motif_8_1, count_motif_8_2, count_motif_8_3,
              count_motif_9_1, count_motif_9_2, count_motif_9_3,
              count_motif_10_1, count_motif_10_2, count_motif_10_3,
              count_motif_11_1, count_motif_11_2, count_motif_11_3,
              count_motif_12_1, count_motif_12_2, count_motif_12_3]

  g.edata['motif_count'] = torch.zeros(g.number_of_edges(), 36)

  for eid in tqdm(range(g.num_edges())):
    for f in func_ls:
      f(g, threshold_time, eid)