from collections import defaultdict
import torch
import time
import numpy as np

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []

    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

class IdentityMessageAggregator(MessageAggregator):
  def __init__(self, device, gru_max_len=5):
    super(IdentityMessageAggregator, self).__init__(device)
    self.gru_max_len = gru_max_len

  def aggregate(self, node_ids, messages):
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.stack([m[0] for m in messages[node_id]]))
        unique_timestamps.append(messages[node_id][-1][1])

    if len(to_update_node_ids) != 0:
      max_len = self.gru_max_len
      unique_messages = [element[-max_len:][:] for element in unique_messages]
      messages_len = [element.shape[0] for element in unique_messages]
      unique_messages = torch.stack([F.pad(m, (0, 0, 0, max_len-m.shape[0])) for m in unique_messages])
      unique_messages = unique_messages.permute(1, 0, 2)
      unique_messages = pack_padded_sequence(unique_messages, messages_len, enforce_sorted=False)
      unique_timestamps = torch.stack(unique_timestamps)

    return to_update_node_ids, unique_messages, unique_timestamps   


  

def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "identity":
    return IdentityMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
