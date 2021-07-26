import torch
import argparse
import numpy as np

def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs, data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return adj_list, NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.uniform = uniform

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # Sort the ajd list of each node based on neighbor's timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
             
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)


    def find_before(self, src_idx, cut_time):
        '''
        Extracts all the interactions happening before cut_time for node src_idx.
        The returned interactions are sorted by time.
        Returns 3 lists: neighbors, edge_idxs, timestamps
        '''
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        neighbors = self.node_to_neighbors[src_idx][:i]
        edge_idxs = self.node_to_edge_idxs[src_idx][:i]
        edge_timestamps = self.node_to_edge_timestamps[src_idx][:i]
        return neighbors, edge_idxs, edge_timestamps

    def find_after(self, src_idx, cut_time, current_time):
        """
        Extracts all the interactions happening between cut_time and current time for node src_idx.
        The returned interactions are sorted by time.
        Returns 3 lists: neighbors, edge_idxs, timestamps
        """
        # i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time, side='right')
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        j = np.searchsorted(self.node_to_edge_timestamps[src_idx], current_time)
        neighbors = self.node_to_neighbors[src_idx][i:j]
        edge_idxs = self.node_to_edge_idxs[src_idx][i:j]
        edge_timestamps = self.node_to_edge_timestamps[src_idx][i:j]
        # print(neighbors, edge_idxs, edge_timestamps)
        return neighbors, edge_idxs, edge_timestamps

    def get_temporal_neighbor(self, source_nodes, cut_timestamps, current_timestamps, n_neighbors=10):
        """
        Given a list of source node ids and relative cut times and current times, 
        extracts sampled neighbors of eace node.

        Params
        ------
        source_nodes: List[int]
        cut_timestamps: List[float],
        current_timestamps: List[float],
        n_neighbors: int
        """

        # print(len(source_nodes), len(cut_timestamps), len(current_timestamps))
        cut_timestamps = cut_timestamps.astype(np.float32)
        current_timestamps = current_timestamps.astype(np.float32)
        assert (len(source_nodes) == len(cut_timestamps) == len(current_timestamps))
        assert (np.count_nonzero((current_timestamps - cut_timestamps) >= 0.0) == current_timestamps.shape[0]), (cut_timestamps[np.nonzero((current_timestamps - cut_timestamps) >= 0)], current_timestamps[np.nonzero((current_timestamps - cut_timestamps) >= 0)], cut_timestamps[np.nonzero((current_timestamps - cut_timestamps) >= 0)].shape)
        # assert (all(cut_time <= current_time for (cut_time, current_time) in zip(cut_timestamps, current_timestamps))), (cut_timestamps, current_timestamps)
        
        tmp_n_neighbors_before = n_neighbors // 2 if n_neighbors > 0 else 1
        tmp_n_neighbors_after = n_neighbors - tmp_n_neighbors_before
        # NB! All interactions described in these matrices are sorted in each row by time

        # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        neighbors_before = np.zeros((len(source_nodes), tmp_n_neighbors_before)).astype(np.int32)
        # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_times_before = np.zeros((len(source_nodes), tmp_n_neighbors_before)).astype(np.float32)
        # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs_before = np.zeros((len(source_nodes), tmp_n_neighbors_before)).astype(np.int32)

        # extracts all neighbors, interactions indexes and timestamps of all interactions of source_node happening before cut_time
        for i, (source_node, cut_timestamp) in enumerate(zip(source_nodes, cut_timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, cut_timestamp)
            # print(i, source_neighbors.shape)

            if len(source_neighbors) > 0 and tmp_n_neighbors_before > 0:
                if self.uniform: # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), tmp_n_neighbors_before)

                    neighbors_before[i, :] = source_neighbors[sampled_idx]
                    edge_times_before[i, :] = source_edge_times[sampled_idx]
                    edge_idxs_before[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times_before[i, :].argsort()
                    neighbors_before[i, :] = neighbors_before[i, :][pos]
                    edge_times_before[i, :] = edge_times_before[i, :][pos]
                    edge_idxs_before[i, :] = edge_idxs_before[i, :][pos]
                
                else: # Take most recent interactions
                    source_edge_times = source_edge_times[-tmp_n_neighbors_before:]
                    source_neighbors = source_neighbors[-tmp_n_neighbors_before:]
                    source_edge_idxs = source_edge_idxs[-tmp_n_neighbors_before:]

                    assert (len(source_neighbors) <= tmp_n_neighbors_before)
                    assert (len(source_edge_times) <= tmp_n_neighbors_before)
                    assert (len(source_edge_idxs) <= tmp_n_neighbors_before)

                    neighbors_before[i, tmp_n_neighbors_before - len(source_neighbors):] = source_neighbors
                    edge_times_before[i, tmp_n_neighbors_before - len(source_edge_times):] = source_edge_times
                    edge_idxs_before[i, tmp_n_neighbors_before - len(source_edge_idxs):] = source_edge_idxs                

        # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        neighbors_after = np.zeros((len(source_nodes), tmp_n_neighbors_after)).astype(np.int32)
        # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_times_after = np.zeros((len(source_nodes), tmp_n_neighbors_after)).astype(np.float32)
        # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs_after = np.zeros((len(source_nodes), tmp_n_neighbors_after)).astype(np.int32)

        # extracts all neighbors, interactions indexes and timestamps of all interactions of source_node happening between cut_time and current_time
        for i, (source_node, cut_timestamp, current_timestamp) in enumerate(zip(source_nodes, cut_timestamps, current_timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_after(source_node, cut_timestamp, current_timestamp)
            # print(i, source_neighbors.shape)

            if len(source_neighbors) > 0 and tmp_n_neighbors_after > 0:
                if self.uniform: # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), tmp_n_neighbors_after)

                    neighbors_after[i, :] = source_neighbors[sampled_idx]
                    edge_times_after[i, :] = source_edge_times[sampled_idx]
                    edge_idxs_after[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times_after[i, :].argsort()
                    neighbors_after[i, :] = neighbors_after[i, :][pos]
                    edge_times_after[i, :] = edge_times_after[i, :][pos]
                    edge_idxs_after[i, :] = edge_idxs_after[i, :][pos]
                
                else: # Take most recent interactions
                    source_edge_times = source_edge_times[-tmp_n_neighbors_after:]
                    source_neighbors = source_neighbors[-tmp_n_neighbors_after:]
                    source_edge_idxs = source_edge_idxs[-tmp_n_neighbors_after:]

                    assert (len(source_neighbors) <= tmp_n_neighbors_after)
                    assert (len(source_edge_times) <= tmp_n_neighbors_after)
                    assert (len(source_edge_idxs) <= tmp_n_neighbors_after)

                    neighbors_after[i, tmp_n_neighbors_after - len(source_neighbors):] = source_neighbors
                    edge_times_after[i, tmp_n_neighbors_after - len(source_edge_times):] = source_edge_times
                    edge_idxs_after[i, tmp_n_neighbors_after - len(source_edge_idxs):] = source_edge_idxs

        return neighbors_before, edge_idxs_before, edge_times_before,\
                neighbors_after, edge_idxs_after, edge_times_after
