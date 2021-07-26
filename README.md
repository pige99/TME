# TME
The source codes for TME-BNA: Temporal Motif-Preserving Network Embedding with Bicomponent Neighbor Aggregation.  
Our implementation is based on TGNs, and the user guide is below:
## Datasets
All datasets can be download from here: http://snap.stanford.edu/data/index.html.
The raw file should be saved to /data folder.
## Preprocess
python utils/preprocess_data.py --data wikipedia --bipartite  
python utils/count_motif.py --data wikipedia --threshold_time 86400 --bipartite
## Link prediction
python train_self_supervised.py --data wikipedia --use_memory --aggregator identity --memory_updater gru_long --prefix TME
## Node classfication
python train_self_supervised.py --data wikipedia --use_memory --aggregator last --memory_updater gru --prefix TME_GRUCell  
python train_supervised.py --data wikipedia --use_memory --aggregator last --memory_updater gru --prefix TME_GRUCell
