import torch
import os
print(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
path = os.getcwd()
print(path)
#from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Coauthor


import torch
import numpy as np
import argparse
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

seed=100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



dataset_name = "Cora"
#dataset_name = "Citeseer"
# dataset_name = "CS"
dataset_path = "../datasets/PYG"
# for dataset_name in dataset_name_list:

print('==> Preparing data..')
if dataset_name == 'Cora':
    graph_dataset = Planetoid(root=dataset_path, name='Cora')
elif dataset_name == 'Citeseer':
    graph_dataset = Planetoid(root=dataset_path, name='Citeseer')
elif dataset_name == 'CS':
    graph_dataset = Coauthor(root=dataset_path, name='CS')
# elif dataset_name == 'Arxiv':
#     graph_dataset = PygNodePropPredDataset(root=dataset_path, name='ogbn-arxiv')



ood_noise_dataset = Planetoid(root = dataset_path, name = "Pubmed")
g = graph_dataset[0]
g1 = ood_noise_dataset[0]

print("graph_dataset load done...")

far_ood_rate = "25%_"
if dataset_name == "Cora" or "Citeseer":
    node_num_to_get = 390  # 25% ood
elif dataset_name == "CS":
    node_num_to_get = 2331  # 25% ood


# node_num_to_get = 78  # 5% ood
# node_num_to_get = 389  # 25% ood for citeseer
# node_num_to_get = 778   # 50% ood for citeseer
# node_num_to_get = 1167   # 75% ood for citeseer


size = g1.x.shape[0]
new_order = np.random.permutation(size)
shuffled_tensor = g1.x[new_order]
node_feature = shuffled_tensor[:node_num_to_get]
print(node_feature)
print(node_feature.shape)


expanded_feature = np.pad(node_feature, ((0, 0), (0, g.num_features - g1.num_features)), mode='constant')
print(type(expanded_feature))  # 'numpy.ndarray'>
print(expanded_feature)
print(expanded_feature.shape)


save_path = "../node_feature/pubmed" + far_ood_rate + str(node_num_to_get) + "_node_features_for_" + dataset_name + ".npy"
np.save(save_path, expanded_feature)
