import torch
import numpy as np
import argparse
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
#from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Coauthor
from dgl.data import CitationGraphDataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"


print('==> Begin to Read XW_idx')
seed=100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



dataset_name = 'cora'
#dataset_name = 'citeseer'
# dataset_name = 'coauthor_cs'

print('==> Begin to Read XW_idx')
data_idx = np.load('../idx/' + dataset_name + '_5_ood_0_idx.npz')
train_indices = data_idx['train_indices']
valid_indices = data_idx['valid_indices']
test_indices = data_idx['test_indices']

y_true = data_idx['y_true']

# y = torch.from_numpy(y_true).to(device)
y = torch.from_numpy(y_true)
y_train = y[train_indices]
y_test = y[test_indices]
y_valid = y[valid_indices]

y_real_true = data_idx['y_real_true']
y_train_real_true = y_real_true[train_indices]
n_classes = np.max(y_true) + 1
print(n_classes)


# 读取cora 数据集的 x和 edge_index ------
print('==> Preparing data..')
dataset_path = '../datasets/PYG'
if dataset_name == 'cora':
    graph = Planetoid(root=dataset_path, name='Cora')
elif dataset_name == 'citeseer':
    graph = Planetoid(root=dataset_path, name='Citeseer')

elif dataset_name == 'coauthor_cs':
    graph = Coauthor(root=dataset_path, name='CS')
# elif dataset_name == 'arxiv':
#     graph = PygNodePropPredDataset(root=dataset_path, name='ogbn-arxiv')

g = graph[0]
dim_feats = g.num_node_features

if dataset_name == 'citeseer':
    dataset = CitationGraphDataset("citeseer")
    dgl_g = dataset[0]
    in_nodes, out_nodes = dgl_g.edges()
    g.edge_index = np.stack((in_nodes, out_nodes))


# node_num_to_get = 0
# total_node_num = 0
if dataset_name == "cora":
    node_num_to_get = [78, 390]  # 5% 25%
    total_node_num = 2708
elif dataset_name == "citeseer":
    node_num_to_get = [78, 390]  # 5% 25%
    total_node_num = 3327
elif dataset_name == "coauthor_cs":
    node_num_to_get = [466, 2331]  # 5% 25%
    total_node_num = 18333


ood_rate_list = [5, 25]
for num_rate_i in range(len(ood_rate_list)):
    print("Begin to get ood rate" + str(ood_rate_list[num_rate_i]) + "_data")
    ood_num = node_num_to_get[num_rate_i]
    print(dataset_name)
    print(ood_num)

    ood_features_path = "../node_feature/pubmed" + str(ood_rate_list[num_rate_i]) + "%_" + str(ood_num) + "_node_features_for_" + dataset_name + ".npy"
    print(ood_features_path)
    pubmed_ood_node_features = np.load(ood_features_path)
    print(pubmed_ood_node_features, pubmed_ood_node_features.shape)


    concatenated_tensor = np.concatenate((g.x, pubmed_ood_node_features), axis=0)
    g.x = concatenated_tensor


    ood_node_idx_list = list(range(total_node_num, total_node_num+ood_num))

    edge_index_ood_list = [[],[]]
    for ood_node_idx in ood_node_idx_list:
        k = random.randint(1,5)
        in_k = random.randint(1,k)
        out_k = k - in_k
        # print(k, in_k, out_k)

        for i in range(in_k):
            edge_index_ood_list[0].append(ood_node_idx)
            random_number = random.choice([x for x in range(total_node_num+ood_num) if x != ood_node_idx])
            edge_index_ood_list[1].append(random_number)

        for i in range(out_k):
            edge_index_ood_list[1].append(ood_node_idx)
            random_number = random.choice([x for x in range(total_node_num+ood_num) if x != ood_node_idx])
            edge_index_ood_list[0].append(random_number)

    edge_index_ood = torch.tensor(edge_index_ood_list)
    g.edge_index = np.concatenate((g.edge_index, edge_index_ood), axis=1)


    temp_labels = []
    for ii in range(ood_num):
        temp_labels.append(random.randint(0, n_classes-1))
    # print(temp_labels)

    y_train_ = np.concatenate((y_train.numpy(), np.array(temp_labels)), axis=0)
    y_true_ = np.concatenate((y_true, np.array(temp_labels)), axis=0)

    ood_labels = [-1] * ood_num
    ood_labels_numpy = np.array(ood_labels)
    y_train_real_true_ = np.concatenate((y_train_real_true, ood_labels_numpy), axis=0)
    y_real_true_ = np.concatenate((y_real_true, ood_labels_numpy), axis=0)



    train_indices_ = np.concatenate((train_indices, ood_node_idx_list), axis=0)

    print("i:", num_rate_i)

    data_path = "../new_idx/" + dataset_name + "_ind_5_ood_" + "_pubmed_"+ str(ood_rate_list[num_rate_i]) + "_idx.npz"
    np.savez(data_path, train_indices=train_indices_, valid_indices=valid_indices, test_indices=test_indices, y_true=y_true_, y_real_true=y_real_true_)

    citeseer_pubmed_graph_path = "../new_idx/"+ dataset_name + "_pubmed_ood_" + str(ood_rate_list[num_rate_i]) + "_graph.npz"
    np.savez(citeseer_pubmed_graph_path, g_x=g.x, g_edge_index=g.edge_index)











