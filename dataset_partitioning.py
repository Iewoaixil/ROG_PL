import random
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from utils.label_division_utils import reassign_labels, special_train_test_split
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor

dataset_name = "cora"
unseen_num = 2
training_rate = 0.7
test_rate = 0.1
unseen_labels = [-2, -1]
train_seed = 100 
ind_noise_percentage = 5

np.random.seed(train_seed)

if dataset_name == 'cora':
    graph = Planetoid(root='./datasets/PYG',
                      name='Cora')  
elif dataset_name == 'citeseer':
    graph = Planetoid(root='./datasets/PYG',
                      name='Citeseer') 
elif dataset_name == 'coauthor_cs':
    graph = Coauthor(root='./datasets/Coauthor',
                      name='CS') 
g = graph[0]
original_num_classes = graph.num_classes 
seen_labels = list(range(original_num_classes - unseen_num))  
y_true = reassign_labels(g.y.numpy(), seen_labels, unseen_labels)  

# ood_noise add to train set, unseen_indices add to test set
train_indices, test_valid_indices, ood_noise, unseen_indices = special_train_test_split(y_true,  unseen_labels=unseen_labels, test_size=1 - training_rate) 

test_indices, valid_indices = train_test_split(test_valid_indices,
                                               test_size=test_rate / (1 - training_rate))  

test_indices = np.concatenate([test_indices, unseen_indices], axis=0)

num_classes = np.max(y_true) + 1 
for i in range(len(y_true)):
    if y_true[i] == -2:
        y_true[i] = -1
        
# label without noise
y_real_true = y_true.copy()

##add IND noise
ind_noise_num = int(len(train_indices) * ind_noise_percentage / 100)
train_indices_IND_noise = train_indices[:ind_noise_num]
train_indices_CLEAN = train_indices[ind_noise_num:]

for idx in train_indices_IND_noise:
    while y_true[idx] == y_real_true[idx]:
        y_true[idx] = random.randint(0, num_classes - 1)

##add OOD noise
train_indices = np.concatenate((train_indices, ood_noise))
y_true[ood_noise] = np.random.choice(list(range(np.max(y_true) + 1)), size=ood_noise.size, replace=True)

#save file
ind_noise_percentage = str(ind_noise_percentage)
np.savez('./idx/' + dataset_name + '_' + ind_noise_percentage + '_idx.npz', train_indices=train_indices, valid_indices=valid_indices, test_indices=test_indices, y_true=y_true, y_real_true=y_real_true)