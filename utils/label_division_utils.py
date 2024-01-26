import os
import numpy as np
from sklearn.model_selection import train_test_split


def reassign_labels(y, seen_labels, unseen_labels):
    if isinstance(y, list):
        y = np.array(y, dtype=np.int64) 
    old_new_label_dict = {old_label: new_label for new_label, old_label in enumerate(seen_labels)}
    seen_nums = len(seen_labels)
    unseen_nums = len(unseen_labels)
    for i in range(unseen_nums):
        old_new_label_dict[seen_nums + i] = unseen_labels[i]
    def convert_label(old_label):
        return old_new_label_dict[old_label]
    new_y = [
        convert_label(label) for label in y
    ]
    new_y = np.array(new_y, dtype=np.int64)
    return new_y


def special_train_test_split(y, unseen_labels, test_size):
    if isinstance(y, list):
        y = np.array(y, dtype = np.int64) 
    seen_indices = [i for i in range(len(y)) if y[i] not in unseen_labels]
    ood_noise = np.where(y == unseen_labels[0])[0]
    unseen_indices = np.where(y == unseen_labels[1])[0] 
    seen_train_indices, seen_test_indices = train_test_split(seen_indices, test_size=test_size) 
    train_indices = seen_train_indices 
    return train_indices, seen_test_indices, ood_noise, unseen_indices

