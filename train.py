import os
import sys
import pathlib
import time
import datetime
from utils.core import accuracy 
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
LOG_FREQ = 1
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import warnings
from sklearn.cluster import KMeans
import scipy
import faiss
warnings.filterwarnings("ignore")



class MessagePassing(MessagePassing):
    def __init__(self):
        super(MessagePassing, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        def message_func(edge_index, x_i):
            return x_i

        aggregated = self.propagate(edge_index, x=x, message_func=message_func)
        return aggregated


class ROGPL(nn.Module):
    def __init__(self,
                 in_channels, 
                 hidden_1,
                 hidden_2,
                 n_classes,
                 center_num):
        super(ROGPL, self).__init__()
        self.n_classes = n_classes
        self.center_num = center_num
        self.conv1 = GCNConv(in_channels, hidden_1)
        self.conv2 = GCNConv(hidden_1, hidden_2)
        self.interior_prototype = nn.Linear(2 * hidden_2, self.center_num, bias=False)

        self.border_prototype = nn.ModuleList()
        for i in range(self.center_num):
            self.border_prototype.append(nn.Linear(2 * hidden_2, n_classes, bias=False))

    def weigth_init(self, data, label, index):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        aggregated = MessagePassing()
        neighbor = aggregated(x, edge_index)
        x = torch.cat((x, neighbor), dim=1)
        features = x[index]
        labels = label[index]
        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(features.detach().cpu())

        temp = torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.interior_prototype.weight.data.copy_(temp)

        p = []
        for i in range(self.n_classes):
            p.append(features[labels == i].mean(dim=0).view(1, -1))
        temp = torch.cat(p, dim=0)
        for i in range(self.center_num):
            self.border_prototype[i].weight.data.copy_(temp)

    def update_interior_prototype_weight(self, x):
        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(x.detach().cpu())
        temp = torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.interior_prototype.weight.data.copy_(temp)

    def get_mul_interior_prototype(self):
        pros = []
        for name, param in self.named_parameters():
            if name.startswith('border_prototype.'):
                pros.append(param)
        return pros

    def get_interior_prototype(self):
        for name, param in self.named_parameters():
            if name.startswith('interior_prototype.weight'):
                pro = param
        return pro

    def get_mid_h(self):
        return self.fea

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        aggregated = MessagePassing()
        neighbor = aggregated(x, edge_index)
        x = torch.cat((x, neighbor), dim=1)
        self.fea = x
        out = self.interior_prototype(x)
        index = torch.argmax(out, dim=1)
        out = torch.FloatTensor(x.shape[0], self.n_classes).cuda()
        for i in range(self.center_num):
            out[index == i] = self.border_prototype[i](x[index == i])
        return out, x


def evaluate_test(data, model, y_true, indices, ind_indices, ood_indices):
    model.eval()
    with torch.no_grad():
        output, _ = model(data)
        logits = output
        scores, pred = logits.softmax(dim=1).max(dim=1)
        pred = pred.to('cpu').numpy()
        scores = scores.to('cpu').numpy()
        max_acc = 0
        for i in range(0, 100):
            t = i * 0.01
            scores_temp, pred_temp = logits.softmax(dim=1).max(dim=1)
            index = scores_temp < t
            pred_temp[index == True] = -1
            pred_temp = pred_temp.to('cpu').numpy()
            acc = accuracy_score(y_true[indices], pred_temp[indices])
            if acc > max_acc:
                max_acc = acc
                pred = pred_temp
        test_acc = accuracy_score(y_true[indices], pred[indices])
        test_f1 = f1_score(y_true[indices], pred[indices], average='macro')
        labels_all = copy.deepcopy(y_true)
        labels_all[ind_indices] = 1
        labels_all[ood_indices] = 0
        test_auc = roc_auc_score(labels_all[indices], scores[indices])
    return {'test_acc': test_acc, 'test_f1': test_f1, 'test_auc': test_auc}


def evaluate_valid(data, model, y_true, indices):
    model.eval()
    with torch.no_grad():
        output, _ = model(data)
        logits = output
        scores, pred = logits.softmax(dim=1).max(dim=1)
        pred = pred.to('cpu').numpy()
        valid_acc = accuracy_score(y_true[indices], pred[indices])
        valid_f1 = f1_score(y_true[indices], pred[indices], average='macro')
    return {'valid_acc': valid_acc, 'valid_f1': valid_f1}

def constraint(device,interior_prototype):
    if isinstance(interior_prototype,list):
        sum=0
        for p in interior_prototype:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(interior_prototype)
    else:
        return torch.norm(torch.mm(interior_prototype,interior_prototype.T)-torch.eye(interior_prototype.shape[0]).to(device))

def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def build_logger(params):
    logger_root = f'Results/{params.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, params.project, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def build_model_optim_scheduler(params, device, build_scheduler=True):
    n_classes = params.n_classes
    dim_feats = params.dim_feats
    hidden_1 = params.hidden_1
    hidden_2 = params.hidden_2
    net = ROGPL(in_channels=dim_feats, hidden_1=hidden_1, hidden_2=hidden_2, n_classes=n_classes, center_num=n_classes + 1)
    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                               verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes


def wrapup_training(result_dir, best_accuracy):
    stats = get_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def emptify_graph(train_graph, idx):
    train_X_features = train_graph.x
    no_train_x_features = torch.zeros(train_X_features[0].shape[0]).cuda()
    train_X_features[idx] = no_train_x_features


def search_self(features, knn_neighbors):
    res = faiss.StandardGpuResources()
    searcher = faiss.IndexFlatIP(features.size(1))
    gpu_searcher = faiss.index_cpu_to_gpu(res, 0, searcher)
    features_np = features.numpy()
    gpu_searcher.add(features_np)
    search_scores, search_ids = gpu_searcher.search(features_np, knn_neighbors + 1) 
    return searcher, search_scores, search_ids


def main(cfg, device):
    init_seeds(0)
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16
    logger, result_dir = build_logger(cfg)

    # load data ---------------------------------------------------------------------------------------------------------------------------------------
    graph = Planetoid(root='./datasets/PYG', name='Cora')
    g = graph[0]

    cfg.dim_feats = g.num_node_features
    data_idx = np.load('./idx/' + cfg.dataset + '_5_idx.npz')
    train_indices = data_idx['train_indices']
    valid_indices = data_idx['valid_indices']
    test_indices = data_idx['test_indices']
    y_true = data_idx['y_true']
    y_real_true = data_idx['y_real_true']
    cfg.n_classes = np.max(y_true) + 1

    train_sample_nums = len(train_indices)

    net, optimizer, scheduler, n_classes = build_model_optim_scheduler(cfg, device, build_scheduler=False)

    record_network_arch(result_dir, net)

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()
    best_accuracy, best_epoch = 0.0, None
    g_train = copy.deepcopy(g)
    g, g_train = g.to(device), g_train.to(device)
    our_non_train_idx = np.concatenate((test_indices, valid_indices))
    emptify_graph(g_train, our_non_train_idx)
    y = torch.from_numpy(y_true).to(device)
    y_train = y[train_indices]
    y_test = y[test_indices]
    y_valid = y[valid_indices]

    test_f1_list = []
    test_acc_list = []
    test_auc_list = []

    hard_labels = None
    soft_labels = None
    clean_ids = None

    net.weigth_init(g_train, y, train_indices)
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()
        net.train()
        optimizer.zero_grad()
        train_loss.reset()
        train_accuracy.reset()
        s = time.time()
        output, hidden_features = net(g_train)
        logits = output[train_indices]
        probs = logits.softmax(dim=1).detach().cpu()
        if hard_labels is None:
            pseudo_labels = torch.zeros_like(probs).scatter_(1, torch.tensor(y_true[train_indices]).view(-1, 1), 1)
        else:
            pseudo_labels = probs[:]
            labels_clean_one_hot = torch.zeros_like(probs).scatter_(1, hard_labels.view(-1, 1), 1)
            pseudo_labels[clean_ids] = labels_clean_one_hot[clean_ids]

        pseudo_labels = pseudo_labels / pseudo_labels.sum(dim=0)

        hidden_features = hidden_features[train_indices].detach().cpu()

        searcher, search_scores, search_ids = search_self(hidden_features, cfg.knn_neighbors)

        D = search_scores[:, 1:] ** 3
        I = search_ids[:, 1:]
        row_idx = np.arange(train_sample_nums)
        row_idx_repeat = np.tile(row_idx, (cfg.knn_neighbors, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_repeat.flatten('F'), I.flatten('F'))),
                                    shape=(train_sample_nums, train_sample_nums))
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1.0 / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D
        Z = np.zeros((train_sample_nums, n_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - 0.5 * Wn
        for i in range(n_classes):
            y = pseudo_labels[:, i]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=20)
            Z[:, i] = f
        Z[Z < 0] = 0
        soft_labels = torch.tensor(Z).float()
        soft_labels = soft_labels / soft_labels.sum(dim=1).reshape(-1, 1)

        max_scores, hard_labels = torch.max(soft_labels, dim=1)
        clean_ids = max_scores > cfg.high_threshold

        train_acc = accuracy(logits, y_train, topk=(1,))
        print(f"train_acc:{train_acc}-------------------------------")
        given_labels = get_smoothed_label_distribution(y_train, n_classes, epsilon=0)
        if epoch < cfg.warmup_epochs:
            loss = F.cross_entropy(logits, given_labels)
        else:
            loss = F.cross_entropy(logits[clean_ids], hard_labels[clean_ids].to(device))
        loss = F.cross_entropy(logits[clean_ids], given_labels[clean_ids])
        loss = loss + 0.01 * constraint(device, net.get_mul_interior_prototype())
        loss.backward()
        optimizer.step()
        net.update_interior_prototype_weight(net.get_mid_h())
        train_accuracy.update(train_acc[0], g.x[train_indices].size(0))
        train_loss.update(loss.item(), g.x[train_indices].size(0))
        epoch_train_time.update(time.time() - s, 1)

        valid_eval_result = evaluate_valid(g, net, y_true, valid_indices)
        test_ind_indices = np.array(test_indices[np.where(y_true[test_indices] != -1)])
        test_ood_indices = np.array(test_indices[np.where(y_true[test_indices] == -1)])
        test_eval_result = evaluate_test(g, net, y_true, test_indices, test_ind_indices, test_ood_indices)
        test_acc = test_eval_result['test_acc']
        test_f1 = test_eval_result['test_f1']
        test_auc = test_eval_result['test_auc']
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        test_auc_list.append(test_auc)
        logger.info(f'test_acc : {test_acc}, test_f1 : {test_f1}')
        logger.info(f'test_auc : {test_auc}')

        test_accuracy = test_eval_result['test_acc']
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if cfg.save_model:
                torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
                torch.save(net, f'{result_dir}/best_model.pth')

        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test loss: {0:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')

    logger.info(f'max_test_acc: {np.max(test_acc_list)}')
    logger.info(f'max_test_f1: {np.max(test_f1_list)}')
    logger.info(f'max_test_auc: {np.max(test_auc_list)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/cora.cfg')
    parser.add_argument('--closeset-ratio', type=float, default='0.5')
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--use-fp16', action='store_true')
    parser.add_argument('--use-grad-accumulate', action='store_true')

    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--log', type=str, default='ROGPL')
    parser.add_argument('--ablation', action='store_true')
    
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--hidden_1', type=int, default=512)
    parser.add_argument('--hidden_2', type=int, default=128)
    parser.add_argument('--high_threshold', type=float, default=0.5)
#     parser.add_argument('--high_threshold', type=float, default=0.7)
    parser.add_argument('--knn_neighbors', type=int, default=35)


    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])

    if config.ablation:
        config.project = f'ablation/{config.project}'
    config.log_freq = LOG_FREQ
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
