import os
import dgl
import torch
import numpy as np
import torch.nn as nn
from model import PGNN
from tqdm.auto import tqdm
from dataset import get_dataset
from sklearn.metrics import roc_auc_score
from utils import preselect_anchor

import warnings
warnings.filterwarnings('ignore')

def get_loss(p, data, out, loss_func, device):
    edge_mask = np.concatenate((data['positive_edges_{}'.format(p)], data['negative_edges_{}'.format(p)]), axis=-1)

    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask[0, :]).long().to(out.device))
    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask[1, :]).long().to(out.device))

    pred = torch.sum(nodes_first * nodes_second, dim=-1)

    label_positive = torch.ones([data['positive_edges_{}'.format(p)].shape[1], ], dtype=pred.dtype)
    label_negative = torch.zeros([data['negative_edges_{}'.format(p)].shape[1], ], dtype=pred.dtype)
    label = torch.cat((label_positive, label_negative)).to(device)
    loss = loss_func(pred, label)

    auc = roc_auc_score(label.flatten().cpu().numpy(), torch.sigmoid(pred).flatten().data.cpu().numpy())

    return loss, auc

def train_model(data, model, args, loss_func, optimizer, device, anchor_sets):
    graph, anchor_eid, dists_max, edge_weights = anchor_sets

    g_data = {}
    g = dgl.graph(graph)
    g.ndata['feat'] = torch.as_tensor(data['feature'], dtype=torch.float)
    g.edata['sp_dist'] = torch.as_tensor(edge_weights, dtype=torch.float)
    g_data['graph'], g_data['anchor_eid'], g_data['dists_max'] = g.to(device), anchor_eid, dists_max

    out = model(g_data)

    loss = get_loss('train', data, out, loss_func, device)

    loss.backward()
    if idx % args.batch_size == args.batch_size - 1:
        if args.batch_size > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= args.batch_size
        optimizer.step()
        optimizer.zero_grad()

    return g_data

def eval_model(data_list, graph_data_list, model, loss_func, device):
    model.eval()
    loss_train = 0
    loss_val = 0
    loss_test = 0

    auc_train = 0
    auc_val = 0
    auc_test = 0

    out = model(graph_data_list[idx])

    # train loss and auc
    tmp_loss, tmp_auc = get_loss('train', data, out, loss_func, device)
    loss_train += tmp_loss.cpu().data.numpy()
    auc_train += tmp_auc

    # val loss and auc
    tmp_loss, tmp_auc = get_loss('val', data, out, loss_func, device)
    loss_val += tmp_loss.cpu().data.numpy()
    auc_val += tmp_auc

    # test loss and auc
    tmp_loss, tmp_auc = get_loss('test', data, out, loss_func, device)
    loss_test += tmp_loss.cpu().data.numpy()
    auc_test += tmp_auc

    return loss_train, auc_train, auc_val, auc_test

def main(args):
    # The mean and standard deviation of the experiment results
    # are stored in the 'results' folder
    if not os.path.isdir('results'):
        os.mkdir('results')

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print('Learning Type: {}'.format(['Transductive', 'Inductive'][args.inductive]),
          'Task: {}'.format(args.task))

    results = []

    for repeat in range(args.repeat_num):
        data = get_dataset(args)

        # data
        g, anchor_eid, dists_max, edge_weight = preselect_anchor(data, args)

        # model
        model = PGNN(input_dim=data['feature'].shape[1]).to(device)

        # loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        loss_func = nn.BCEWithLogitsLoss()

        best_auc_val = -1
        best_auc_test = -1



        for epoch in range(args.epoch_num):
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            model.train()
            optimizer.zero_grad()

            anchor_sets = [g, anchor_eid, dists_max, edge_weight]
            graph_data_list = train_model(data, model, args, loss_func, optimizer, device, anchor_sets)

            loss_train, auc_train, auc_val, auc_test = eval_model(
                data_list, graph_data_list, model, loss_func, device)
            if auc_val > best_auc_val:
                best_auc_val = auc_val
                best_auc_test = auc_test

            if epoch % args.epoch_log == 0:
                print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                      'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test),
                      'Best Val AUC: {:.4f}'.format(best_auc_val), 'Best Test AUC: {:.4f}'.format(best_auc_test))

        results.append(best_auc_test)

    results = np.array(results)
    results_mean = np.mean(results).round(6)
    results_std = np.std(results).round(6)
    print('-----------------Final-------------------')
    print(results_mean, results_std)

    with open('results/{}_{}_{}.txt'.format(['Transductive', 'Inductive'][args.inductive], args.task,
                                            args.k_hop_dist), 'w') as f:
        f.write('{}, {}\n'.format(results_mean, results_std))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='link', choices=['link', 'link_pair'])
    parser.add_argument('--inductive', action='store_true',
                        help='Inductive learning or transductive learning')
    parser.add_argument('--k_hop_dist', default=-1, type=int,
                        help='K-hop shortest path distance, -1 means exact shortest path.')

    parser.add_argument('--epoch_num', type=int, default=2001)
    parser.add_argument('--repeat_num', type=int, default=10)
    parser.add_argument('--epoch_log', type=int, default=10)

    args = parser.parse_args()
    main(args)
