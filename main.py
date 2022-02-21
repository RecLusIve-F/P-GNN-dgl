import os
import dgl
import time
import torch
import numpy as np
import torch.nn as nn
from model import PGNN
from random import shuffle
from args import make_args
from tqdm.auto import tqdm
from dataset import get_dataset
from sklearn.metrics import roc_auc_score
from utils import preselect_all_anchor_parallel, preselect_single_anchor


import warnings
warnings.filterwarnings('ignore')


def get_loss(p, data, out, loss_func, device, out_act=None):
    edge_mask = np.concatenate((data['mask_link_positive_{}'.format(p)], data['mask_link_negative_{}'.format(p)]), axis=-1)

    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask[0, :]).long().to(out.device))
    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask[1, :]).long().to(out.device))

    pred = torch.sum(nodes_first * nodes_second, dim=-1)

    label_positive = torch.ones([data['mask_link_positive_{}'.format(p)].shape[1], ], dtype=pred.dtype)
    label_negative = torch.zeros([data['mask_link_negative_{}'.format(p)].shape[1], ], dtype=pred.dtype)
    label = torch.cat((label_positive, label_negative)).to(device)
    loss = loss_func(pred, label)

    if out_act is not None:
        auc = roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

        return loss, auc

    return loss


def train_model(data_list, model, args, loss_func, optimizer, device, anchor_sets):
    if anchor_sets is None:
        graphs, anchor_eids, dists_max, edge_weights = preselect_all_anchor_parallel(data=None, args=args,
                                                                                     data_list=data_list)
    else:
        graphs, anchor_eids, dists_max, edge_weights = anchor_sets

    graph_data_list = []
    for idx, data in enumerate(tqdm(data_list, leave=False)):
        g_data = {}
        g = dgl.graph(graphs[idx])
        g.ndata['feat'] = torch.as_tensor(data['feature'], dtype=torch.float)
        g.edata['sp_dist'] = torch.as_tensor(edge_weights[idx], dtype=torch.float)
        g_data['graph'], g_data['anchor_eid'], g_data['dists_max'] = g.to(device), anchor_eids[idx], dists_max[idx]

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
        graph_data_list.append(g_data)

    return graph_data_list


def eval_model(data_list, graph_data_list, model, loss_func, out_act, device):
    model.eval()
    loss_train = 0
    loss_val = 0
    loss_test = 0

    auc_train = 0
    auc_val = 0
    auc_test = 0

    data_list_len = len(data_list)

    for idx, data in enumerate(data_list):
        out = model(graph_data_list[idx])

        # train loss and auc
        tmp_loss, tmp_auc = get_loss('train', data, out, loss_func, device, out_act)
        loss_train += tmp_loss.cpu().data.numpy() / data_list_len
        auc_train += tmp_auc / data_list_len

        # val loss and auc
        tmp_loss, tmp_auc = get_loss('val', data, out, loss_func, device, out_act)
        loss_val += tmp_loss.cpu().data.numpy() / data_list_len
        auc_val += tmp_auc / data_list_len

        # test loss and auc
        tmp_loss, tmp_auc = get_loss('test', data, out, loss_func, device, out_act)
        loss_test += tmp_loss.cpu().data.numpy() / data_list_len
        auc_test += tmp_auc / data_list_len

    return loss_train, auc_train, loss_val, auc_val, loss_test, auc_test


def main():
    # The mean and standard deviation of the experimental results are stored
    # in the 'results' folder with the file name of the experimental parameter.
    if not os.path.isdir('results'):
        os.mkdir('results')

    args = make_args()

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    dataset_name = args.dataset

    print('Dataset: {}'.format(dataset_name), 'Learning Type: {}'.format(['Transductive', 'Inductive'][args.inductive]),
          'Task: {}'.format(args.task), 'Model layer num: {}-layer'.format(args.layer_num),
          'Shortest Path Approximate: {}'.format(['Fast', 'Exact'][args.K_hop_dist == -1]))

    results = []

    for repeat in range(args.repeat_num):
        time1 = time.time()
        data_list = get_dataset(args, dataset_name, remove_feature=args.inductive)
        time2 = time.time()
        print(dataset_name, 'Dateset load time', time2 - time1)

        num_features = data_list[0]['feature'].shape[1]
        args.batch_size = min(args.batch_size, len(data_list))

        # data
        graphs = []
        anchor_eids = []
        dists_max_list = []
        edge_weights = []
        anchor_sets = None

        if dataset_name not in ['ppi', 'protein']:
            for i, data in enumerate(data_list):
                if not args.permute:
                    g, anchor_eid, dists_max, edge_weight = preselect_single_anchor(data)
                    g = g * args.epoch_num
                    anchor_eid = anchor_eid * args.epoch_num
                    dists_max = dists_max * args.epoch_num
                    edge_weight = edge_weight * args.epoch_num
                else:
                    g, anchor_eid, dists_max, edge_weight = preselect_all_anchor_parallel(data, args)

                graphs.append(g)
                anchor_eids.append(anchor_eid)
                dists_max_list.append(dists_max)
                edge_weights.append(edge_weight)

        # model
        input_dim = num_features
        output_dim = args.output_dim
        model = PGNN(input_dim=input_dim, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
                     output_dim=output_dim, feature_pre=args.feature_pre, layer_num=args.layer_num,
                     dropout=args.dropout).to(device)

        # loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        loss_func = nn.BCEWithLogitsLoss()
        out_act = nn.Sigmoid()

        best_auc_val = -1
        best_auc_test = -1
        for epoch in range(args.epoch_num):
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            model.train()
            optimizer.zero_grad()
            shuffle(data_list)

            if dataset_name not in ['ppi', 'protein']:
                g = [i[epoch] for i in graphs]
                anchor_eid = [i[epoch] for i in anchor_eids]
                dists_max = [i[epoch] for i in dists_max_list]
                edge_weight = [i[epoch] for i in edge_weights]
                anchor_sets = [g, anchor_eid, dists_max, edge_weight]

            graph_data_list = train_model(data_list, model, args, loss_func, optimizer, device, anchor_sets)

            loss_train, auc_train, loss_val, auc_val, loss_test, auc_test = eval_model(data_list, graph_data_list,
                                                                                       model, loss_func, out_act,
                                                                                       device)
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

    with open('results/{}_{}_{}_{}_{}.txt'.format(dataset_name, ['Transductive', 'Inductive'][args.inductive], args.task,
                                              args.layer_num, args.K_hop_dist), 'w') as f:
        f.write('{}, {}\n'.format(results_mean, results_std))


if __name__ == '__main__':
    main()
