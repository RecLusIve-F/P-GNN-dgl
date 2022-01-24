import os
import time
import torch
import numpy as np
import torch.nn as nn
from model import PGNN
from random import shuffle
from args import make_args
from dataset import get_dataset
from sklearn.metrics import roc_auc_score
from utils import preselect_all_anchor, preselect_single_anchor


def get_loss(p, data, out, loss_func, device, out_act=None):
    edge_mask = np.concatenate((data[f'mask_link_positive_{p}'], data[f'mask_link_negative_{p}']), axis=-1)

    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask[0, :]).long().to(out.device))
    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask[1, :]).long().to(out.device))

    pred = torch.sum(nodes_first * nodes_second, dim=-1)

    label_positive = torch.ones([data[f'mask_link_positive_{p}'].shape[1], ], dtype=pred.dtype)
    label_negative = torch.zeros([data[f'mask_link_negative_{p}'].shape[1], ], dtype=pred.dtype)
    label = torch.cat((label_positive, label_negative)).to(device)
    loss = loss_func(pred, label)

    if out_act is not None:
        auc = roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

        return loss, auc

    return loss


def train_model(data_list, graphs, anchor_eids, dists_max_list, model, args, loss_func, optimizer, device):
    for idx, data in enumerate(data_list):
        data['graph'], data['anchor_eid'], data['dists_max'] = graphs[idx].to(device), anchor_eids[idx], dists_max_list[idx]
        data['graph'].edata['sp_dist'] = data['graph'].edata['sp_dist'].to(device)
        data['graph'].ndata['feat'] = data['graph'].ndata['feat'].to(device)
        out = model(data)
        # get_link_mask(data, re_split=False)  # resample negative links

        loss = get_loss('train', data, out, loss_func, device)

        # update
        loss.backward()
        if idx % args.batch_size == args.batch_size - 1:
            if args.batch_size > 1:
                # if this is slow, no need to do this normalization
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad /= args.batch_size
            optimizer.step()
            optimizer.zero_grad()


def eval_model(data_list, model, loss_func, out_act, device):
    model.eval()
    loss_train = 0
    loss_val = 0
    loss_test = 0

    auc_train = 0
    auc_val = 0
    auc_test = 0

    data_list_len = len(data_list)

    for idx, data in enumerate(data_list):
        out = model(data)

        # train
        tmp_loss, tmp_auc = get_loss('train', data, out, loss_func, device, out_act)
        loss_train += tmp_loss.cpu().data.numpy() / data_list_len
        auc_train += tmp_auc / data_list_len

        # val
        tmp_loss, tmp_auc = get_loss('val', data, out, loss_func, device, out_act)
        loss_val += tmp_loss.cpu().data.numpy() / data_list_len
        auc_val += tmp_auc / data_list_len

        # test
        tmp_loss, tmp_auc = get_loss('test', data, out, loss_func, device, out_act)
        loss_test += tmp_loss.cpu().data.numpy() / data_list_len
        auc_test += tmp_auc / data_list_len

    return loss_train, auc_train, loss_val, auc_val, loss_test, auc_test


def main():
    if not os.path.isdir('results'):
        os.mkdir('results')

    args = make_args()
    print(args)

    # set up gpu
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print('Using CPU')
    device = torch.device('cuda:' + str(args.cuda) if args.gpu else 'cpu')

    if args.dataset == 'All':
        if args.task == 'link':
            datasets_name = ['grid', 'communities', 'ppi']
        else:
            datasets_name = ['communities', 'email', 'protein']
    else:
        datasets_name = [args.dataset]

    for dataset_name in datasets_name:
        print(f'Dataset: {dataset_name}, Learning Type: {"Inductive" if args.rm_feature else "Transductive"}, '
              f'Task: {args.task}, Model layer num: {args.layer_num}-layer, Shortest Path Approximate: '
              f'{"Exact" if args.approximate == -1 else "Fast"}')
        results = []

        for repeat in range(args.repeat_num):
            result_val = []
            result_test = []
            time1 = time.time()
            data_list = get_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
            time2 = time.time()
            print(dataset_name, 'load time', time2 - time1)

            num_features = data_list[0]['feature'].shape[1]
            args.batch_size = min(args.batch_size, len(data_list))
            print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

            # data
            graphs = []
            anchor_eids = []
            dists_max_list = []
            for i, data in enumerate(data_list):
                if not args.permute:
                    g, anchor_eid, dists_max = preselect_single_anchor(data)
                    g = g * args.epoch_num
                    anchor_eid = anchor_eid * args.epoch_num
                    dists_max = dists_max * args.epoch_num
                else:
                    g, anchor_eid, dists_max = preselect_all_anchor(data, args)

                graphs.append(g)
                anchor_eids.append(anchor_eid)
                dists_max_list.append(dists_max)

            print('Preselect anchor_set Finished!')

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
                effective_len = len(data_list) // args.batch_size * len(data_list)

                g = [i[epoch] for i in graphs[:effective_len]]
                anchor_eid = [i[epoch] for i in anchor_eids[:effective_len]]
                dists_max = [i[epoch] for i in dists_max_list[:effective_len]]

                train_model(data_list[:effective_len], g, anchor_eid, dists_max, model, args, loss_func, optimizer,
                            device)

                loss_train, auc_train, loss_val, auc_val, loss_test, auc_test = eval_model(data_list, model, loss_func,
                                                                                           out_act, device)
                if auc_val > best_auc_val:
                    best_auc_val = auc_val
                    best_auc_test = auc_test

                if epoch % args.epoch_log == 0:
                    print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                          'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test),
                          'Best Val AUC: {:.4f}'.format(best_auc_val), 'Best Test AUC: {:.4f}'.format(best_auc_test))

                result_val.append(auc_val)
                result_test.append(auc_test)

            result_val = np.array(result_val)
            result_test = np.array(result_test)
            results.append(result_test[np.argmax(result_val)])

        results = np.array(results)
        results_mean = np.mean(results).round(6)
        results_std = np.std(results).round(6)
        print('-----------------Final-------------------')
        print(results_mean, results_std)

        with open(f'results/{dataset_name}_{"Inductive" if args.rm_feature else "Transductive"}_{args.task}_'
                  f'{args.layer_num}-layer_approximate{args.approximate}', 'w') as f:
            f.write('{}, {}\n'.format(results_mean, results_std))


if __name__ == '__main__':
    main()
