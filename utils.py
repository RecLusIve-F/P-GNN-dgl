import dgl
import torch
import random
import numpy as np
import networkx as nx
import multiprocessing as mp


def deduplicate_edges(edges):
    edges_new = np.zeros((2, edges.shape[1] // 2), dtype=int)
    # add none self edge
    j = 0
    skip_node = set()  # node already put into result
    for i in range(edges.shape[1]):
        if edges[0, i] < edges[1, i]:
            edges_new[:, j] = edges[:, i]
            j += 1
        elif edges[0, i] == edges[1, i] and edges[0, i] not in skip_node:
            edges_new[:, j] = edges[:, i]
            skip_node.add(edges[0, i])
            j += 1

    return edges_new


def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1, :]), axis=-1)


# # approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negative_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0], mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes, num_nodes)
    prob = num_negative_edges / (num_nodes * num_nodes - mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp < prob))
    return mask_link_negative


# exact version, slower
def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negative_edges):
    mask_link_positive_set = []
    mask_link_positive = duplicate_edges(mask_link_positive)
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:, i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2, num_negative_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negative_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes, size=(2,), replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:, i] = mask_temp
                break

    return mask_link_negative


def resample_edge_mask_link_negative(data, is_approximate=False):
    sample_function = [get_edge_mask_link_negative, get_edge_mask_link_negative_approximate][is_approximate]
    tmp_links = data['mask_link_positive']

    data['mask_link_negative_train'] = sample_function(tmp_links, num_nodes=data['num_nodes'],
                                                       num_negative_edges=data['mask_link_positive_train'].shape[1])

    tmp_links = np.concatenate([tmp_links, data['mask_link_negative_train']], axis=1)

    data['mask_link_negative_val'] = sample_function(tmp_links, num_nodes=data['num_nodes'],
                                                     num_negative_edges=data['mask_link_positive_val'].shape[1])

    tmp_links = np.concatenate([tmp_links, data['mask_link_negative_val']], axis=1)

    data['mask_link_negative_test'] = sample_function(tmp_links, num_nodes=data['num_nodes'],
                                                      num_negative_edges=data['mask_link_positive_test'].shape[1])
    return data


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        idx = 0
        for i in range(e):
            idx = i
            node1 = edges[0, i]
            node2 = edges[1, i]
            if node_count[node1] > 1 and node_count[node2] > 1:  # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(idx + 1, e))
        index_test = index_val[:len(index_val) // 2]
        index_val = index_val[len(index_val) // 2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1 - remove_ratio) * e)
        split2 = int((1 - remove_ratio / 2) * e)
        edges_train = edges[:, :split1]
        edges_val = edges[:, split1:split2]
        edges_test = edges[:, split2:]

    return edges_train, edges_val, edges_test


def get_link_mask(data, remove_ratio=0.2, re_split=True, infer_link_positive=True, is_approximate=False):
    if re_split:
        if infer_link_positive:
            data['mask_link_positive'] = deduplicate_edges(data['edge_index'].numpy())
        data['mask_link_positive_train'], data['mask_link_positive_val'], data['mask_link_positive_test'] = split_edges(
            data['mask_link_positive'], remove_ratio)
    data = resample_edge_mask_link_negative(data, is_approximate)
    return data


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(
                                    graph,
                                    nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))],
                                    cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
    """
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    """
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array


def construct_sp_graph(feature, dists_max, dists_argmax, device):
    src = []
    dst = []
    real_src = []
    real_dst = []
    edge_weight = []
    for i in range(dists_max.shape[0]):
        tmp_dists_argmax, tmp_dists_argmax_idx = np.unique(dists_argmax[i, :], True)
        src.extend([i] * tmp_dists_argmax.shape[0])
        real_src.extend([i] * dists_argmax[i, :].shape[0])
        real_dst.extend(list(dists_argmax[i, :].numpy()))
        dst.extend(list(tmp_dists_argmax))
        edge_weight.extend(list(dists_max[i, tmp_dists_argmax_idx]))
    eid_dict = {(u, v): i for i, (u, v) in enumerate(list(zip(dst, src)))}
    anchor_eid = [eid_dict.get((u, v)) for u, v in list(zip(real_dst, real_src))]
    g = dgl.graph((dst, src)).to(device)
    g.edata['sp_dist'] = torch.tensor(edge_weight).float().to(device)
    g.ndata['feat'] = torch.tensor(feature).float().to(device)

    return g, anchor_eid


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c * m)
    anchor_set_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            anchor_set_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchor_set_id


def get_dist_max(anchor_set_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchor_set_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchor_set_id))).long().to(device)
    for i in range(len(anchor_set_id)):
        temp_id = torch.as_tensor(anchor_set_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):
    data['anchor_size_num'] = anchor_size_num
    data['anchor_set'] = []
    anchor_num_per_size = anchor_num // anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2 ** (i + 1) - 1
        anchors = np.random.choice(data['num_nodes'], size=(layer_num, anchor_num_per_size, anchor_size),
                                   replace=True)
        data['anchor_set'].append(anchors)
    data['anchor_set_indicator'] = np.zeros((layer_num, anchor_num, data['num_nodes']), dtype=int)

    anchor_set_id = get_random_anchorset(data['num_nodes'], c=1)
    data['dists_max'], data['dists_argmax'] = get_dist_max(anchor_set_id, data['dists'], 'cpu')
    data['graph'], data['anchor_eid'] = construct_sp_graph(data['feature'], data['dists_max'], data['dists_argmax'],
                                                           device)
    return data
