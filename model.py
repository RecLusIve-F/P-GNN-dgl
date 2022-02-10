import torch
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import torch.nn.functional as F


class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=False):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden_u = nn.Linear(input_dim, output_dim)
        self.linear_hidden_v = nn.Linear(input_dim, output_dim)
        self.linear_out_position = nn.Linear(output_dim, 1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, graph, feature, anchor_eid, dists_max):
        with graph.local_scope():
            if self.dist_trainable:
                dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()
                graph.edata['sp_dist'] = dists_max

            u_feat = self.linear_hidden_u(feature)
            v_feat = self.linear_hidden_v(feature)
            graph.srcdata.update({'u_feat': u_feat})
            graph.dstdata.update({'v_feat': v_feat})

            graph.apply_edges(fn.u_mul_e('u_feat', 'sp_dist', 'u_message'))
            graph.apply_edges(fn.v_add_e('v_feat', 'u_message', 'message'))

            messages = graph.edata.pop('message')
            messages = torch.index_select(messages, 0, torch.as_tensor(anchor_eid, dtype=torch.long).to(feature.device)).\
                reshape(dists_max.shape[0], dists_max.shape[1], messages.shape[-1])

            messages = self.act(messages)  # n*m*d

            out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
            out_structure = torch.mean(messages, dim=1)  # n*d

            return out_position, out_structure


# Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre=True, layer_num=2, dropout=0.5):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = nn.Dropout(dropout)
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num > 1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for _ in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def forward(self, data):
        x = data['graph'].ndata['feat']
        graph = data['graph']
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(graph, x, data['anchor_eid'], data['dists_max'])
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        x = self.dropout(x)
        for i in range(self.layer_num - 2):
            _, x = self.conv_hidden[i](graph, x, data['anchor_eid'], data['dists_max'])
            # x = F.relu(x) # Note: optional!
            x = self.dropout(x)
        x_position, x = self.conv_out(graph, x, data['anchor_eid'], data['dists_max'])
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position
