from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    # general
    parser.add_argument('--task', type=str, default='link', choices=['link', 'link_pair'], help='Type of task.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')

    # dataset
    parser.add_argument('--dataset', type=str, default='grid',
                        choices=['grid', 'communities', 'ppi', 'email', 'protein'], help='Name of dataset.')
    parser.add_argument('--cache', action='store_true', help='Whether use cached dataset.')
    parser.add_argument('--remove_link_ratio', type=float, default=0.2, help='Validation and test set size.')
    parser.add_argument('--inductive', action='store_true', help='Inductive learning or transductive learning.')
    parser.add_argument('--permute', action='store_true', help='Whether permute anchor set subsets.')
    parser.add_argument('--feature_pre', action='store_true', help='Whether pre transform feature.')
    parser.add_argument('--K_hop_dist', default=-1, type=int,
                        help='K-hop shortest path distance. -1 means exact shortest path.')  # -1, 2

    # model
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=8)  # implemented via accumulating gradient
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--output_dim', type=int, default=32)
    parser.add_argument('--anchor_num', type=int, default=64)

    parser.add_argument('--epoch_num', type=int, default=2001)
    parser.add_argument('--repeat_num', type=int, default=10)  # 10
    parser.add_argument('--epoch_log', type=int, default=100)

    args = parser.parse_args()
    return args
