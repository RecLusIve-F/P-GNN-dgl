# DGL Implementations of MixHop

This DGL example implements the GNN model proposed in the paper [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b/you19b.pdf). For the original implementation, see [here](https://github.com/JiaxuanYou/P-GNN).

Contributor: [RecLusIve-F](https://github.com/RecLusIve-F)

### Requirements

The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl 0.7.2
numpy 1.21.2
torch 1.10.1
networkx 2.6.3
scikit-learn 1.0.2
```

### Usage

###### General options

```
--task                str        Task type.                          Default is 'link_pair'.
--dataset             str        The graph dataset name.             Default is 'communities'.
--gpu(--cpu)          bool       Whether use gpu.                    Default is False.
--cache(_no)          bool       Whether use cache.                  Default is False.
--cuda                str        GPU index.                          Defaule is '0'.
```

###### Dataset options

```
--remove_link_ratio   float      Validation and test set ratio.      Default is 0.2.
--rm_feature(_no)     bool       Whether rm_feature.                 Default is True.
--feature_pre(_no)    bool       Whether pre transform feature.      Default is True.
--permute(_no)        bool       Whether permute subsets.            Default is True.
--approximate         int        K-hop shortest path distance.       Deafault is -1.
```

###### Model options

```
--epoch_num           int        Number of training epochs.          Default is 2001.
--epoch_log           int        Frequency of report result.         Default is 10.
--lr                  float      Adam optimizer learning rate.       Default is 0.01.
--batch_size          float      Batch Size.                         Default is 8.
--repeat_num          int        Number of experiments.              Default is 2.
--feature_dim         int        Feature dimensionalities.           Default is 32.
--hidden_dim          int        Hidden layer dimensionalities.      Default is 32.
--output_dim          int        Output layer dimensionalities.      Default is 32.
--anchor_num          int        Number of Anchor sets.              Default is 64.
--layer_num           int        Number of P-GNN layers.             Default is 2.
--dropout(_no)        bool       Whether dropout, default 0.5        Default is True.
```

###### Run

The following commands learn a neural network and predict on the test set.
Training a P-GNN model on the default dataset with default options.

```bash
python main.py
```

Training a P-GNN model on the different dataset with default options.

```bash
# Communities transductive link prediction
python main.py --dataset communities

# Communities inductive link prediction
python main.py --dataset communities --rm_feature
```

Train a model with different model hyperparameters.

```bash
python main.py --num-layers 1 --lr 0.001 --anchor_num 32
```

### Performance

###### Link prediction(Grid-T and Communities-T refer to the transductive learning setting of Grid and Communities)

|             Dataset              |    Grid-T     | Communities-T |     Grid      |  Communities  |      PPI      |
| :------------------------------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| ROC AUC ( P-GNN-E-2L in Table 1) | 0.834 ± 0.099 | 0.988 ± 0.003 | 0.940 ± 0.027 | 0.985 ± 0.008 | 0.808 ± 0.003 |
|    ROC AUC (DGL: P-GNN-E-2L)     | 0.657 ± 0.034 | 0.965 ± 0.025 | 0.923 ± 0.027 | 0.991 ± 0.040 |       —       |

###### Link pair prediction

|             Dataset              | Communities |     Email     |    Protein    |
| :------------------------------: | :---------: | :-----------: | :-----------: |
| ROC AUC ( P-GNN-E-2L in Table 1) | 1.0 ± 0.001 | 0.640 ± 0.029 | 0.631 ± 0.175 |
|    ROC AUC (DGL: P-GNN-E-2L)     | 1.0 ± 0.001 | 0.654 ± 0.114 |       —       |
