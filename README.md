# DGL Implementations of P-GNN

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
--task                str        Type of task.                           Default is 'link'.
--gpu                 int        GPU index.                              Default is -1, using CPU.
```

###### Dataset options

```
--dataset             str        The graph dataset name.                 Default is 'grid'.
--cache               bool       Whether use cached dataset.             Default is False.
--remove_link_ratio   float      Validation and test set size.           Default is 0.2, 0.1 for each set.
--inductive           bool       Inductive or transductive learning.     Default is False, transductive learning.
--feature_pre         bool       Whether pre transform feature.          Default is False.
--permute             bool       Whether permute anchor set subsets.     Default is True.
--K_hop_dist          int        K-hop shortest path distance.           Deafault is -1, -1 means exact shortest path.
```

###### Model options

```
--lr                  float      Adam optimizer learning rate.           Default is 0.01.
--batch_size          float      Batch Size.                             Default is 8.
--dropout             float      Dropout ration.                         Default is 0.5.
--feature_dim         int        Feature dimensionalities.               Default is 32.
--hidden_dim          int        Hidden layer dimensionalities.          Default is 32.
--output_dim          int        Output layer dimensionalities.          Default is 32.
--anchor_num          int        Number of Anchor sets.                  Default is 64.
--layer_num           int        Number of P-GNN layers.                 Default is 2.

--repeat_num          int        Number of experiments.                  Default is 2.
--epoch_num           int        Number of training epochs.              Default is 2001.
--epoch_log           int        Frequency of report result.             Default is 10.
```

###### Run

The following commands train a neural network and predict on the test set.

Training a P-GNN model on the dataset with default options.


```bash
# Communities transductive link prediction
python main.py --dataset communities --task link

# Communities inductive link prediction
python main.py --dataset communities --inductive --task link
```

Train a model with different model hyperparameters.

```bash
python main.py --dataset communities --task link --layer_num 1 --lr 0.1 --anchor_num 32 --dropout 0.2
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
