import logging
import os
import warnings
from typing import List, Sequence

import numpy as np
import torch
import random

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn as nn


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def jaccard_similarity(set1, set2):
    # 计算交集
    intersection = len(set1.intersection(set2)) - len(set1 - set2) - len(set2 - set1)
    # 计算并集
    union = len(set1.union(set2))
    # 计算Jaccard相似系数
    similarity = intersection / union
    return similarity

def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()

def random_walk(graph, start_node, steps):
    current_node = start_node
    node_sequence = [current_node]

    for _ in range(steps):
        if current_node in graph:
            neighbors = graph[current_node]
        else:
            neighbors = []
        if len(neighbors) > 0 and random.randint(1, 5) < 4:
            current_node = random.choice(neighbors)
            node_sequence.append(current_node)
        else:
            node_sequence.append(current_node)
    return node_sequence

def co_occurrence_frequency(matrix):
    m, n = matrix.shape
    co_occurrence_matrix = np.zeros((n, n), dtype=int)

    for row in matrix:
        non_zero_indices = np.where(row != 0)[0]
        for i in range(len(non_zero_indices)):
            for j in range(i+1, len(non_zero_indices)):
                co_occurrence_matrix[non_zero_indices[i], non_zero_indices[j]] += 1
                co_occurrence_matrix[non_zero_indices[j], non_zero_indices[i]] += 1
    row_sums = co_occurrence_matrix.sum(axis=1)
    co_occurrence_matrix = co_occurrence_matrix / (row_sums[:, np.newaxis]+0.01)
    np.fill_diagonal(co_occurrence_matrix, 1)
    return torch.tensor(co_occurrence_matrix).to(torch.float32)


def cluster_pic_show(data, cluster=6):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    # 使用t-SNE降维到二维
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(data)
    # 使用K-means对降维后的数据进行聚类
    kmeans = KMeans(n_clusters=cluster)
    labels = kmeans.fit_predict(data_tsne)
    # 绘制聚类结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('PCA')
    plt.subplot(1, 2, 2)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('t-SNE')
    plt.show()

def select_negative_samples(label: torch.Tensor, negative_sample_ratio: int = 5):
    r"""select negative samples in training stage.

    Args:
        label (List[np.ndarray]): Label indicating the APIs called by mashup.
        negative_sample_ratio (int): Ratio of negative to positive in the training stage. (default: :obj:`5`)

    Returns:
        indices of positive samples, indices of negative samples, indices of all samples, and new label.
    """
    num_candidate = label.size(0)
    positive_idx = label.nonzero(as_tuple=True)[0]
    if len(positive_idx) > 0:
        positive_idx = positive_idx.cpu().numpy()
    else:
        positive_idx = torch.tensor([0], dtype=torch.int64)
    negative_idx = np.random.choice(np.delete(np.arange(num_candidate), positive_idx),
                                    size=min(negative_sample_ratio * len(positive_idx), len(np.delete(np.arange(num_candidate), positive_idx))), replace=False)
    sample_idx = np.concatenate((positive_idx, negative_idx), axis=None)
    label_new = torch.tensor([1] * len(positive_idx) + [0] * len(negative_idx), dtype=torch.float32)
    return positive_idx, negative_idx, sample_idx, label_new.cuda()

def initialize_weights(m, method='uniform'):
    if isinstance(m, nn.Linear):
        if method == 'uniform':
            # 1. 均匀分布初始化
            nn.init.uniform_(m.weight, -0.1, 0.1)
            nn.init.uniform_(m.bias, -0.1, 0.1)
        elif method == 'normal':
            # 2. 正态分布初始化
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.normal_(m.bias, mean=0.0, std=0.02)
        elif method == 'xavier':
            # 3. Xavier初始化
            nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_normal_(m.weight)
        elif method == 'he':
            # 4. He初始化
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif method == 'constant':
            # 5. 常数初始化
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.1)
        elif method == 'zero':
            # 6. 零初始化
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif method == 'orthogonal':
            # 7. 正交初始化
            nn.init.orthogonal_(m.weight)
            nn.init.orthogonal_(m.bias)
        elif method == 'sparse':
            # 8. 稀疏初始化
            nn.init.sparse_(m.weight, sparsity=0.1)
            nn.init.sparse_(m.bias, sparsity=0.1)
        else:
            raise ValueError('Invalid initialization method: %s' % method)


def kmeans_clustering(S, num_clusters, num_iters=50):
    """
    S: (N, D) service embeddings (torch tensor or numpy)
    returns:
      centers: (K, D)
      assignments: (N,) in [0, K)
    """
    # simple torch k-means (offline; sklearn 也可以)
    N, D = S.shape
    idx = torch.randperm(N)[:num_clusters]
    centers = S[idx].clone()
    assign = None
    for _ in range(num_iters):
        # assignment
        dist = torch.cdist(S, centers)           # (N, K)
        assign = dist.argmin(dim=1)              # (N,)

        # update
        new_centers = []
        for k in range(num_clusters):
            mask = (assign == k)
            if mask.any():
                new_centers.append(S[mask].mean(dim=0))
            else:
                new_centers.append(centers[k])
        centers = torch.stack(new_centers, dim=0)

    return centers, assign


