from collections.abc import Mapping

import torch

import dgl

class NeighborSamplerUni(dgl.dataloading.NeighborSampler):
    """ This sampler is a variant of NeighborSampler, make the `seed_nodes` is unique ..."""
    def __init__(self, fanouts, edge_dir='in', prob=None, mask=None, replace=False, prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None, output_device=None):
        super().__init__(fanouts, edge_dir, prob, mask, replace, prefetch_node_feats, prefetch_labels, prefetch_edge_feats, output_device)
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seed_nodes = seed_nodes.unique()
        return super().sample_blocks(g, seed_nodes, exclude_eids)