import numpy as np
import dgl 
import torch as th
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.datamodules.dataset.WSDreamStaticDataset import WSDreamStaticDataset
from src.datamodules.UniNeighborSampler import NeighborSamplerUni

class QoSGNNWSDreamDatamodule(pl.LightningDataModule):
    """ Datamodule for QoSGNN `FIXME: paper_url`
    Parameters
    ----------
    qos: str
        Specify which QoS Record will be used.
    u_fanouts: list[int]
        List of neighbors to sample user-user edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
    s_fanouts: list[int]
        List of neighbors to sample service-service edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
    batch_size: int
        Num of QoS record in a mini batch.
    density: float
        Data density.
    raw_dir : str
        Specifying the directory that will store the downloaded data
        or the directory that already stores the input data.
        Default: ~/.dgl/
    force_reload : bool
        Wether to reload the dataset. Default: False
    verbose : bool
        Wether to print out progress information. Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """
    def __init__(self, qos, u_fanouts, s_fanouts, batch_size = 1024, density = 0.025, raw_dir=None, force_reload=False, verbose=False, transform=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = WSDreamStaticDataset(qos, density, raw_dir, force_reload, verbose, transform)
        self.u_fanouts = u_fanouts
        self.s_fanouts = s_fanouts
        self.batch_size = batch_size
        
    def setup(self, stage: str = None):
        qos = self.dataset.graph.edges['invoke'].data['qos'].view(-1,).numpy()
        user, service = self.dataset.graph.all_edges(form='uv', etype=('user','invoke', 'service'))
        user, service = user.numpy(), service.numpy()
        
        self.u_graph = dgl.edge_type_subgraph(self.dataset.graph, etypes=["impacts_uu"])
        self.s_graph = dgl.edge_type_subgraph(self.dataset.graph, etypes=["impacts_ss"])
        # Assign train/val split(s) for use in Dataloaders.
        
        train_mask = self.dataset.graph.edges['invoke'].data['train_edge_mask'].view(-1,).numpy()
        valid_mask = self.dataset.graph.edges['invoke'].data['valid_edge_mask'].view(-1,).numpy()
        train_idx, valid_idx = np.nonzero(train_mask == 1), np.nonzero(valid_mask == 1)
        self.user_train, self.service_train, self.qos_train = user[train_idx], service[train_idx], qos[train_idx]
        self.user_valid, self.service_valid, self.qos_valid = user[valid_idx], service[valid_idx], qos[valid_idx]
        
        # Assign test split(s) for use in Dataloaders.
        test_mask = self.dataset.graph.edges['invoke'].data['test_edge_mask'].view(-1,).numpy()
        test_idx = np.nonzero(test_mask == 1)
        self.user_test, self.service_test, self.qos_test = user[test_idx], service[test_idx], qos[test_idx]

        print(len(self.qos_train), len(self.qos_valid), len(self.qos_test))
    def train_dataloader(self):
        assert len(self.user_train) == len(self.service_train) == len(self.qos_train)
        # NOTE: We need three dataloaders: sampled user graph dataloader, sampled service graph dataloader and qos record dataloader.
        #       The first two dataloader should based on the last dataloader (i.e., seed_nodes are from qos records). We need to
        #       make sure three dataloader in consistent. Thus we set `shuffle=Fales` in three dataloaders and we force reload
        #       train_dataloader each epoch.
        p = np.random.permutation(len(self.user_train))
        user_nid, service_nid, qos = self.user_train[p], self.service_train[p], self.qos_train[p]
        u_sampler, s_sampler = NeighborSamplerUni(self.u_fanouts), NeighborSamplerUni(self.s_fanouts)
        u_dataloader = dgl.dataloading.DataLoader(self.u_graph, user_nid, u_sampler, batch_size=self.batch_size, shuffle=False, drop_last=False)
        s_dataloader = dgl.dataloading.DataLoader(self.s_graph, service_nid, s_sampler, batch_size=self.batch_size, shuffle=False, drop_last=False)
        qos_dataloader = DataLoader(list(zip(user_nid, service_nid, qos)), shuffle=False, batch_size=self.batch_size)
        return {
            "ug": u_dataloader,
            "sg": s_dataloader,
            "qr": qos_dataloader
        }
    
    def val_dataloader(self):
        assert len(self.user_valid) == len(self.service_valid) == len(self.qos_valid)
        qos_dataloader = DataLoader(list(zip(self.user_valid, self.service_valid, self.qos_valid)), shuffle=False, batch_size=self.batch_size)
        return qos_dataloader
    
    def test_dataloader(self):
        assert len(self.user_test) == len(self.service_test) == len(self.qos_test)
        qos_dataloader = DataLoader(list(zip(self.user_test, self.service_test, self.qos_test)), shuffle=False, batch_size=self.batch_size)
        return qos_dataloader