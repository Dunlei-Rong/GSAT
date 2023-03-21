import os
import pandas as pd
import numpy as np
import torch as th
import dgl
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
from clearml import Dataset


class WSDreamStaticDataset(DGLBuiltinDataset):
    """ The most popular (static) QoS prediction dataset.
    
    Parameters
    ----------
    qos: str
        Specify which QoS Record will be used.
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
    _url = "https://github.com/icecity96/ServiceComputingDataset/raw/main/WSDream-static.zip"
    _sha1_str = 'e4de41e252483297fe3822af6f4718a783d1c04d'
    
    def __init__(self, qos, density = 0.025, raw_dir=None, force_reload=False, verbose=False, transform=None):
        assert qos in ["rt", "tp"], ("The QoS must be one of `rt` or `tp` in lower case!")
        self.qos = qos
        self.density = density
        super(WSDreamStaticDataset, self).__init__(name="wsdream_static", url=self._url, raw_dir=raw_dir,
                                                   force_reload=force_reload,  verbose=verbose, transform=transform)
    
    
    def download(self):
        """ FIXME: HIT-ICES inter implement using clearML. When Publish Code, please modify this part!
        """
        dataset = Dataset.get(dataset_id="586f4c66aa5f43e68dccf046819d6a86")
        dataset.get_mutable_local_copy(target_folder=self.save_dir)
    
    def process(self):
        if self.qos == "rt":
            qos_value_path = os.path.join(self.raw_dir, 'rtMatrix.txt')
        else:
            qos_value_path = os.path.join(self.raw_dir, 'tpMatrix.txt')
        
        qos_record = pd.read_csv(qos_value_path, header=None, sep='\t').to_numpy()[:,:-1]
        uu_src, uu_dst = (np.ones((qos_record.shape[0], qos_record.shape[0]))).nonzero()
        us_src, us_dst = (qos_record > 0).nonzero()
        qos_value = qos_record[(us_src, us_dst)]
        ss_src, ss_dst = (np.ones((qos_record.shape[1], qos_record.shape[1]))).nonzero()
        
        graph = dgl.heterograph({
            ('user', 'impacts_uu', 'user'): (uu_src, uu_dst),
            ('user', 'invoke', 'service'): (us_src, us_dst),
            ('service', 'impacts_ss', 'service'): (ss_src, ss_dst)
        })
        
        self._num_users = graph.num_nodes('user')
        self._num_services = graph.num_nodes('service')
        
        
        # set qos record graph attributes.
        # TODO: we can make code more clear by extracting the following snip as a seperate func `_random_split(self, graph, seed=234)`
        graph.edges['invoke'].data['qos'] = th.tensor(qos_value).view(-1, 1)
        num_qos_record = graph.num_edges('invoke')
        indices = list(range(num_qos_record))
        np.random.shuffle(indices)
        valid_split = int(np.floor(self.density * 0.1 * num_qos_record))
        test_split = int(np.floor((self.density * 0.1 + 1 - self.density) * num_qos_record))
        valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]
        train_mask = th.zeros((num_qos_record, 1), dtype=th.bool)
        train_mask[train_idx] = True
        valid_mask = th.zeros((num_qos_record, 1), dtype=th.bool)
        valid_mask[valid_idx] = True
        test_mask = th.zeros((num_qos_record, 1), dtype=th.bool)
        test_mask[test_idx] = True
        graph.edges['invoke'].data['train_edge_mask'] = train_mask
        graph.edges['invoke'].data['valid_edge_mask'] = valid_mask
        graph.edges['invoke'].data['test_edge_mask'] = test_mask
        
        self._num_qos_records =  num_qos_record
        self._num_train_qos_record, self._num_valid_qos_record, self._num_test_qos_record = len(train_idx), len(valid_idx), len(test_idx)
        
        # set users and services contexts.
        user_contexs_path = os.path.join(self.raw_dir, "userlist.txt")
        user_pd = pd.read_csv(user_contexs_path, header=None, sep="\t", skiprows=2,
                        names=["uid", "ip", "RS", "ipno", "AS", "Latitude", "Longitude"], encoding='latin-1')
        user_pd = user_pd.fillna(0)
        user_pd["RS"] = pd.Categorical(user_pd["RS"]).codes
        user_pd["AS"] = pd.Categorical(user_pd["AS"]).codes
        # here we only use RS and AS... you may modify to use Latitude and Longitude.
        # u_contexts = user_pd[["RS", "AS", "Latitude", "Longitude"]].to_numpy()
        u_contexts = user_pd[["RS", "AS"]].to_numpy()
        graph.nodes['user'].data['contexts'] = th.tensor(u_contexts, dtype=th.long)
        
        service_contexs_path = os.path.join(self.raw_dir, "wslist.txt")
        service_pd = pd.read_csv(service_contexs_path, header=None, sep="\t", skiprows=2, names=["sid", "WSDL", "Provider", "IPAddress", "RS", "ipno", "AS", "Latitude",
                                        "Longitude"], encoding='latin-1')
        service_pd = service_pd.fillna(0)
        service_pd["RS"] = pd.Categorical(service_pd["RS"]).codes
        service_pd["AS"] = pd.Categorical(service_pd["AS"]).codes
        service_pd["Provider"] = pd.Categorical(service_pd["Provider"]).codes
        s_contexts = service_pd[["RS", "AS", "Provider"]].to_numpy()
        graph.nodes['service'].data['contexts'] = th.tensor(s_contexts, dtype=th.long)
        
        # NOTE: here you can use other `similarity` function
        graph.apply_edges(lambda edges: {'e': (edges.src['contexts'] == edges.dst['contexts']).float()}, etype='impacts_uu')
        graph.apply_edges(lambda edges: {'e': (edges.src['contexts'] == edges.dst['contexts']).float()}, etype='impacts_ss')
        
        self.graph = graph

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph
    
    def __len__(self):
        return 1
    
    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, f"wsdream-static-{self.qos}-{self.density}.bin")
        info_path = os.path.join(self.save_path, f"wsdream-static-{self.qos}-{self.density}.pkl")
        
        
        
        save_graphs(str(graph_path), self.graph)
        save_info(str(info_path), {
            'num_users': self._num_users,
            'num_services': self._num_services,
            'density': self.density,
            'qos': self.qos,
            'num_records': self._num_qos_records,
            'num_train_records': self._num_train_qos_record,
            'num_valid_records': self._num_valid_qos_record,
            'num_test_records': self._num_test_qos_record
        })
        
    
    def load(self):
        graph_path = os.path.join(self.save_path, f"wsdream-static-{self.qos}-{self.density}.bin")
        info_path = os.path.join(self.save_path, f"wsdream-static-{self.qos}-{self.density}.pkl")
        
        graphs, _ = load_graphs(str(graph_path))
        info = load_info(str(info_path))
        if self.verbose:
            print(info)
        self.graph = graphs[0]
    
    def has_cache(self):
        graph_path = os.path.join(self.save_path, f"wsdream-static-{self.qos}-{self.density}.bin")
        return os.path.exists(graph_path)    
           