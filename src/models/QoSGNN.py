import os

import pytorch_lightning as pl

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.utils import expand_as_pair

from torchmetrics import MeanAbsoluteError, MeanSquaredError

from src.models.modules.TransformerConv import TransformerConv

class QoSGNN(pl.LightningModule):
    r""" QoSGNN for QoS Prediction. This code has some difference with paper. (Caused by DGL and PyG)...
     
    Parameters
    ----------
    num_user : int
        Number of users
        
        
    num_service : int
        Number of services
    
    edge_feat : int or tuple of int
        User/Service-graph edge feat size. If `int` is provided, we set user-graph' and service-graph' 
        edge feat size is same. Or tuple of (user_edge_feat_size, service_edge_feat_size)
        
    node_feat : int or tuple of int
        Learned user/service embeddings size. If `int` is provided, we set user emebed size equal to 
        service embed size. Or tuple of (user_node_feat_size, service_node_feat_size)
    
    lr : float
        learning rate. Default `0.001`
        
    
    weight_decay : float
        Default `1e-4`
    """
    def __init__(self,  num_user, num_service, edge_feat, node_feat, lr, weight_decay) -> None:
        super().__init__()
        self.eval_metric()
        
        self.num_user = num_user
        self.num_service = num_service
        self.u_e_feat_dim, self.s_e_feat_dim = expand_as_pair(edge_feat)
        self.u_n_dim, self.s_n_dim = expand_as_pair(node_feat)
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        self._build_layers()
    
    def eval_metric(self):
        self.MAE = MeanAbsoluteError()
        self.MSE = MeanSquaredError()   # target is RMSE.
        self._val_MAE = MeanAbsoluteError()
        
    def _build_layers(self):
        self.user_embed = nn.Embedding(self.num_user, self.u_n_dim) 
        self.service_embed = nn.Embedding(self.num_service, self.s_n_dim)
        
        # TODO: Do we really need to stack two layers? May be a new experiment?
        self.user_transformerConv = TransformerConv(self.u_n_dim, self.u_n_dim, num_heads=1, edge_feats=self.u_e_feat_dim)
        self.user_transformerConv2 = TransformerConv(self.u_n_dim, self.u_n_dim, num_heads=1, edge_feats=self.u_e_feat_dim)
        
        self.service_transformerConv = TransformerConv(self.s_n_dim, self.s_n_dim, num_heads=1, edge_feats=self.s_e_feat_dim)
        self.service_transformerConv2 = TransformerConv(self.s_n_dim, self.s_n_dim, num_heads=1, edge_feats=self.s_e_feat_dim)

        self.qos_scorer = nn.Sequential(
            nn.Linear(self.u_n_dim + self.s_n_dim, int((self.u_n_dim + self.s_n_dim)/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int((self.u_n_dim + self.s_n_dim)/2), 1),
            nn.ReLU(inplace=True)
        )
        
        self.register_buffer('user_embeds_f', self.user_embed.weight.data.clone().detach())
        self.register_buffer('service_embeds_f', self.service_embed.weight.data.clone().detach())
    
    
        
    def training_step(self, batch, batch_idx):
        ug, sg, qr = batch["ug"], batch["sg"], batch["qr"]
        user_idx, service_idx, qos_value = qr
        u_src_nids, u_dst_nids, u_mfgs = ug
        s_src_nids, s_dst_nids, s_mfgs = sg
        
        u_nid_map = {value.item(): index for index, value in enumerate(u_dst_nids)}
        s_nid_map = {value.item(): index for index, value in enumerate(s_dst_nids)}
        
        # TODO: Better Way?
        user_idx_mapped = th.tensor([u_nid_map[item.item()] for item in user_idx]).type_as(user_idx)
        service_idx_mapped = th.tensor([s_nid_map[item.item()] for item in service_idx]).type_as(service_idx)
        
        # input feats for first layer
        u_feats, s_feats = self.user_embed(u_mfgs[0].srcdata[dgl.NID]), self.service_embed(s_mfgs[0].srcdata[dgl.NID])
        u_e_feats, s_e_feats = u_mfgs[0].edata["e"], s_mfgs[0].edata["e"]
        
        u_feats = self.user_transformerConv(u_mfgs[0], u_feats, u_e_feats)
        u_feats = F.relu(u_feats).view(u_feats.size(0), -1)
        u_e_feats = u_mfgs[1].edata["e"]
        u_dst_feats = self.user_transformerConv2(u_mfgs[1], u_feats, u_e_feats)
        u_dst_feats = u_dst_feats.view(u_dst_feats.size(0), -1)
        u_dst_feats = F.relu(u_dst_feats)
        
        
        s_feats = self.service_transformerConv(s_mfgs[0], s_feats, s_e_feats)
        s_feats = F.relu(s_feats).view(s_feats.size(0), -1)
        s_e_feats = s_mfgs[1].edata["e"]
        s_dst_feats = self.service_transformerConv(s_mfgs[1], s_feats, s_e_feats)
        s_dst_feats = s_dst_feats.view(s_dst_feats.size(0), -1)
        s_dst_feats = F.relu(s_dst_feats)
        
        
        self.user_embeds_f[u_dst_nids] = u_dst_feats.clone().detach()
        self.service_embeds_f[s_dst_nids] = s_dst_feats.clone().detach()
        
        u_inputs, s_inputs = u_dst_feats[user_idx_mapped], s_dst_feats[service_idx_mapped]
        
        predict_feat = th.cat([u_inputs, s_inputs], dim=1)
        
        predicted_qos = self.qos_scorer(predict_feat)
        predicted_qos = predicted_qos.view(-1)
        
        loss = nn.L1Loss()(predicted_qos, qos_value)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        user_idx, service_idx, qos_value = batch
        user_feats, service_feats = self.user_embeds_f[user_idx], self.service_embeds_f[service_idx]
        
        predict_feats = th.cat([user_feats, service_feats], dim=1)
        predicted_qos = self.qos_scorer(predict_feats)
        predicted_qos = predicted_qos.view(-1)
        
        mae = self._val_MAE(predicted_qos, qos_value)
        self.log("val/mae", mae, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        user_idx, service_idx, qos_value = batch
        user_feats, service_feats = self.user_embeds_f[user_idx], self.service_embeds_f[service_idx]
        
        predict_feats = th.cat([user_feats, service_feats], dim=1)
        predicted_qos = self.qos_scorer(predict_feats)
        predicted_qos = predicted_qos.view(-1)
        
        mae = self.MAE(predicted_qos, qos_value)
        mse = self.MSE(predicted_qos, qos_value)
        
        self.log("test/MAE", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/RMSE", mse, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return th.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    