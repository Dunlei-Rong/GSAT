import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import swanlab as wandb
from torch.nn.utils.rnn import pad_sequence

from src.utils.metrics import Precision, NormalizedDCG
from src.models.modules.Attention import SelfAttention, MotifAttention, MultiHeadAttention
from src.utils.utils import cluster_pic_show, random_walk, RequirementNet, total_cluster_based_loss, kmeans_clustering
from src.models.modules.random_walk import random_walk_subgraph_pyg
# from src.models.modules.augs import Augmentor
# from src.models.modules.GNN import GraphNet


class GSAT(pl.LightningModule):
    @property
    def device(self):
        return self._device

    def __init__(self, data_dir, lr, head_num, weight_decay, embedding_model='openai', fine_tuning=False,
                 topk=None, dataset='Mashup', RSGAP=False):
        super().__init__()

        self.device = torch.device('cuda:0')
        self.dataset = dataset
        self.fine_tuning = fine_tuning
        self.lr = lr
        self.weight_decay = weight_decay
        self.topk = topk
        self.mashup_topk = 100

        if dataset == 'Youshu':
            edge_path = '/standard/train_edges.emb'
        else:
            edge_path = '/train_edges.emb'
        if dataset == 'Youshu':
            self.input_channel = 96
            enhenced_edges = torch.load(data_dir + edge_path).numpy().T
            with open(data_dir + '/raw/train_item', 'rb') as f:
                user_item = pickle.load(f)
            self.mashup_embeds = nn.Embedding(user_item.shape[0], self.input_channel)
            self.api_embeds = nn.Embedding(user_item.shape[1], self.input_channel)
            subgraph_path = 'data/Youshu/preprocessed_data/sub_graph_list.pt'
            self.hidden_channel = int(self.input_channel / 2)
            self.num_api = self.api_embeds.num_embeddings
            self.num_mashup = self.mashup_embeds.num_embeddings
        else:
            if embedding_model == 'openai':
                pre_data_dir = data_dir + '/preprocessed_data/openai_emb/'
                self.api_embeds = torch.stack(torch.load(pre_data_dir + 'api_openai_text_embedding.pt'), dim=0)
                self.mashup_embeds = torch.stack(torch.load(pre_data_dir + 'mashup_openai_text_embedding.pt'), dim=0)
            elif embedding_model == 'bert' and fine_tuning is True:
                pre_data_dir = data_dir + '/preprocessed_data/description/'
                self.mashup_embeds = torch.load(pre_data_dir + 'pre_bert_mashup_embedding.pt')
                self.api_embeds = torch.load(pre_data_dir + 'pre_bert_api_embedding.pt')
            elif embedding_model == 'bert' and fine_tuning is False:
                pre_data_dir = data_dir + '/preprocessed_data/bert/'
                self.api_embeds = torch.load(pre_data_dir + 'bert_apis_embeddings.emb')
                self.mashup_embeds = torch.load(pre_data_dir + 'bert_mashup_embeddings.emb')[1]
            else:
                pre_data_dir = data_dir + '/preprocessed_data/word2vec/'
                self.api_embeds = torch.stack(torch.load(pre_data_dir + 'api_word2vec_text_embedding.pt'), dim=0)
                self.mashup_embeds = torch.stack(torch.load(pre_data_dir + 'mashup_word2vec_text_embedding.pt'), dim=0)
            enhenced_edges = torch.load(data_dir + edge_path).numpy().T

            self.input_channel = self.api_embeds.shape[1]
            self.hidden_channel = int(self.api_embeds.shape[1] / 2)
            self.num_api = self.api_embeds.size(0)
            self.num_mashup = self.mashup_embeds.size(0)

        edge_index = []
        for edge in enhenced_edges.T:
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])
        self.edge_index = torch.tensor(edge_index).T.to(self.device)

        self.api_embeds = self.api_embeds.to(self.device)
        self.mashup_embeds = self.mashup_embeds.to(self.device)

        self.GRU = nn.GRU(self.input_channel, self.input_channel)
        self.mashup_api = {}
        self.api_mashup = {}
        for edge in enhenced_edges.T:
            if int(edge[0]) not in self.mashup_api:
                self.mashup_api[int(edge[0])] = [int(edge[1])]
            else:
                self.mashup_api[int(edge[0])].append(int(edge[1]))
            if int(edge[1]) not in self.api_mashup:
                self.api_mashup[int(edge[1])] = [int(edge[0])]
            else:
                self.api_mashup[int(edge[1])].append(int(edge[0]))
        self.mashup_api.update(self.api_mashup)

        node_sequence_list = []
        for node in range(self.num_mashup + self.num_api):
            seq = random_walk(self.mashup_api, node, steps=20)
            node_sequence_list.append(torch.tensor(seq, dtype=torch.long))
        self.node_sequence_list = pad_sequence(node_sequence_list, batch_first=True).to(self.device)

        self.node_attention_mlp = nn.Linear(self.input_channel, self.input_channel)
        self.motivation = nn.LeakyReLU()
        self.node_attention = SelfAttention(self.input_channel)

        self.mashup_attention_1 = MultiHeadAttention(self.input_channel, self.input_channel)
        self.mashup_attention_2 = MultiHeadAttention(self.input_channel, self.input_channel)

        self.mashups_MLP = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.input_channel)
        )
        self.api_MLP = nn.Sequential(
            nn.Linear(self.input_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.hidden_channel),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_channel, self.input_channel)
        )

        self.P = {}
        self.P_val = {}
        self.DCG = {}
        self.DCG_val = {}

        for k in self.topk:
            self.P[k] = Precision(k)
            self.P_val[k] = Precision(k)
            self.DCG[k] = NormalizedDCG(k)
            self.DCG_val[k] = NormalizedDCG(k)

        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, users):
        embeddings = torch.cat([self.mashup_embeds, self.api_embeds], dim=0)


        seq_nodes = self.node_sequence_list[users]
        seq_emb = embeddings[seq_nodes]
        enhanced_embedding = self.GRU(seq_emb)[0][:, 0, :]

        input1 = self.node_attention_mlp(embeddings)
        input1 = self.motivation(input1)
        input1 = self.node_attention(input1, self.edge_index)

        embeddings = embeddings.clone()
        embeddings.index_add_(0, users, enhanced_embedding)
        embeddings += input1

        mashup_embeddings = embeddings[:self.num_mashup]
        user_emb = mashup_embeddings[users]
        sim = user_emb @ mashup_embeddings.T
        _, idx = torch.topk(sim, k=self.mashup_topk, dim=-1)
        neighbor_emb = mashup_embeddings[idx]
        mashup_embeddings = torch.cat([user_emb.unsqueeze(1), neighbor_emb], dim=1)

        mashup_embeddings = self.mashup_attention_1(mashup_embeddings, mashup_embeddings, mashup_embeddings)[0]
        mashup_embeddings = self.mashup_attention_2(mashup_embeddings, mashup_embeddings, mashup_embeddings)[0]
        mashup_embeddings = mashup_embeddings[:, 0, :].squeeze(1)
        embeddings = embeddings.clone()
        embeddings[users] = mashup_embeddings

        mashup_embeddings = self.mashups_MLP(mashup_embeddings)
        api_embeddings = self.api_MLP(embeddings[self.num_mashup:])

        return mashup_embeddings, api_embeddings

    def on_train_start(self) -> None:
        pass

    def training_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch['users'], batch['pos_items'], batch['neg_items']
        mashup_embeddings, api_embeddings = self.forward(user)
        preference = torch.matmul(mashup_embeddings, api_embeddings.T)
        loss = self.criterion(preference, pos_item)
        self.log("loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user, pos_items, _ = batch['users'], batch['pos_items'], batch['neg_items']
        mashup_embeddings, api_embeddings = self.forward(user)
        out = torch.matmul(mashup_embeddings, api_embeddings.transpose(0, 1))
        preds = torch.sigmoid(out)
        # topk, indices = torch.topk(preds, 10, dim=1)

        if not self.trainer.sanity_checking:
            for k in self.topk:
                self.P_val[k].update(preds, pos_items)
                self.DCG_val[k].update(preds, pos_items)
                self.log("val/P@" + str(k), self.P_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)

                self.log("val/DCG@" + str(k), self.DCG_val[k].compute(), on_step=False, on_epoch=True, prog_bar=True)

                wandb.log({"val/P@" + str(k): self.P_val[k].compute()})

    def test_step(self, batch, batch_idx):
        # todo: test
        user, pos_items, _ = batch['users'], batch['pos_items'], batch['neg_items']
        mashup_embeddings, api_embeddings = self.forward(user)
        out = torch.matmul(mashup_embeddings, api_embeddings.transpose(0, 1))
        preds = torch.sigmoid(out)
        if not self.trainer.sanity_checking:
            for k in self.topk:
                self.P[k].update(preds, pos_items)
                self.DCG[k].update(preds, pos_items)
                self.log("test/P@" + str(k), self.P[k].compute(), on_step=False, on_epoch=True)
                self.log("test/DCG@" + str(k), self.DCG[k].compute(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

    @device.setter
    def device(self, value):
        self._device = value




