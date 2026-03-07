import torch
from torchmetrics import Metric


class Precision(Metric):
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(Precision, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propensity_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        top_k, indices = top_k.to(self.device), indices.to(self.device)
        res = torch.zeros_like(preds).type_as(preds).to(self.device)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))

        score = res * target.to(self.device)

        if self.propensity_score is not None:
            score = score / (self.propensity_score)

        score = score.sum(dim=1).to(self.device)
        # print(score)
        # if idx %100 == 0:
        #     print('topk:{}, target:{}'.format(indices, [torch.nonzero(target[i], as_tuple=True)[0] for i in range(preds.shape[0])]))
        score = score / target.sum(dim=1).to(self.device)
        self.score += score.sum().to(self.device)
        self.total += preds.size(0)

    def compute(self):
        return self.score / self.total


class HIT(Metric):
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(HIT, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = top_k
        self.propensity_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        res = torch.zeros_like(preds).type_as(preds)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))
        score = res * target

        if self.propensity_score is not None:
            score = score / (self.propensity_score)

        score = score.sum(dim=1)
        self.score += score.sum()
        self.total += preds.size(0)

    def compute(self):
        if self.total == 0:
            return 0
        return self.score / self.total


class NormalizedDCG(Metric):
    def __init__(self, top_k, propensity_score=None, dist_sync_on_step=False):
        super(NormalizedDCG, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.propen_score = propensity_score
        self.add_state('score', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.top_k = top_k

    def update(self, preds, target):
        top_k, indices = torch.topk(preds, self.top_k, dim=1)
        top_k, indices = top_k.to(self.device), indices.to(self.device)
        res = torch.zeros_like(preds).type_as(preds).to(self.device)
        res = res.scatter(1, indices, torch.ones_like(top_k).type_as(top_k))
        score = (res * target.to(self.device)).to(self.device)

        log_val = torch.log2(torch.arange(2, self.top_k + 2)).type_as(preds).to(self.device)
        res_log = torch.ones_like(preds).type_as(preds) .to(self.device) # prevent zero
        log_val = log_val.view(1, -1).repeat(preds.size(0), 1)
        res_log = res_log.scatter(1, indices, log_val)

        score = score / res_log
        if self.propen_score is not None:
            score = score / (self.propen_score)

        score = score.sum(dim=1).to(self.device)
        self.score += score.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.score / self.total




