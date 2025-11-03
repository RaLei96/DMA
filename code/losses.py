import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.nn.parameter import Parameter
from pytorch_metric_learning import miners, losses
import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.nn import init
import numpy as np

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output



class dma(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, nb_c, tau, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        self.C  = nb_c
        self.nb_classes = nb_classes
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes*self.C, sz_embed).cuda())
        # self.fc = Parameter(torch.Tensor(sz_embed, nb_classes*8))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

        self.gamma = 1./0.1
        self.tau = tau

        self.weight = torch.zeros(self.nb_classes*self.C, self.nb_classes*self.C, dtype=torch.bool).cuda()
        
    def forward(self, X, T):
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity

        simStruc = cos.reshape(-1, self.nb_classes, self.C)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)

        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (simClass - self.mrg))
        neg_exp = torch.exp(self.alpha * (simClass + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)
        num_valid_proxies = len(with_pos_proxies)
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term   


        if self.tau > 0 and self.C > 1:
            simCenter = F.linear(l2_norm(P), l2_norm(P)) 
            # import ipdb; ipdb.set_trace()
            simStruc_p = simCenter.reshape(-1, self.nb_classes, self.C)
            # prob_p = F.softmax(simStruc_p*self.gamma, dim=2)
            prob_p = 1
            # simClass_p = torch.sum(prob_p*simStruc_p, dim=2)
            simClass_p = torch.mean(prob_p*simStruc_p, dim=2)
            
            kmeans = KMeans(n_clusters=self.nb_classes).fit(P.cpu().detach().numpy())
            P_one_hot_p = binarize(T = torch.tensor(kmeans.labels_), nb_classes = self.nb_classes)
            
            N_one_hot_p = 1 - P_one_hot_p
            pos_exp_p = torch.exp(-self.alpha * (simClass_p - self.mrg))
            neg_exp_p = torch.exp(self.alpha * (simClass_p + self.mrg))

            with_pos_proxies_p = torch.nonzero(P_one_hot_p.sum(dim = 0) != 0).squeeze(dim = 1)
            num_valid_proxies_p = len(with_pos_proxies_p)

            P_sim_sum_p = torch.where(P_one_hot_p == 1, pos_exp_p, torch.zeros_like(pos_exp_p)).sum(dim=0) 
            N_sim_sum_p = torch.where(N_one_hot_p == 1, neg_exp_p, torch.zeros_like(neg_exp_p)).sum(dim=0)
            pos_term_p = torch.log(1 + P_sim_sum_p).sum() / num_valid_proxies_p
            neg_term_p = torch.log(1 + N_sim_sum_p).sum() / self.nb_classes
            reg = pos_term_p + neg_term_p   

            return loss+self.tau*reg
        else:
            return loss


