import torch.nn as nn
import torch.nn.functional as F
import torch

import os

import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-4

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float()
            * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        losses = F.relu((distance_positive - distance_negative) + self.margin)
        return losses.mean() if size_average else losses.sum()


No_CLASSES = len(os.listdir("data"))
No_EMBDIM = 32
class ArcLoss(nn.Module):
    def __init__(self,margin,fs):
        super(ArcLoss, self).__init__()
        self.angular_margin = margin
        self.feature_scale = fs
        
        self.class_map = torch.nn.Parameter(torch.Tensor(No_CLASSES,No_EMBDIM)) #edit
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv,stdv)

    def forward(self, embs, labels):
        bs, labels = len(embs), labels-1

        class_map = torch.nn.functional.normalize(self.class_map,dim=1)
        cos_similarity = embs.mm(class_map.T).clamp(min=1e-10, max=1-1e-10)

        pick = torch.zeros([bs, No_CLASSES]).byte().to(device)#edit
        pick[torch.arange(bs), labels] = 1
        original_target_logit = cos_similarity[pick]
        theta = torch.acos(original_target_logit)
        marginal_target_logit = torch.cos(theta+self.angular_margin)

        class_pred = self.feature_scale * (cos_similarity+(marginal_target_logit-original_target_logit).unsqueeze(1))
        loss = torch.nn.CrossEntropyLoss()(class_pred,labels)
        return loss 

class CircleLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# ############## ___Below Code subject to deletion soon___ ###############
# ##################### SAMPLE CIRCLELOSS 1 #####################
# class CircleLoss(nn.Module):
#     def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
#         super(CircleLoss, self).__init__()
#         self.scale = scale
#         self.margin = margin
#         self.similarity = similarity

#     def forward(self, feats, labels):
#         assert feats.size(0) == labels.size(0), \
#             f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

#         m = labels.size(0)
#         mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
#         pos_mask = mask.triu(diagonal=1)
#         neg_mask = (mask - 1).abs_().triu(diagonal=1)
#         if self.similarity == 'dot':
#             sim_mat = torch.matmul(feats, torch.t(feats))
#         elif self.similarity == 'cos':
#             feats = F.normalize(feats)
#             sim_mat = feats.mm(feats.t())
#         else:
#             raise ValueError('This similarity is not implemented.')

#         pos_pair_ = sim_mat[pos_mask == 1]
#         neg_pair_ = sim_mat[neg_mask == 1]

#         alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
#         alpha_n = torch.relu(neg_pair_ + self.margin)
#         margin_p = 1 - self.margin
#         margin_n = self.margin
#         loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
#         loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
#         loss = torch.log(1 + loss_p * loss_n)
#         return loss


# if __name__ == '__main__':
#     batch_size = 10
#     feats = torch.rand(batch_size, 1028)
#     labels = torch.randint(high=10, dtype=torch.long, size=(batch_size,))
#     circleloss = CircleLoss(similarity='cos')
#     print(circleloss(feats, labels))

################################################################################################

#######################  SAMPLE CIRCLELOSS 2 ##########################################
# def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
#     similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
#     label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

#     positive_matrix = label_matrix.triu(diagonal=1)
#     negative_matrix = label_matrix.logical_not().triu(diagonal=1)

#     similarity_matrix = similarity_matrix.view(-1)
#     positive_matrix = positive_matrix.view(-1)
#     negative_matrix = negative_matrix.view(-1)
#     return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


# class CircleLoss(nn.Module):
#     def __init__(self, m: float, gamma: float) -> None:
#         super(CircleLoss, self).__init__()
#         self.m = m
#         self.gamma = gamma
#         self.soft_plus = nn.Softplus()

#     def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
#         ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
#         an = torch.clamp_min(sn.detach() + self.m, min=0.)

#         delta_p = 1 - self.m
#         delta_n = self.m

#         logit_p = - ap * (sp - delta_p) * self.gamma
#         logit_n = an * (sn - delta_n) * self.gamma

#         loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

#         return loss


# if __name__ == "__main__":
#     feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
#     lbl = torch.randint(high=10, size=(256,))

#     inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

#     criterion = CircleLoss(m=0.25, gamma=256)
#     circle_loss = criterion(inp_sp, inp_sn)

#     print(circle_loss)

################################################################################################


###################### ARCLOSS SAMPLE 1 #####################################

### This implementation follows the pseudocode provided in the original paper.
# class Criterion(torch.nn.Module):
#     def __init__(self, opt):
#         super(Criterion, self).__init__()
#         self.par = opt

#         ####
#         self.angular_margin = opt.loss_arcface_angular_margin
#         self.feature_scale  = opt.loss_arcface_feature_scale

#         self.class_map = torch.nn.Parameter(torch.Tensor(opt.n_classes, opt.embed_dim))
#         stdv = 1. / np.sqrt(self.class_map.size(1))
#         self.class_map.data.uniform_(-stdv, stdv)

#         self.name  = 'arcface'

#         self.lr    = opt.loss_arcface_lr

#         ####
#         self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
#         self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
#         self.REQUIRES_OPTIM      = REQUIRES_OPTIM




#     def forward(self, batch, labels, **kwargs):
#         bs, labels = len(batch), labels.to(self.par.device)

#         class_map      = torch.nn.functional.normalize(self.class_map, dim=1)
#         #Note that the similarity becomes the cosine for normalized embeddings. Denoted as 'fc7' in the paper pseudocode.
#         cos_similarity = batch.mm(class_map.T).clamp(min=1e-10, max=1-1e-10)

#         pick = torch.zeros(bs, self.par.n_classes).byte().to(self.par.device)
#         pick[torch.arange(bs), labels] = 1

#         original_target_logit  = cos_similarity[pick]

#         theta                 = torch.acos(original_target_logit)
#         marginal_target_logit = torch.cos(theta + self.angular_margin)

#         class_pred = self.feature_scale * (cos_similarity + (marginal_target_logit-original_target_logit).unsqueeze(1))
#         loss       = torch.nn.CrossEntropyLoss()(class_pred, labels)

#         return loss
