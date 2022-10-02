# -*- coding: utf-8 -*-
# @File  : binary_loss.py
# @Author: 汪畅
# @Time  : 2022/9/11  15:18
import torch
import torch.nn as nn
import torch.nn.functional as F


class My_CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(My_CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):  # [bs,num_class]  CE=q*-log(p), q*log(1-p),p=softmax(logits)
        target = target.reshape(logits.shape[0], 1)
        log_pro = -1.0 * F.log_softmax(logits, dim=1)
        one_hot = torch.zeros(logits.shape[0], logits.shape[1]).cuda()
        one_hot = one_hot.scatter_(1, target, 1)
        loss = torch.mul(log_pro, one_hot).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Self_cross_entropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(Self_cross_entropy, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, ignore_index=None):  # [bs,num_class]  CE=q*-log(p), q*log(1-p),p=softmax(logits)
        input = input.contiguous().view(-1, input.shape[-1])
        log_prb = F.log_softmax(input, dim=1)

        one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)  # 将target转换成one-hot编码
        loss = -(one_hot * log_prb).sum(dim=1)  # n,得到每个样本的loss

        if ignore_index:  # 忽略[PAD]的label
            non_pad_mask = target.ne(0)
            loss = loss.masked_select(non_pad_mask)

        not_nan_mask = ~torch.isnan(loss)  # 找到loss为非nan的样本
        if self.reduction == 'mean':
            loss = loss.masked_select(not_nan_mask).mean()
        elif self.reduction == 'sum':
            loss = loss.masked_select(not_nan_mask).sum()
        return loss


