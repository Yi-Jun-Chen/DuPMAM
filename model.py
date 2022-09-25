import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def atten_get_graph_feature(x, k=20, idx=None, firstlayer=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k) 
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x_ver = x.unsqueeze(-1)
    x_glo1 = torch.sum(x, dim=1)
 
    x_glo2 = torch.mean(x, dim=1)
    x_glo = torch.cat((x_glo1,x_glo2), dim=-1)

    x = x.transpose(2, 1).contiguous()
    
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    DEd1 = feature-x
    DEd2 = DEd1.mul(DEd1)
    DEd3 = torch.sum(DEd2,dim=-1,keepdim=True)
    if firstlayer == True:
        feature = torch.cat(( DEd3, feature-x, x, feature ), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return x_ver, feature,x_glo

class LAB(nn.Module):
    def __init__(self, size1, size2, size3):
        super(LAB,self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3

        self.bn1 = nn.BatchNorm2d(self.size3)
        self.bn2 = nn.BatchNorm2d(self.size3)
        self.bn3 = nn.BatchNorm2d(1)

        self.keyconv = nn.Sequential(nn.Conv2d(self.size1, self.size3, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.valconv = nn.Sequential(nn.Conv2d(self.size2, self.size3, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.scoconv = nn.Sequential(nn.Conv2d(self.size3, 1, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self,x_query,x_key):
        querys, values = self.keyconv(x_query), self.valconv(x_key)
        features = querys + values
        scores = self.scoconv(features).squeeze(1)
        scores = F.softmax(scores,dim=2)
        scores = scores.unsqueeze(1).repeat(1, self.size3, 1, 1)
        feature = values.mul(scores)
        feature = torch.sum(feature,dim=-1)
        return feature

class GAB(nn.Module):
    def __init__(self):
        super(GAB,self).__init__()
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(1)

        self.Linear1 = nn.Linear(512, 2048, bias = False)
        self.Linear2 = nn.Linear(2048, 512, bias = False)
        self.Linear3 = nn.Linear(512, 64, bias = False)
        self.Linear4 = nn.Linear(64, 1, bias = False)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.dp3 = nn.Dropout(p=0.5)
        self.dp4 = nn.Dropout(p=0.5)

    def forward(self,x_query,x_key):
        values =  F.leaky_relu(self.bn1(self.Linear1(x_key).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        
        values = self.dp1(values)
        querys = x_query.unsqueeze(1)
        features = values + querys
        scores = F.leaky_relu(self.bn2(self.Linear2(features).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        scores = self.dp2(scores)
        scores = F.leaky_relu(self.bn3(self.Linear3(scores).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        scores = self.dp3(scores)
        scores = F.leaky_relu(self.bn4(self.Linear4(scores).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        scores = self.dp4(scores)
        scores = scores.squeeze(-1)
        scores = F.softmax(scores,dim=1)
        scores = scores.unsqueeze(-1).repeat(1, 1, 2048)
        feature = values.mul(scores)
        feature = torch.sum(feature,dim=1)
        return feature


class PointFormer(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointFormer, self).__init__()
        self.args = args 
        self.graph_attention1 = LAB(3, 10, 64)
        self.graph_attention2 = LAB(64, 64*2, 64)
        self.graph_attention3 = LAB(64, 64*2, 128)
        self.graph_attention4 = LAB(128, 128*2, 256)
        self.GlobalAtten = GAB()
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, k=20):
        self.k = k
        batch_size = x.size(0)
        x_ver1, x_fea1,x_glo = atten_get_graph_feature(x,  k=self.k, firstlayer=True)
        x1 = self.graph_attention1(x_ver1, x_fea1)

        x_ver2, x_fea2, _= atten_get_graph_feature(x1, k=self.k)
        x2 = self.graph_attention2(x_ver2, x_fea2)

        x_ver3, x_fea3, _= atten_get_graph_feature(x2, k=self.k)
        x3 = self.graph_attention3(x_ver3, x_fea3)

        x_ver4, x_fea4, _= atten_get_graph_feature(x3, k=self.k)
        x4 = self.graph_attention4(x_ver4, x_fea4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.transpose(2,1).contiguous()
        x = self.GlobalAtten(x_glo, x)
     
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x 

