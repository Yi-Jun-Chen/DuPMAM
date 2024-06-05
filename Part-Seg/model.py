#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def atten_get_graph_feature(x, k=20, idx=None, firstlayer=False):
    batch_size = x.size(0)#32
    num_points = x.size(2)#1024
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()#num_dims=3
    x_ver = x.unsqueeze(-1)
    x_glo1 = torch.sum(x, dim=1)
    #x_glo2 = x_glo1.mul(x_glo1)
    x_glo2 = torch.mean(x, dim=1)
    x_glo = torch.cat((x_glo1,x_glo2), dim=-1)
    #x_glo = x_glo1 + x_glo2
    x = x.transpose(2, 1).contiguous()
    
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    DEd1 = feature-x
    DEd6 = torch.sum(DEd1,dim=-1,keepdim=True)
    DEd2 = DEd1.mul(DEd1)
    DEd5 = torch.sum(DEd2,dim=-1,keepdim=True)
    DEd3 = DEd2.mul(DEd1)
    DEd4 = torch.sum(DEd3,dim=-1,keepdim=True)
    if firstlayer == True:
        feature = torch.cat(( DEd5, feature-x, x, feature ), dim=3).permute(0, 3, 1, 2).contiguous()
    else:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return x_ver, feature,x_glo



class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class LAB(nn.Module):
    def __init__(self, size1, size2, size3):
        super(LAB,self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3

        self.bn1 = nn.BatchNorm2d(self.size3)
        self.bn2 = nn.BatchNorm2d(self.size3)
        self.bn3 = nn.BatchNorm2d(1)

        self.conkey = nn.Conv2d(self.size1, self.size3, kernel_size=1, bias=False)
        self.conval = nn.Conv2d(self.size2, self.size3, kernel_size=1, bias=False)
        self.consco = nn.Conv2d(self.size3, 1, kernel_size=1, bias=False)
        
        #init(self.conkey.weight,a=0.2)
        #init(self.conval.weight,a=0.2)
        #init(self.consco.weight,a=0.2)

        self.keyconv = nn.Sequential(self.conkey,
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.valconv = nn.Sequential(self.conval,
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.scoconv = nn.Sequential(self.consco,
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self,x_query,x_key):
        querys, values = self.keyconv(x_query), self.valconv(x_key)
        #querys = querys.repeat(1, 1, 1, 20)
        features = querys + values
        #features = torch.cat((querys, values), dim=1)
        scores = self.scoconv(features).squeeze(1)
        scores = F.softmax(scores,dim=2)
        scores = scores.unsqueeze(1).repeat(1, self.size3, 1, 1)
        feature = values.mul(scores)
        feature = torch.sum(feature,dim=-1)
        return feature

class GAB(nn.Module):
    def __init__(self, init = nn.init.kaiming_normal):
        super(GAB,self).__init__()
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1)

        self.Linear1 = nn.Linear(512, 2048, bias = False)
        #init(self.Linear1.weight,a=0.2)
        self.Linear2 = nn.Linear(2048, 512, bias = False)
        #init(self.Linear2.weight,a=0.2)
        self.Linear3 = nn.Linear(512, 128, bias = False)
        #init(self.Linear3.weight,a=0.2)
        self.Linear4 = nn.Linear(128, 1, bias = False)
        #init(self.Linear4.weight,a=0.2)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
        self.dp3 = nn.Dropout(p=0.5)
        self.dp4 = nn.Dropout(p=0.5)

    def forward(self,x_query,x_key):
        values =  F.leaky_relu(self.bn1(self.Linear1(x_key).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        
        values = self.dp1(values)
        if x_query.size()[1] == 2048:
            x_query = x_query
        else :
            x_query = x_query.repeat(1,20)
            x_query = x_query[:,:2048]
        querys = x_query.unsqueeze(1)
        #querys = querys.repeat(1, 1024, 1)
        #features = querys.mul(values)
        features = values + querys
        scores = F.leaky_relu(self.bn2(self.Linear2(features).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)#维度为（32,1024,20),每一行为对应顶点20个邻居节点的attention分数
        
        scores = self.dp2(scores)
        
        scores = F.leaky_relu(self.bn3(self.Linear3(scores).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        
        scores = self.dp3(scores)
        
        scores = F.leaky_relu(self.bn4(self.Linear4(scores).transpose(2, 1).contiguous()).transpose(2, 1).contiguous(),negative_slope=0.2)
        
        scores = self.dp4(scores)
        
        scores = scores.squeeze(-1)
        scores = F.softmax(scores,dim=1)
        
        scores1 = scores.sort()[1]

        scores = scores.unsqueeze(-1).repeat(1, 1, 2048)
        feature = values.mul(scores)

        #feature1 = torch.zeros([scores.size()[0],920,2048]).to(torch.device("cuda"))
        #for i in range(scores.size()[0]):
            #feature1[i,:,:] = torch.index_select(feature[i,:,:], 0, scores1[i,104:])
        #feature = feature.max(dim=1, keepdim=False)[0]
        feature = torch.sum(feature,dim=1)
        #feature = x + feature
        return feature

class DuPPAM(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DuPPAM, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.transform_net = Transform_Net(args)
        
        self.LAB1 = LAB(3, 10, 64)
        self.LAB2 = LAB(64, 64*2, 64)
        self.LAB3 = LAB(64, 64*2, 128)
        self.LAB4 = LAB(128, 128*2, 256)

        self.GAB = GAB()
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(2624, 512, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.LAB5 = LAB(128, 128*2, 64)
        self.LAB6 = LAB(64, 64*2, 64)
        self.LAB7 = LAB(64, 64*2, seg_num_all)
        

    def forward(self, x, l, k):
        self.k = k
        batch_size = x.size(0)
        num_points = x.size(2)

        x_ver1, x_fea1, _= atten_get_graph_feature(x, k=self.k)    
        t = self.transform_net(x_fea1)              
        x = x.transpose(2, 1)                   
        x = torch.bmm(x, t)                     
        x = x.transpose(2, 1)                  
        x_ver2, x_fea2, x_glo = atten_get_graph_feature(x, k=self.k, firstlayer=True)      
        x2 = self.LAB1(x_ver2, x_fea2)
        x_ver3, x_fea3, _ = atten_get_graph_feature(x2, k=self.k)                      
        x3 = self.LAB2(x_ver3, x_fea3)                       


        x_ver4, x_fea4, _ = atten_get_graph_feature(x3, k=self.k)    
        x4 = self.LAB3(x_ver4, x_fea4)
        x_ver5, x_fea5, _ = atten_get_graph_feature(x4, k=self.k)     
        x5 = self.LAB4(x_ver5, x_fea5)
        
     


        x = torch.cat((x2, x3, x4, x5), dim=1)      

                        
        x = x.transpose(2,1).contiguous()
        x = self.GAB(x_glo, x)
        x = x.unsqueeze(-1)      

        l = l.view(batch_size, -1, 1)          
        l = self.conv7(l)                   

        x = torch.cat((x, l), dim=1)            
        x = x.repeat(1, 1, num_points)       

        x = torch.cat((x, x2, x3, x4, x5), dim=1)  

        x = self.conv8(x)                     
        x = self.dp1(x)
        x = self.conv9(x)                       
        x = self.dp2(x)
        x = self.conv10(x)  
        
        x_ver6, x_fea6, _ = atten_get_graph_feature(x, k=self.k)                       
        x = self.LAB5(x_ver6, x_fea6)                      


        x_ver7, x_fea7, _ = atten_get_graph_feature(x, k=self.k)     
        x = self.LAB6(x_ver7, x_fea7)
        x_ver8, x_fea8, _ = atten_get_graph_feature(x, k=self.k)     
        x = self.LAB7(x_ver8, x_fea8)                   
        
        return x
