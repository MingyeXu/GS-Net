import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from OP import pointnet2_utils


def knn(x, k):
    '''
    get k nearest neighbors' indices for a single point cloud feature
    :param x:  x is point cloud feature, shape: [B, F, N]
    :param k:  k is the number of neighbors
    :return: KNN graph, shape: [B, N, k]
    '''      
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def eigen_function(X):
    '''
    get eigen and eigenVector for a single point cloud neighbor feature
    :param X:  X is a Tensor, shape: [B, N, K, F]
    :return eigen: shape: [B, N, F]
    '''
    B, N, K, F = X.shape
    # X_tranpose [N,F,K] 
    X_tranpose = X.permute(0, 1, 3, 2)
    # high_dim_matrix [N, F, F]
    high_dim_matrix = torch.matmul(X_tranpose, X)

    high_dim_matrix = high_dim_matrix.cpu().detach().numpy()
    eigen, eigen_vec = np.linalg.eig(high_dim_matrix)
    eigen_vec = torch.Tensor(eigen_vec).cuda()
    eigen = torch.Tensor(eigen).cuda()

    return eigen, eigen_vec


def eigen_Graph(x, k=20):
    '''
    get eigen Graph for point cloud
    :param X: x is a Tensor, shape: [B, F, N]
    :param k: the number of neighbors
    :return feature: shape: [B, F, N]
    :retrun idx_EuclideanSpace: k nearest neighbors of Euclidean Space, shape[B, N, k]
    :retrun idx_EigenSpace: k nearest neighbors of Eigenvalue Space, shape[B, N, k]
    '''    
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    device = torch.device('cuda')
    x = x.view(batch_size, -1, num_points)

    # idx [batch_size, num_points, k]
    idx_EuclideanSpace = knn(x, k=k)   
    idx_EuclideanSpace = idx_EuclideanSpace + torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_EuclideanSpace = idx_EuclideanSpace.view(-1)
 

    x = x.transpose(2, 1).contiguous()# (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx_EuclideanSpace, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    
    eigen,eigen_vec = eigen_function(feature-x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1))
    eigen_vec = eigen_vec.reshape([batch_size, num_points, -1])

    feature = torch.cat(( x, eigen, eigen_vec), dim=2)

    idx_EigenSpace = knn(eigen.permute(0,2,1), k=k)   # (batch_size, num_points, k)
    idx_EigenSpace = idx_EigenSpace + torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_EigenSpace = idx_EigenSpace.view(-1)

    return feature.permute(0,2,1), idx_EuclideanSpace, idx_EigenSpace


def first_GroupLayer(x, idx_EU, idx_EI, k=20):
    '''
    group Features for point cloud (Frist Layer)
    :param x: x is a Tensor, shape: [B, F, N]
    :param idx_EU: k nearest neighbors of Euclidean Space, shape[B, N, k]
    :param idx_EI: k nearest neighbors of Eigenvalue Space, shape[B, N, k]
    :param k: the number of neighbors
    :return output feature: shape: [B, F, N, k]
    '''        
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    org_xyz = x[:,0:3,:] # coordinate
    org_feats = x[:,3:6,:] #eigenValue

    org_xyz = org_xyz.transpose(2, 1).contiguous()
    xyz = org_xyz.view(batch_size*num_points, -1)[idx_EU, :]
    xyz = xyz.view(batch_size, num_points, k, 3)
    org_xyz = org_xyz.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1) 

    grouped_xyz = torch.cat((xyz - org_xyz, xyz), dim = 3)

    org_feats = org_feats.transpose(2, 1).contiguous()
    feats = org_feats.view(batch_size*num_points, -1)[idx_EI, :]
    feats = feats.view(batch_size, num_points, k, 3)
    org_feats = org_feats.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1) 

    # feat2 = feats -org_feats
    grouped_feats = torch.cat((feats - org_feats, feats), dim = 3)

    output = torch.cat((grouped_xyz, grouped_feats), dim = 3).permute(0, 3, 1, 2)
    return output



def GroupLayer(x, k=20, idx=None):
    '''
    group Features for point cloud
    :param x: x is a Tensor, shape: [B, F, N]
    :param idx: k nearest neighbors , shape[B, N, k]
    :param k: the number of neighbors
    :return output feature: shape: [B, F, N, k]
    '''     
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, feature), dim=3).permute(0, 3, 1, 2)
  
    return feature

def get_graph_distance(x, k=20, idx=None):
    '''
    get Graph Distance for point cloud
    :param x: x is a Tensor, shape: [B, F, N]
    :param idx: k nearest neighbors , shape[B, N, k]
    :param k: the number of neighbors
    :return output feature: shape: [B, F, N, k]
    '''     
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')
    _, num_dims, _ = x.size()


    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    knn_points = x.view(batch_size*num_points, -1)[idx, :]#[B,N,K,3]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    distance = knn_points-x #[B,N,K,3]
    distance = torch.sqrt(torch.sum(distance * distance, dim = -1))# [B,N,K]

    return distance.reshape((batch_size,1,num_points,k))




class GSNET(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GSNET, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(13, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*4, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*4, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)



    def GSCM(self, points, feats, k, conv, isFirstLayer=False):
        '''
        Geometry Similarity Connection Module
        :param points:  points' coordinates, shape: [B, N, 3]
        :param feats: points' feature, shape: [B, N, F]
        :param k: the number of neighbors
        :param conv: convolution layers
        :return output feature: shape: [B, F, N]
        '''    
        if isFirstLayer:
            x, idx_EU, idx_EI = eigen_Graph(points.permute(0,2,1).contiguous(), k=k)
            x = first_GroupLayer(x, idx_EU, idx_EI,k=k)
            distance = get_graph_distance(points.permute(0,2,1).contiguous(),k=k, idx = idx_EU)
            x = torch.cat((x, distance),dim = 1)           
        else:
            _, idx_EU, idx_EI = eigen_Graph(points.permute(0,2,1).contiguous(), k=k)
            x_knn_EU = GroupLayer(feats, k=k, idx=idx_EU)
            x_knn_EI = GroupLayer(feats, k=k, idx=idx_EI)
            x = torch.cat((x_knn_EU,x_knn_EI),dim = 1)
        x = conv(x)            
        x = x.max(dim=-1, keepdim=False)[0]
        return x


    def forward(self, x):
        batch_size = x.size(0)
        num_points_1 = x.size(2)
        num_points_2 = int(num_points_1/2)
        num_points_3 = int(num_points_1/4)

        ########################BLOCK1##############################
        N1_points = x.permute(0,2,1).contiguous()
        x1 = self.GSCM( N1_points, None, self.k, self.conv1, isFirstLayer=True)

        ########################BLOCK2##############################
        fps_id_2 = pointnet2_utils.furthest_point_sample(N1_points, num_points_2)
        N2_points = (
            pointnet2_utils.gather_operation(
                N1_points.transpose(1, 2).contiguous(), fps_id_2
            ).transpose(1, 2).contiguous())
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1, fps_id_2)
            )
        x2 = self.GSCM( N2_points, x1_downSample, self.k, self.conv2)

        ########################BLOCK3##############################
        fps_id_3 = pointnet2_utils.furthest_point_sample(N2_points, num_points_3)
        N3_points = (
            pointnet2_utils.gather_operation(
                N2_points.transpose(1, 2).contiguous(), fps_id_3
            ).transpose(1, 2).contiguous())
        x2_downSample = (
            pointnet2_utils.gather_operation(
                x2, fps_id_3)
            )
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1_downSample, fps_id_3)
            )   
        x3 = self.GSCM( N3_points, x2_downSample, self.k, self.conv3)


        x = torch.cat((x1_downSample, x2_downSample, x3), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
