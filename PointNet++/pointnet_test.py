import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch
import torch.nn as nn
from torchsummary import summary
from torch.nn import Conv2d
from einops.layers.torch import Rearrange, Reduce
# from tensorboardX import SummaryWriter
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models
import torchvision.datasets as dset
import numpy as np # linear algebra
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from collections import Counter
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple


def ball_query(xyz, new_xyz, radius, K):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)




def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids

def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    new_xyz = gather_points(xyz, fps(xyz, M))
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''

    :param xyz: shape=(B, M, 3)
    :param points: shape=(B, M, C)
    :param use_xyz:
    :return: new_xyz, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
             group_inds, shape=(B, 1, M); grouped_xyz, shape=(B, 1, M, 3)
    '''
    B, M, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)
    grouped_xyz = xyz.view(B, 1, M, C)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, grouped_inds, grouped_xyz


class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     QuantizeConv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        # print(new_points)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class PointNet_SA_Module_1(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_1, self).__init__()
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class PointNet_SA_Module_MSG(nn.Module):
    def __init__(self, M, radiuses, Ks, in_channels, mlps, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_MSG, self).__init__()
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbones = nn.ModuleList()
        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                         QuantizeConv2d(in_channels, out_channels, 1,
                                                   stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                             nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())
                in_channels = out_channels
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        new_xyz = gather_points(xyz, fps(xyz, self.M))
        new_points_all = []
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]
            grouped_inds = ball_query(xyz, new_xyz, radius, K)
            grouped_xyz = gather_points(xyz, grouped_inds)
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
            if points is not None:
                grouped_points = gather_points(points, grouped_inds)
                if self.use_xyz:
                    new_points = torch.cat(
                        (grouped_xyz.float(), grouped_points.float()),
                        dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]
            new_points = new_points.permute(0, 2, 1).contiguous()
            new_points_all.append(new_points)
        return new_xyz, torch.cat(new_points_all, dim=-1)
    
class Quantize_8bit(Function):
    def __init__(self):
        super(Quantize_8bit, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=2):
        ctx.save_for_backward(input)
        s = (input.max() - input.min()) / (127 - (-128))
        zero_point = 127-input.max()/s
        out = torch.round(input / s + zero_point)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved_tensors
        grad_input = grad_outputs
        # grad_input[input[0].gt(1)] = 0
        # grad_input[input[0].lt(-1)] = 0
        return grad_input, None
    
class Quantize(Function):
    def __init__(self):
        super(Quantize, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=8):
        ctx.save_for_backward(input)
        q_max = 2 ** (nbits - 1) - 1
        q_min = -2 ** (nbits - 1)
        s = (input.max() - input.min()) / (q_max - q_min)
        z = -(input.min() / s) + q_min
        out = torch.round(input / s + z)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved_tensors
        grad_input = grad_outputs
        return grad_input, None
    
class Quantize_int8(Function):
    def __init__(self):
        super(Quantize, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=8):
        ctx.save_for_backward(input)
        q_max = 2 ** (nbits - 1) - 1
        q_min = -2 ** (nbits - 1)
        s = (input.max() - input.min()) / (q_max - q_min)
        z = -(input.min() / s) + q_min
        out = torch.round(input / s + z)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved_tensors
        grad_input = grad_outputs
        return grad_input, None

class TernaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # get output
        ctx_max, ctx_min = torch.max(input), torch.min(input)
        lower_interval = ctx_min + (ctx_max - ctx_min) / 3
        higher_interval = ctx_max - (ctx_max - ctx_min) / 3
        out = torch.where(input < lower_interval, torch.tensor(-1.).to(input.device, input.dtype), input)
        out = torch.where(input > higher_interval, torch.tensor(1.).to(input.device, input.dtype), out)
        out = torch.where((input >= lower_interval) & (input <= higher_interval), torch.tensor(0.).to(input.device, input.dtype), out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input
    
class TriConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(TriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # self.weight = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size),requires_grad=True)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = TernaryQuantize().apply(bw)
        self.weight.data = bw
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    # def forward(self,input):
    #     return(self.conv2d_forward(input,self.weight))

def quantize_2bit(input_tensor, min_val=-1.5, max_val=1.5):
    # 将张量值限制在指定范围内
    clamped_tensor = torch.clamp(input_tensor, min=min_val, max=max_val)
    # 缩放因子和偏移量
    s = 2.0
    z = 0.5
    # 量化到最接近的半整数值
    quantized_values = torch.round(clamped_tensor * s - z) / s
    # 调整值以避免量化为1，-1或0
    mask = (quantized_values == 1.0) | (quantized_values == -1.0) | (quantized_values == 0.0) | (quantized_values == -2.0)
    quantized_values = torch.where(mask, quantized_values + 0.5, quantized_values)
    return quantized_values


# class QuantizeConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(QuantizeConv2d, self).__init__(*args, **kwargs)
    
#     def forward(self, x):
#         # 输入量化为 2 比特
#         # x_q = torch.clamp(x, min=-1.5, max=1.5)
#         # x_q = torch.round(x_q * 2) / 2
#         x_q = Quantize.apply(x)
#         # x_q = x
        
#         # 获取权重
#         weight = self.weight
#         # 权重量化为 2 比特
#         # weight_q = torch.clamp(weight, min=-1.5, max=1.5)
#         # weight_q = torch.round(weight_q * 2) / 2
#         weight_q = Quantize.apply(weight)
        
#         # 卷积操作
#         return nn.functional.conv2d(x_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(QuantizeConv2d, self).__init__(*args, **kwargs)
        # self.qmin, self.qmax = -2 ** (8 - 1), 2 ** (8 - 1) - 1
    
    def forward(self, x):
        # 输入量化为 2 比特
        # x_q = torch.clamp(x, min=-1.5, max=1.5)
        # x_q = torch.round(x_q * 2) / 2
        x_q = Quantize.apply(x)
        # x_q = x
        
        # 获取权重
        weight = self.weight
        # 权重量化为 2 比特
        # weight_q = torch.clamp(weight, min=-1.5, max=1.5)
        # weight_q = torch.round(weight_q * 2) / 2
        weight_q = Quantize.apply(weight)
        # s = (self.weight.max() - self.weight.min()) / (self.qmax - self.qmin)
        # weight_q = s * weight_q
        # z = -(self.weight.min() / s) + self.qmin
        # Z = s * z * torch.ones_like(self.weight)

        output = F.conv2d(x_q, weight_q, self.bias, self.stride,self.padding, self.dilation, self.groups)
        # mapping back to the dequantized range
        # output = s * (output - z)
        
        # 卷积操作
        # return nn.functional.conv2d(x_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # return F.conv2d(x_q, weight_q, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return output

class QuantizeConv2d_1(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(QuantizeConv2d_1, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # self.weight = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size),requires_grad=True)

    def forward(self, input):

        bw = self.weight
        ba = input
        ba = Quantize().apply(ba)
        # ba = ba - ba.mean()
        # bw = bw - bw.mean()
        bw = Quantize().apply(bw)
        self.weight.data = bw
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    # def forward(self,input):
    #     return(self.conv2d_forward(input,self.weight))


class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_cls_ssg, self).__init__()
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels, mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=131, mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        batchsize = xyz.shape[0]
        print(xyz.shape)
        print(points.shape)
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)
        net = new_points.view(batchsize, -1)
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))
        net = self.cls(net)
        return net

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [directory for directory in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/directory)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
                'category': self.classes[category]}
    
train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

path = Path("/home/yue/pointnet2/ModelNet10")
folders = [directory for directory in sorted(os.listdir(path)) if os.path.isdir(path/directory)]
classes = {folder: i for i, folder in enumerate(folders)}
train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()};

print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', inv_classes[train_ds[0]['category']])

train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=valid_ds, batch_size=64)



os.makedirs("./int4/path", exist_ok=True)
# 计算汉明距离的函数
def calculate_hamming_distance(weight1, weight2):
    # 实现汉明距离的计算逻辑
    # 将卷积核的权重矩阵展开为一维向量
    weights1 = weight1.view(-1)
    weights2 = weight2.view(-1)
    # 计算汉明距离
    hamming_distance = (weight1 != weight2).sum().item()
    return hamming_distance
    pass


def euclidean_distance(A, B):
    # 计算欧几里得距离
    distance = torch.norm(A - B)
    # 返回距离
    return distance

def train_model(model, epoch, traindata):
    for epoch in range(epoch):  # 开始迭代
        train_loss_all = []
        train_accur_all = []
        train_loss = 0  # 训练集的损失初始设为0
        train_num = 0.0  #
        train_accuracy = 0.0  # 训练集的准确率初始设为0
        model.train()  # 将模型设置成 训练模式
        train_bar = tqdm(train_loader)  # 用于进度条显示，没啥实际用处
        for step, data in enumerate(train_bar):  # 开始迭代跑， enumerate这个函数不懂可以查查，将训练集分为 data是序号，data是数据
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)  # 将data 分位 img图片，target标签
            optimizer.zero_grad()  # 清空历史梯度
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]
            outputs = model(xyz.to(device), points.to(device))  # 将图片打入网络进行训练,outputs是输出的结果

            loss1 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别-这就是我们通常情况下称为的损失
            outputs = torch.argmax(outputs, 1)  # 会输出10个值，最大的值就是我们预测的结果 求最大值
            loss1.backward()  # 神经网络反向传播
            for name, param in model.named_parameters():
                if 'weight' and 'Conv' in name:
                    if param.requires_grad:
                        if param.grad is not None:
                            grad_mask = param.data != 0  # 创建一个掩码，只有非零的权重的梯度才被保留
                            param.grad *= grad_mask

            optimizer.step()  # 梯度优化 用上面的abam优化
            train_loss += abs(loss1.item()) * inputs.size(0)  # 将所有损失的绝对值加起来
            accuracy = torch.sum(outputs == target.to(device))  # outputs == target的 即使预测正确的，统计预测正确的个数,从而计算准确率
            train_accuracy = train_accuracy + accuracy  # 求训练集的准确率
            train_num += xyz.shape[0]  #

        print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(epoch + 1, train_loss / train_num,  # 输出训练情况
                                                                    train_accuracy / train_num))
        train_loss_all.append(train_loss / train_num)  # 将训练的损失放到一个列表里 方便后续画图
        train_accur_all.append(train_accuracy.double().item() / train_num)  # 训练集的准确率

        # 保存训练完的权重
        torch.save(model, f'./int4/weights_epoch{epoch + 1}_before_pruning_2_4.pth')

        # 统计卷积层中权重为0的个数
        count_zero_weights_in_conv_layers_beforep(model)

        # 测试剪枝后的模型
        test_beforep_model(model, test_loader)

        # 剪枝
        quantize_prune_model(model, threshold_c=0.6, threshold_b=0.1)

        # 保存剪枝后的权重
        pruned_weight_path = f'./int4/weights_epoch_{epoch + 1}_after_pruning_2_4.pth'
        torch.save(model.state_dict(), pruned_weight_path)
        print(f"Pruned model weights saved to {pruned_weight_path}")

        # 统计剪枝后卷积层中权重为0的个数
        count_zero_weights_in_conv_layers_afterp(model)




def count_zero_weights_in_conv_layers_beforep(model):
    zero_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        # print(name)
        # print(param.shape)
        if 'weight' in name and 'Conv' and 'pt_sa1' in name:
            # print(param)
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")
    # 写入信息到文件
    with open('./int4/before_pruning_info_3.txt', 'a') as f:
        f.write(f'Total parameters: {total_count}\n')
        f.write(f'Zero parameters before pruning: {zero_count}\n')
        f.write(f'pruning rate: {zero_percentage}\n')
    # return zero_count
    # total_params = sum(p.numel() for p in model.parameters())
    # zero_params_before_pruning = sum(p.numel() for p in model.parameters() if torch.sum(p.data) == 0)


def count_zero_weights_in_conv_layers_afterp(model):
    zero_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name and 'Conv' and 'pt_sa1' in name:
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")
    # 写入信息到文件
    with open('./int4/after_pruning_info_3.txt', 'a') as f:
        f.write(f'Total parameters: {total_count}\n')
        f.write(f'Zero parameters after pruning: {zero_count}\n')
        f.write(f'pruning rate: {zero_percentage}\n')
    # return zero_count


def test_beforep_model(model, testdata):
    # test biinary
    test_loss_all = []
    test_accur_all = []
    test_loss = 0  # 同上 测试损失
    test_accuracy = 0.0  # 测试准确率
    test_num = 0
    model.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        test_bar = tqdm(testdata)
        for data in test_bar:
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]
            outputs = model(xyz.to(device), points.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * inputs.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += xyz.shape[0]
            accuracy_before_pruning = test_accuracy / test_num

    print("test-Loss：{} , test-accuracy-before-pruning：{}".format(test_loss / test_num, accuracy_before_pruning))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)
    # 写入信息到文件
    with open('./int4/test-accuracy-before-pruning_3.txt', 'a') as f:
        f.write(f'Accuracy on test set before pruning: {accuracy_before_pruning}\n')
        # f.write(f'Accuracy on test set after pruning: {accuracy_after_pruning}\n')


def test_afterp_model(model, testdata):
    # test biinary
    test_loss_all = []
    test_accur_all = []
    test_loss = 0  # 同上 测试损失
    test_accuracy = 0.0  # 测试准确率
    test_num = 0
    model.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        test_bar = tqdm(testdata)
        for data in test_bar:
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]
            outputs = model(xyz.to(device), points.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * inputs.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += xyz.shape[0]
            accuracy_after_pruning = test_accuracy / test_num

    print("test-Loss：{} , test-accuracy-after-pruning：{}".format(test_loss / test_num, accuracy_after_pruning))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)
    # 写入信息到文件
    with open('./int4/test-accuracy-after-pruning_3.txt', 'a') as f:
        # f.write(f'Accuracy on test set before pruning: {accuracy_before_pruning}\n')
        f.write(f'test_afterp_model: {accuracy_after_pruning}\n')

# 保存模型参数为quantize形式
def quantize_prune_model(model, threshold_c, threshold_b):
    for key, value in model.state_dict().items():
        if 'Conv' in key and 'weight' and 'pt_sa1' in key:  # 对卷积层进行pruning
            print(key)
            # quantize_value = torch.clamp(value, min=-1.5, max=1.5)
            # quantize_value = torch.round(quantize_value * 2) / 2
            # quantize_value = quantize_2bit(value)
            weights = value
            close_filters = []
            num_filters = weights.size(0)
            hamming_distances = []

            for i in range(num_filters):
                if torch.sum(weights[i]) != 0:
                    for j in range(i + 1, num_filters):
                        if torch.sum(weights[j]) != 0:
                            hamming_distance_ij = euclidean_distance(weights[i], weights[j])
                            # print(hamming_distance_ij)
                            hamming_distances.append((i, j, hamming_distance_ij))
            distances = [distance.cpu() for _, _, distance in hamming_distances]
            # print(f'Hamming distances: {distances}')
            ean_distance = np.mean(distances)
            variance_distance = np.var(distances)
            # print(f'ean_distance: {ean_distance}, variance_distance: {variance_distance}')
            filters_to_prune = []
            for i, j, distance in hamming_distances:
                if distance < ean_distance - threshold_b * variance_distance:
                    filters_to_prune.append(i)
                    filters_to_prune.append(j)

            # 使用Counter计算集合中每个元素的频率
            filter_counts = Counter(filters_to_prune)
            print(filter_counts)
            # 计算每个元素出现的次数
            filters_to_prune_final = [filter_num for filter_num, count in filter_counts.items()
                                      if count > threshold_c * (num_filters - 1)]
            # print(f'Filters to prune in layer {key}: {filters_to_prune_final}')
            # 进行滤波器剪枝操作
            for filter_num in filters_to_prune_final:
                # 将需要剪枝的滤波器权重设置为0
                # 进行对应filter位置的pruning操作
                # print(f'Pruning filter {filter_num} in layer {key}')
                weights[filter_num] = 0

            # 更新模型的state_dict以保留权重
            model.state_dict()[key].copy_(weights)  # 使用.copy_()确保权重被正确更新


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

model = pointnet2_cls_ssg(in_channels=3, nclasses=10)
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
# model_weight_path = "./resnet34-pre.pth"
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
# print(net.to(device))  # 输出模型结构

# test1 = torch.ones(1, 1, 7, 7)  # 测试一下输出的形状大小 输入一个64,3,120,120的向量
#
# test1 = net(test1.to(device))  # 将向量打入神经网络进行测试
# print(test1.shape)  # 查看输出的结果
# epoch = 200  # 迭代次数即训练次数
# learning = 0.001  # 学习率
# optimizer = torch.optim.Adam(model.parameters(), lr=learning)  # 使用Adam优化器-写论文的话可以具体查一下这个优化器的原理
loss = nn.CrossEntropyLoss()  # 损失计算方式，交叉熵损失函数

# train_model(model, epoch=100, traindata=train_loader)
# m_state_dict = torch.load('/home/yue/cimcam/weights_epoch75_before_pruning.pth')
# model.load_state_dict(m_state_dict)
# semantic_center = torch.load('C:\\Users\\zy\\Desktop\\Dynamic\\Pointnet++\\pointnet2_semantic_center_0.88.pth', map_location='cuda:0')
# 加载整个模型对象
# saved_model = torch.load('/home/yue/cimcam/weights_epoch_75_after_pruning.pth')  # 加载保存的模型对象
# model = saved_model  # 如果直接保存了整个模型对象，可以直接赋值给模型
saved_model = torch.load('/home/yue/cimcam/weights_epoch_75_after_pruning_2_4.pth')  # 加载整个模型对象
model.load_state_dict(torch.load('/home/yue/cimcam/weights_epoch_75_after_pruning_2_4.pth'))
# m_state_dict = saved_model.state_dict()  # 提取 state_dict
# model.load_state_dict(m_state_dict)  # 正确加载
# 确保模型在测试时使用评估模式
model.eval()  # 设置模型为评估模式（evaluation mode）

# 定义测试损失和准确率的统计变量
test_loss_all = []
test_accur_all = []
test_loss = 0.0  # 初始化测试损失
test_accuracy = 0.0  # 初始化测试准确率
test_num = 0  # 测试样本总数

# 使用 no_grad()，避免计算梯度（测试阶段不需要反向传播）
with torch.no_grad():
    test_bar = tqdm(test_loader)  # 可视化测试进度条
    for data in test_bar:
        # 从数据加载器中获取输入数据和目标标签
        inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)

        # 分离 xyz 坐标和其他特征点
        xyz, points = inputs[:, :, :3], inputs[:, :, 3:]

        # 前向传播，获取模型输出
        outputs = model(xyz.to(device), points.to(device))

        # 计算损失
        loss2 = loss(outputs, target.to(device))

        # 获取预测的类别（通过 argmax 获取最大值的索引）
        outputs = torch.argmax(outputs, dim=1)

        # 累计损失（按样本数量加权平均）
        test_loss += abs(loss2.item()) * inputs.size(0)

        # 计算准确率
        accuracy = torch.sum(outputs == target.to(device))  # 预测结果和目标比较
        test_accuracy += accuracy  # 累计正确预测的样本数
        test_num += xyz.shape[0]  # 更新测试样本总数

        # 计算当前批次后的累积准确率
        accuracy_after_pruning = test_accuracy / test_num

    # 打印最终的测试损失和准确率
    print("test-Loss：{} , test-accuracy-after-pruning：{}".format(test_loss / test_num, accuracy_after_pruning))

    # 保存测试结果
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)