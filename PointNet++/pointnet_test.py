# ===========================================
# Import standard Python libraries
# ===========================================
import argparse  # For parsing command-line arguments
import numpy as np  # For numerical operations
import os  # For file and directory operations
import time  # For tracking time and timestamps
import math  # For mathematical functions
import random  # For generating random numbers
import pandas as pd  # For structured data manipulation with DataFrame

# ===========================================
# Import PyTorch core libraries
# ===========================================
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional operations (e.g. activation, conv)
from torch.utils.data import DataLoader, Dataset  # Data loading and custom dataset classes
from torch.autograd import Function  # For defining custom autograd functions

# ===========================================
# Import additional PyTorch-related utilities
# ===========================================
from torchsummary import summary  # To print model summaries similar to Keras
from torch.nn import Conv2d  # Standard 2D convolution layer
from torch.nn.modules.utils import _single, _pair, _triple  # Utility functions for convolution parameters
from torch.utils.tensorboard import SummaryWriter  # For logging metrics to TensorBoard

# ===========================================
# Import external libraries for ML pipelines
# ===========================================
from einops.layers.torch import Rearrange, Reduce  # For tensor rearrangement and reduction
import torchvision.models  # Pretrained models from torchvision
import torchvision.datasets as dset  # Standard datasets
from torchvision import transforms, utils  # Data transformations and visualization utilities
from matplotlib import pyplot as plt  # For plotting
from tqdm import tqdm  # For progress bars
import scipy.spatial.distance  # For computing distances between points
from pathlib import Path  # For object-oriented file paths
from collections import Counter  # For counting frequencies of elements

import kagglehub
from pathlib import Path
from torch.utils.data import DataLoader

# ===========================================
# Function: ball_query
# Purpose: For each query point, find up to K neighbors within a radius
# Inputs:
#   xyz - (B, N, 3) original points
#   new_xyz - (B, M, 3) query points
#   radius - float, search radius
#   K - int, max number of neighbors to sample
# Returns:
#   grouped_inds - (B, M, K) indices of sampled neighbors
# ===========================================
def ball_query(xyz, new_xyz, radius, K):
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N  # Mask out distances greater than radius
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]  # Keep top K closest
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)  # Replace invalid indices with the first valid one
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds

# ===========================================
# Function: get_dists
# Purpose: Calculate pairwise Euclidean distances between two sets of points
# Inputs:
#   points1 - (B, M, C)
#   points2 - (B, N, C)
# Returns:
#   dists - (B, M, N) Euclidean distances
# ===========================================
def get_dists(points1, points2):
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Prevent negative distances due to precision
    return torch.sqrt(dists).float()


# ===========================================
# Function: gather_points
# Purpose: Gather specific points from input using provided indices
# Inputs:
#   points - (B, N, C)
#   inds - (B, M) or (B, M, K)
# Returns:
#   gathered_points - (B, M, C) or (B, M, K, C)
# ===========================================
def gather_points(points, inds):
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

# ===========================================
# Function: setup_seed
# Purpose: Set random seed for reproducibility
# Inputs:
#   seed - int, random seed
# ===========================================
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True  # Ensure deterministic results for cuDNN
    torch.manual_seed(seed)  # Set CPU seed
    torch.cuda.manual_seed_all(seed)  # Set all GPU seeds
    np.random.seed(seed)  # Set NumPy seed

def fps(xyz, M):
    '''
    Perform Farthest Point Sampling (FPS) to select M representative points from the input point cloud.

    :param xyz: torch.Tensor, shape (B, N, 3)
                Batch of input point clouds with N points and 3D coordinates.
    :param M: int
              Number of points to sample.
    :return: torch.Tensor, shape (B, M)
             Indices of the sampled points for each batch.
    '''
    device = xyz.device
    B, N, C = xyz.shape

    # Initialize tensor to store indices of sampled centroids
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)

    # Initialize distances to a large value; will store minimum distances to sampled points so far
    dists = torch.ones(B, N).to(device) * 1e5

    # Randomly initialize the first sampled point index for each batch
    inds = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)

    # Batch index tensor for gathering
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)

    for i in range(M):
        # Assign current sampled indices to centroids
        centroids[:, i] = inds

        # Get coordinates of newly sampled points: shape (B, 3)
        cur_point = xyz[batchlists, inds, :]

        # Compute distances from current sampled point to all other points
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)

        # Update distances with the minimum distance so far for each point
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]

        # Choose the next farthest point as the next sampled point
        inds = torch.max(dists, dim=1)[1]

    return centroids

def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    Sample M points and group their K-nearest neighbors within a given radius.

    :param xyz: torch.Tensor, shape (B, N, 3)
                Input point cloud coordinates.
    :param points: torch.Tensor, shape (B, N, C)
                   Additional point features.
    :param M: int
              Number of points to sample.
    :param radius: float
                   Search radius for ball query.
    :param K: int
              Maximum number of neighbors in each group.
    :param use_xyz: bool
                    Whether to concatenate xyz coordinates to point features.
    :return:
        new_xyz: (B, M, 3) sampled point coordinates
        new_points: (B, M, K, C+3) grouped point features
        grouped_inds: (B, M, K) indices of grouped neighbors
        grouped_xyz: (B, M, K, 3) grouped neighbor coordinates relative to new_xyz
    '''
    # Sample M points using Farthest Point Sampling
    new_xyz = gather_points(xyz, fps(xyz, M))

    # Group K-nearest neighbors within the radius for each sampled point
    grouped_inds = ball_query(xyz, new_xyz, radius, K)
    grouped_xyz = gather_points(xyz, grouped_inds)

    # Normalize grouped coordinates by subtracting new_xyz to get relative positions
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)

    if points is not None:
        # Gather features of grouped points
        grouped_points = gather_points(points, grouped_inds)

        if use_xyz:
            # Concatenate grouped relative xyz with point features
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, grouped_inds, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Group all points into a single group, used for global feature extraction.

    :param xyz: torch.Tensor, shape (B, M, 3)
                Input point cloud coordinates.
    :param points: torch.Tensor, shape (B, M, C)
                   Additional point features.
    :param use_xyz: bool
                    Whether to concatenate xyz coordinates to features.
    :return:
        new_xyz: (B, 1, 3) dummy centroid (set to zero)
        new_points: (B, 1, M, C+3) all point features grouped
        grouped_inds: (B, 1, M) indices of all points
        grouped_xyz: (B, 1, M, 3) all point coordinates
    '''
    B, M, C = xyz.shape

    # Initialize dummy centroid tensor (unused for global pooling)
    new_xyz = torch.zeros(B, 1, C)

    # Group indices: simply all points
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)

    # Reshape xyz for grouped_xyz output
    grouped_xyz = xyz.view(B, 1, M, C)

    if points is not None:
        if use_xyz:
            # Concatenate xyz with features
            new_points = torch.cat([xyz.float(), points.float()], dim=2)
        else:
            new_points = points

        # Add singleton group dimension
        new_points = torch.unsqueeze(new_points, dim=1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, grouped_inds, grouped_xyz

class PointNet_SA_Module(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        '''
        Initialize PointNet Set Abstraction (SA) Module with quantized convolutions.

        :param M: int, number of sampled points (None if group_all is True)
        :param radius: float, ball query radius for neighborhood search
        :param K: int, maximum number of neighbors per sampled point
        :param in_channels: int, input feature channel dimension
        :param mlp: list of int, output channels for MLP layers
        :param group_all: bool, whether to group all points (global abstraction)
        :param bn: bool, whether to use BatchNorm
        :param pooling: str, pooling type: 'max' or 'avg'
        :param use_xyz: bool, whether to concatenate xyz coordinates to point features
        '''
        super(PointNet_SA_Module, self).__init__()

        # Save initialization parameters as class attributes
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz

        # Build MLP backbone as a sequential container of QuantizeConv2d + BN + ReLU layers
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            # Add quantized convolution layer
            self.backbone.add_module('Conv{}'.format(i),
                                     QuantizeConv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
            if bn:
                # Add BatchNorm layer if enabled
                self.backbone.add_module('Bn{}'.format(i), nn.BatchNorm2d(out_channels))
            # Add ReLU activation layer
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz, points):
        '''
        Forward pass for SA module.

        :param xyz: tensor of shape (B, N, 3), input point cloud coordinates
        :param points: tensor of shape (B, N, C), input point features
        :return:
            new_xyz: (B, M, 3), sampled point coordinates
            new_points: (B, M, mlp[-1]), aggregated features for sampled points
        '''
        # Group all points if group_all is True (global abstraction)
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            # Otherwise, sample M points and group neighbors within radius
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(
                xyz=xyz, points=points, M=self.M, radius=self.radius, K=self.K, use_xyz=self.use_xyz)

        # Permute to (B, C_in, K, M) for Conv2d input format
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())

        # Pool features across K neighbors using max or average pooling
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]

        # Permute back to (B, M, C_out)
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class PointNet_SA_Module_1(nn.Module):
    def __init__(self, M, radius, K, in_channels, mlp, group_all, bn=True, pooling='max', use_xyz=True):
        '''
        Initialize PointNet Set Abstraction (SA) Module with standard Conv2d.

        :param M: int, number of sampled points (None if group_all is True)
        :param radius: float, ball query radius for neighborhood search
        :param K: int, maximum number of neighbors per sampled point
        :param in_channels: int, input feature channel dimension
        :param mlp: list of int, output channels for MLP layers
        :param group_all: bool, whether to group all points (global abstraction)
        :param bn: bool, whether to use BatchNorm
        :param pooling: str, pooling type: 'max' or 'avg'
        :param use_xyz: bool, whether to concatenate xyz coordinates to point features
        '''
        super(PointNet_SA_Module_1, self).__init__()

        # Save initialization parameters as class attributes
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz

        # Build MLP backbone as a sequential container of standard Conv2d + BN + ReLU layers
        self.backbone = nn.Sequential()
        for i, out_channels in enumerate(mlp):
            # Add standard Conv2d layer (no quantization)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
            if bn:
                # Add BatchNorm layer if enabled
                self.backbone.add_module('Bn{}'.format(i), nn.BatchNorm2d(out_channels))
            # Add ReLU activation layer
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz, points):
        '''
        Forward pass for SA module.

        :param xyz: tensor of shape (B, N, 3), input point cloud coordinates
        :param points: tensor of shape (B, N, C), input point features
        :return:
            new_xyz: (B, M, 3), sampled point coordinates
            new_points: (B, M, mlp[-1]), aggregated features for sampled points
        '''
        # Group all points if group_all is True (global abstraction)
        if self.group_all:
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            # Otherwise, sample M points and group neighbors within radius
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(
                xyz=xyz, points=points, M=self.M, radius=self.radius, K=self.K, use_xyz=self.use_xyz)

        # Permute to (B, C_in, K, M) for Conv2d input format
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())

        # Pool features across K neighbors using max or average pooling
        if self.pooling == 'avg':
            new_points = torch.mean(new_points, dim=2)
        else:
            new_points = torch.max(new_points, dim=2)[0]

        # Permute back to (B, M, C_out)
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points

class PointNet_SA_Module_MSG(nn.Module):
    def __init__(self, M, radiuses, Ks, in_channels, mlps, bn=True, pooling='max', use_xyz=True):
        '''
        Initialize PointNet++ Set Abstraction (SA) Module with Multi-Scale Grouping (MSG).

        :param M: int, number of sampled points (e.g. using FPS)
        :param radiuses: list of float, neighborhood radii for each scale
        :param Ks: list of int, maximum number of neighbors to group at each scale
        :param in_channels: int, number of input feature channels
        :param mlps: list of list, defines MLP layers per scale [[mlp1_channels], [mlp2_channels], ...]
        :param bn: bool, whether to use BatchNorm after Conv layers
        :param pooling: str, type of pooling to use ('max' or 'avg')
        :param use_xyz: bool, whether to concatenate xyz coordinates to grouped features
        '''
        super(PointNet_SA_Module_MSG, self).__init__()

        # Store module hyperparameters
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz

        # Initialize a list to store backbone MLPs for each scale
        self.backbones = nn.ModuleList()

        # Build backbone MLPs for each scale
        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels

            # Build each Conv-BN-ReLU block for the current scale's MLP
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                    QuantizeConv2d(in_channels, out_channels, 1,
                                                   stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                        nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())
                in_channels = out_channels

            # Append the built backbone to the module list
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        '''
        Forward pass for Multi-Scale Grouping SA module.

        :param xyz: tensor of shape (B, N, 3), input point cloud coordinates
        :param points: tensor of shape (B, N, C), input point features
        :return:
            new_xyz: (B, M, 3), sampled point coordinates
            new_points_concat: (B, M, sum of output channels from all scales), concatenated multi-scale features
        '''
        # Sample M points from input xyz using Farthest Point Sampling
        new_xyz = gather_points(xyz, fps(xyz, self.M))

        # Initialize a list to store features computed at each scale
        new_points_all = []

        # Process each scale separately
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]

            # Perform ball query to group neighbors within radius
            grouped_inds = ball_query(xyz, new_xyz, radius, K)
            grouped_xyz = gather_points(xyz, grouped_inds)

            # Normalize grouped points relative to new_xyz (center the local neighborhood)
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)

            if points is not None:
                grouped_points = gather_points(points, grouped_inds)

                # Optionally concatenate xyz coordinates to grouped point features
                if self.use_xyz:
                    new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz

            # Permute dimensions to match Conv2d input format: (B, C_in, K, M)
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())

            # Apply pooling across the neighborhood dimension (K)
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]

            # Permute back to (B, M, C_out)
            new_points = new_points.permute(0, 2, 1).contiguous()

            # Append the features from this scale to the list
            new_points_all.append(new_points)

        # Concatenate features from all scales along the last dimension (channel dimension)
        return new_xyz, torch.cat(new_points_all, dim=-1)

class Quantize_8bit(Function):
    def __init__(self):
        super(Quantize_8bit, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=2):
        '''
        Forward pass for quantization function labeled as "8bit", 
        but default nbits is set to 2 (likely a naming mismatch).

        :param ctx: context object to save information for backward pass
        :param input: input tensor to quantize
        :param nbits: number of bits for quantization (default = 2)
        :return: quantized output tensor
        '''
        ctx.save_for_backward(input)

        # Calculate scale factor based on int8 range (-128 to 127)
        s = (input.max() - input.min()) / (127 - (-128))

        # Calculate zero point (offset) to align input max to 127
        zero_point = 127 - input.max() / s

        # Apply quantization: scale, shift, round
        out = torch.round(input / s + zero_point)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        '''
        Backward pass using Straight-Through Estimator (STE).
        Passes gradients unchanged through quantization.

        :param ctx: context with saved tensors from forward
        :param grad_outputs: gradient flowing from upper layers
        :return: gradient w.r.t input, None for nbits
        '''
        input = ctx.saved_tensors
        grad_input = grad_outputs

        # Optional gradient clipping (commented out)
        # grad_input[input[0].gt(1)] = 0
        # grad_input[input[0].lt(-1)] = 0

        return grad_input, None

class Quantize(Function):
    def __init__(self):
        super(Quantize, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=8):
        '''
        Generic n-bit symmetric quantization function.

        :param ctx: context for backward computation
        :param input: input tensor to be quantized
        :param nbits: number of quantization bits (default = 8)
        :return: quantized output tensor
        '''
        ctx.save_for_backward(input)

        # Calculate quantization integer range based on nbits
        q_max = 2 ** (nbits - 1) - 1  # e.g. +127 for 8-bit
        q_min = -2 ** (nbits - 1)     # e.g. -128 for 8-bit

        # Calculate scale factor (s)
        s = (input.max() - input.min()) / (q_max - q_min)

        # Calculate zero point (z) to align input min with q_min
        z = -(input.min() / s) + q_min

        # Quantize input: scale, shift, round to nearest integer
        out = torch.round(input / s + z)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        '''
        Straight-Through Estimator (STE) backward pass.

        :param ctx: context with saved tensors from forward
        :param grad_outputs: upstream gradients
        :return: gradient w.r.t input, None for nbits
        '''
        input = ctx.saved_tensors
        grad_input = grad_outputs
        return grad_input, None

class Quantize_int8(Function):
    def __init__(self):
        super(Quantize, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=8):
        '''
        Forward pass for int8 quantization function.

        :param ctx: context for backward computation
        :param input: input tensor to quantize
        :param nbits: number of bits for quantization (default = 8)
        :return: quantized output tensor
        '''
        ctx.save_for_backward(input)

        # Calculate quantization integer range based on nbits
        q_max = 2 ** (nbits - 1) - 1  # +127 for 8-bit
        q_min = -2 ** (nbits - 1)     # -128 for 8-bit

        # Calculate scale factor (s)
        s = (input.max() - input.min()) / (q_max - q_min)

        # Calculate zero point (z) to align input.min() with q_min
        z = -(input.min() / s) + q_min

        # Quantize: scale input, shift, round to nearest integer
        out = torch.round(input / s + z)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        '''
        Backward pass using Straight-Through Estimator (STE).
        Passes gradients unchanged through quantization.

        :param ctx: context with saved tensors from forward
        :param grad_outputs: upstream gradients
        :return: gradient w.r.t input, None for nbits
        '''
        input = ctx.saved_tensors
        grad_input = grad_outputs
        return grad_input, None

class TernaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        '''
        Forward pass for ternary quantization.

        Maps input values into {-1, 0, +1} based on their relative position 
        within the input's min-max range divided into three equal intervals.

        :param ctx: context object to save variables for backward computation
        :param input: input tensor to quantize
        :return: quantized tensor with values in {-1, 0, +1}
        '''
        ctx.save_for_backward(input)

        # Calculate min and max of the input tensor
        ctx_max, ctx_min = torch.max(input), torch.min(input)

        # Define lower and higher thresholds by splitting the range into thirds
        lower_interval = ctx_min + (ctx_max - ctx_min) / 3
        higher_interval = ctx_max - (ctx_max - ctx_min) / 3

        # Quantize to -1 if below lower_interval, +1 if above higher_interval, otherwise 0
        out = torch.where(input < lower_interval,
                          torch.tensor(-1.).to(input.device, input.dtype),
                          input)
        out = torch.where(input > higher_interval,
                          torch.tensor(1.).to(input.device, input.dtype),
                          out)
        out = torch.where((input >= lower_interval) & (input <= higher_interval),
                          torch.tensor(0.).to(input.device, input.dtype),
                          out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Backward pass using Straight-Through Estimator (STE).

        Blocks gradients for input values outside [-1, +1].

        :param ctx: context with saved input tensor from forward
        :param grad_output: gradient flowing from upper layers
        :return: gradient w.r.t input
        '''
        input = ctx.saved_tensors
        grad_input = grad_output

        # Zero gradients for inputs outside the [-1, +1] range
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0

        return grad_input

class TriConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        '''
        2D Convolution layer with ternary weight quantization.

        Inherits from PyTorch's low-level _ConvNd for full configurability.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of convolution kernel
        :param stride: stride of convolution
        :param padding: zero-padding added to both sides
        :param dilation: spacing between kernel elements
        :param groups: number of blocked connections from input channels to output channels
        :param bias: whether to include bias term
        :param padding_mode: type of padding ('zeros', 'circular', etc.)
        '''
        # Convert parameters to tuple form as required by _ConvNd
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        # Initialize parent _ConvNd class
        super(TriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        '''
        Forward pass with ternary quantized weights.

        :param input: input tensor of shape (B, C_in, H, W)
        :return: output tensor after applying convolution with ternary quantized weights
        '''
        bw = self.weight  # Get convolution weights
        ba = input        # Input tensor

        # Center weights by subtracting their mean for better quantization stability
        bw = bw - bw.mean()

        # Apply ternary quantization to weights
        bw = TernaryQuantize().apply(bw)

        # Update internal weight data with quantized weights
        self.weight.data = bw

        # If padding mode is 'circular', apply circular padding before convolution
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        # Standard convolution operation
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def quantize_2bit(input_tensor, min_val=-1.5, max_val=1.5):
    '''
    Perform custom 2-bit quantization on an input tensor.

    Maps input tensor values into discrete levels within a specified min-max range,
    with special adjustment to avoid exact values of -2, -1, 0, or +1.

    :param input_tensor: torch tensor, input tensor to quantize
    :param min_val: float, minimum clamping value (default = -1.5)
    :param max_val: float, maximum clamping value (default = +1.5)
    :return: torch tensor, quantized tensor with values mapped to predefined levels
    '''

    # Clamp tensor values to [min_val, max_val] to avoid overflow
    clamped_tensor = torch.clamp(input_tensor, min=min_val, max=max_val)

    # Define scaling factor (s) and zero-point offset (z)
    # Here s=2.0 means quantization step is 0.5 (since 1/s = 0.5), z=0.5 shifts before rounding
    s = 2.0
    z = 0.5

    # Quantize to the nearest half-integer:
    # Multiply by s, subtract z to shift, round to nearest integer, then divide back by s
    quantized_values = torch.round(clamped_tensor * s - z) / s

    # Adjust quantized values to avoid exact problematic values of 1.0, -1.0, 0.0, or -2.0
    # Adds +0.5 to those values to shift them slightly, preventing issues with extreme quantized outputs
    mask = (quantized_values == 1.0) | \
           (quantized_values == -1.0) | \
           (quantized_values == 0.0) | \
           (quantized_values == -2.0)
    quantized_values = torch.where(mask, quantized_values + 0.5, quantized_values)

    return quantized_values

class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        '''
        Custom convolution layer with input and weight quantization.

        Inherits from nn.Conv2d and overrides the forward pass to apply quantization
        to both inputs and weights before performing the convolution operation.
        '''
        super(QuantizeConv2d, self).__init__(*args, **kwargs)
        # Optional quantization range definitions (commented out)
        # self.qmin, self.qmax = -2 ** (8 - 1), 2 ** (8 - 1) - 1
    
    def forward(self, x):
        '''
        Forward pass for quantized convolution.

        Quantizes both the input tensor and convolution weight tensor before performing convolution.

        :param x: torch tensor, input feature map of shape (B, C_in, H, W)
        :return: torch tensor, output feature map after quantized convolution
        '''

        # ====== INPUT QUANTIZATION ======
        # Apply quantization to input tensor
        # Originally commented options for 2-bit clamped quantization
        # x_q = torch.clamp(x, min=-1.5, max=1.5)
        # x_q = torch.round(x_q * 2) / 2

        x_q = Quantize.apply(x)  # Uses custom Quantize function (default nbits=8)

        # Alternative: use unquantized input
        # x_q = x
        
        # ====== WEIGHT QUANTIZATION ======
        weight = self.weight

        # Apply quantization to weight tensor
        # Originally commented options for 2-bit clamped quantization
        # weight_q = torch.clamp(weight, min=-1.5, max=1.5)
        # weight_q = torch.round(weight_q * 2) / 2

        weight_q = Quantize.apply(weight)  # Uses custom Quantize function (default nbits=8)

        # ====== CONVOLUTION OPERATION ======
        output = F.conv2d(x_q, weight_q, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        # ====== OPTIONAL DEQUANTIZATION (commented out) ======
        # Convert output back to dequantized range if needed
        # s = (self.weight.max() - self.weight.min()) / (self.qmax - self.qmin)
        # output = s * (output - z)
        
        return output
    
class QuantizeConv2d_1(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        '''
        Custom convolution layer that applies input and weight quantization before performing convolution.

        Inherits directly from PyTorch's low-level _ConvNd for maximum configurability.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride of the convolution.
        :param padding: Zero-padding added to both sides of input.
        :param dilation: Spacing between kernel elements.
        :param groups: Number of blocked connections from input channels to output channels.
        :param bias: If True, adds a learnable bias.
        :param padding_mode: Type of padding ('zeros', 'circular', etc.).
        '''
        # Convert parameters to tuple forms required by _ConvNd
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        # Initialize the parent _ConvNd class
        super(QuantizeConv2d_1, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        '''
        Forward pass with quantized input and weight.

        :param input: Input tensor of shape (B, C_in, H, W).
        :return: Output tensor after quantized convolution.
        '''
        bw = self.weight  # Convolution weights
        ba = input        # Input feature map

        # Apply quantization to both input and weights using Quantize function
        ba = Quantize().apply(ba)
        bw = Quantize().apply(bw)

        # Update internal weight data with quantized weights
        self.weight.data = bw

        # Perform convolution operation, handling circular padding if necessary
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            return F.conv2d(ba, bw, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        '''
        PointNet++ Classification Network with Single-Scale Grouping (SSG).

        :param in_channels: Number of input feature channels (e.g. 3 for xyz).
        :param nclasses: Number of output classes for classification.
        '''
        super(pointnet2_cls_ssg, self).__init__()

        # Set Abstraction layers with increasing feature dimensions
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=in_channels,
                                         mlp=[64, 64, 128], group_all=False)
        self.pt_sa2 = PointNet_SA_Module(M=512, radius=0.2, K=32, in_channels=131,
                                         mlp=[128, 128, 256], group_all=False)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None, in_channels=259,
                                         mlp=[256, 512, 1024], group_all=True)

        # Fully connected layers for final classification
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.cls = nn.Linear(256, nclasses)

    def forward(self, xyz, points):
        '''
        Forward pass of the classification network.

        :param xyz: Tensor of shape (B, N, 3), input point coordinates.
        :param points: Tensor of shape (B, N, C), additional point features.
        :return: Tensor of shape (B, nclasses), classification logits.
        '''
        batchsize = xyz.shape[0]

        # Pass through three hierarchical Set Abstraction layers
        new_xyz, new_points = self.pt_sa1(xyz, points)
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)

        # Flatten global feature
        net = new_points.view(batchsize, -1)

        # Pass through fully connected layers with BatchNorm, ReLU, Dropout
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))

        # Final classification layer
        net = self.cls(net)

        return net

def read_off(file):
    '''
    Read an OFF (Object File Format) file to extract mesh vertices and faces.

    :param file: Opened OFF file object.
    :return: verts (list of vertex coordinates), faces (list of face vertex indices).
    '''
    # Check OFF file header
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')

    # Read number of vertices and faces
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])

    # Read vertex coordinates
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]

    # Read face indices, ignoring the first entry indicating number of vertices per face
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]

    return verts, faces

class PointSampler(object):
    def __init__(self, output_size):
        '''
        Initialize PointSampler to sample a fixed number of points from a mesh.

        :param output_size: int, number of points to sample.
        '''
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        '''
        Calculate the area of a triangle given its three vertices using Heron's formula.

        :param pt1: numpy array, vertex 1 coordinates.
        :param pt2: numpy array, vertex 2 coordinates.
        :param pt3: numpy array, vertex 3 coordinates.
        :return: float, area of the triangle.
        '''
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        '''
        Sample a random point inside a triangle using barycentric coordinates.

        :param pt1: numpy array, vertex 1 coordinates.
        :param pt2: numpy array, vertex 2 coordinates.
        :param pt3: numpy array, vertex 3 coordinates.
        :return: tuple, sampled point coordinates.
        '''
        # Barycentric sampling for uniform point sampling on a triangle surface
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        '''
        Sample points uniformly from the surface of a mesh.

        :param mesh: tuple containing vertices and faces (verts, faces).
        :return: numpy array of sampled point coordinates with shape (output_size, 3).
        '''
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        # Calculate area of each face for weighted sampling
        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[faces[i][0]],
                                          verts[faces[i][1]],
                                          verts[faces[i][2]])

        # Sample faces proportional to their area to ensure uniform surface sampling
        sampled_faces = random.choices(faces, weights=areas, k=self.output_size)

        sampled_points = np.zeros((self.output_size, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(verts[sampled_faces[i][0]],
                                                  verts[sampled_faces[i][1]],
                                                  verts[sampled_faces[i][2]])

        return sampled_points

class Normalize(object):
    def __call__(self, pointcloud):
        '''
        Normalize the point cloud to zero mean and scale to unit sphere.

        :param pointcloud: numpy array of shape (N, 3).
        :return: normalized point cloud.
        '''
        assert len(pointcloud.shape) == 2

        # Center the point cloud at the origin
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)

        # Scale to ensure max distance from origin is 1
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        '''
        Apply a random rotation around the Z-axis to the point cloud.

        :param pointcloud: numpy array of shape (N, 3).
        :return: rotated point cloud.
        '''
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi  # Random rotation angle
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        '''
        Add Gaussian noise to the point cloud for data augmentation.

        :param pointcloud: numpy array of shape (N, 3).
        :return: noisy point cloud.
        '''
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, pointcloud.shape)
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        '''
        Convert point cloud numpy array to PyTorch tensor.

        :param pointcloud: numpy array of shape (N, 3).
        :return: torch tensor.
        '''
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)

def default_transforms():
    '''
    Compose default data augmentation and preprocessing transforms:
    - Sample 1024 points.
    - Normalize to zero mean and unit sphere.
    - Convert to PyTorch tensor.

    :return: torchvision.transforms.Compose object.
    '''
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        '''
        Custom PyTorch Dataset for loading point cloud data from ModelNet.

        :param root_dir: string or Path, dataset root directory
        :param valid: bool, if True use validation transforms
        :param folder: string, subfolder name ('train' or 'test')
        :param transform: torchvision transform pipeline applied to data
        '''
        # Convert root_dir to Path object
        self.root_dir = Path(root_dir)

        # If root_dir contains 'ModelNet10' subfolder, adjust path to point to it
        if "ModelNet10" in os.listdir(self.root_dir):
            self.root_dir = self.root_dir / "ModelNet10"

        # List all category folders under root_dir (bathtub, chair, etc.)
        folders = [directory for directory in sorted(os.listdir(self.root_dir))
                   if os.path.isdir(self.root_dir / directory)]

        # Create class name to index mapping
        self.classes = {folder: i for i, folder in enumerate(folders)}

        # Select transform pipeline
        self.transforms = transform if not valid else default_transforms()

        self.valid = valid
        self.files = []  # Store sample file paths and labels

        # Iterate over each category folder to gather .off files
        for category in self.classes.keys():
            new_dir = self.root_dir / category / folder

            # Skip if folder does not exist
            if not new_dir.exists():
                print(f"[Warning] Directory {new_dir} does not exist. Skipping.")
                continue

            # For each .off file, store path and category
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir / file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        '''
        Return number of samples in the dataset.
        '''
        return len(self.files)

    def __preproc__(self, file):
        '''
        Preprocess an OFF file into point cloud.

        :param file: opened OFF file object
        :return: processed point cloud tensor
        '''
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        '''
        Get one sample from dataset.

        :param idx: int, index of sample
        :return: dict with keys 'pointcloud' and 'category'
        '''
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']

        # Open OFF file and preprocess
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)

        return {'pointcloud': pointcloud,
                'category': self.classes[category]}

    
train_transforms = transforms.Compose([
    PointSampler(1024),    # Sample 1024 points from mesh faces
    Normalize(),           # Normalize to zero-mean and unit sphere
    RandRotation_z(),      # Random rotation around Z axis
    RandomNoise(),         # Add Gaussian noise
    ToTensor()             # Convert to PyTorch tensor
])

# ===========================================
# 1. Download ModelNet10 dataset using KaggleHub
# ===========================================

# Download dataset and get extracted directory path
kaggle_dataset_path = kagglehub.dataset_download("balraj98/modelnet10-princeton-3d-object-dataset")
print("Dataset directory:", kaggle_dataset_path)

# Convert to Path object
dataset_dir = Path(kaggle_dataset_path)

# ===========================================
# 2. Initialize train and validation datasets
# Make sure PointCloudData and train_transforms are defined in your project
# ===========================================

train_ds = PointCloudData(dataset_dir, transform=train_transforms)
valid_ds = PointCloudData(dataset_dir, valid=True, folder='test', transform=train_transforms)

# ===========================================
# 3. Build classes dict from dataset folders
# ===========================================
classes = {folder: i for i, folder in enumerate(sorted(train_ds.classes))}
inv_classes = {i: cat for cat, i in classes.items()}

# ===========================================
# 4. Print dataset information
# ===========================================
print('Train dataset size:', len(train_ds))
print('Valid dataset size:', len(valid_ds))
print('Number of classes:', len(train_ds.classes))
print('Sample pointcloud shape:', train_ds[0]['pointcloud'].size())
print('Class:', inv_classes[train_ds[0]['category']])

# ===========================================
# 5. Create DataLoaders
# ===========================================
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=valid_ds, batch_size=64)

# Function to calculate Hamming distance between two tensors
def calculate_hamming_distance(weight1, weight2):
    '''
    Calculate Hamming distance between two weight tensors.

    :param weight1: torch tensor, first weight tensor
    :param weight2: torch tensor, second weight tensor
    :return: int, Hamming distance (number of differing elements)
    '''

    # Reshape both weight tensors into 1D vectors for element-wise comparison
    weights1 = weight1.view(-1)
    weights2 = weight2.view(-1)

    # Calculate Hamming distance:
    # Count number of positions where weights1 and weights2 differ
    hamming_distance = (weight1 != weight2).sum().item()

    return hamming_distance

def euclidean_distance(A, B):
    '''
    Calculate Euclidean distance between two tensors.

    :param A: torch tensor, first input tensor
    :param B: torch tensor, second input tensor
    :return: float tensor, Euclidean distance between A and B
    '''

    # Compute the L2 norm of the difference between A and B
    # Equivalent to sqrt(sum((A - B)^2))
    distance = torch.norm(A - B)

    # Return the computed Euclidean distance
    return distance

# ==============================
# Set device (use GPU if available)
# ==============================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using {} device.".format(device))

# ==============================
# Initialize model and load pretrained weights
# ==============================
model = pointnet2_cls_ssg(in_channels=3, nclasses=10).to(device)

# Load pruned model weights from saved checkpoint
model.load_state_dict(torch.load('/home/songqiwang/Project/CIMCAM/github/PointNet++/data/20250627_204741/weights_epoch_2_after_pruning.pth'))

# Set model to evaluation mode (disable dropout, fix batchnorm)
model.eval()

# ==============================
# Define loss function for evaluation
# ==============================
loss_fn = nn.CrossEntropyLoss()

# ==============================
# Initialize evaluation metrics
# ==============================
test_loss = 0.0
test_accuracy = 0.0
test_num = 0

# ==============================
# Evaluate model on test set
# ==============================
with torch.no_grad():
    test_bar = tqdm(test_loader, desc="Testing")
    for data in test_bar:
        # Extract input point cloud and labels
        inputs = data['pointcloud'].to(device).float()
        targets = data['category'].to(device)

        # Split input into xyz coordinates and additional features (if any)
        xyz, points = inputs[:, :, :3], inputs[:, :, 3:]

        # Forward pass through the model
        outputs = model(xyz, points)

        # Compute loss
        batch_loss = loss_fn(outputs, targets)
        test_loss += batch_loss.item() * inputs.size(0)

        # Compute accuracy
        preds = torch.argmax(outputs, dim=1)
        correct = torch.sum(preds == targets)
        test_accuracy += correct.item()
        test_num += inputs.size(0)

# ==============================
# Compute and print final metrics
# ==============================
avg_test_loss = test_loss / test_num
avg_test_accuracy = test_accuracy / test_num

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
