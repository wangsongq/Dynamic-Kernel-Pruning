# ===========================================
# Standard library imports
# ===========================================
import argparse  # For parsing command line arguments
import numpy as np  # For numerical operations using NumPy
import os  # For operating system dependent functionalities such as file paths
import time  # For measuring time intervals
import math  # For basic mathematical functions
import random  # For random number generation
import pandas as pd  # For data analysis and manipulation using DataFrame

# ===========================================
# PyTorch core imports
# ===========================================
import torch  # PyTorch main library
import torch.nn as nn  # For building neural network layers and models
import torch.nn.functional as F  # For functional interface (e.g. activation functions, conv operations)
from torch.utils.data import DataLoader, Dataset  # For data loading and custom datasets
from torch.utils.tensorboard import SummaryWriter  # For logging training metrics to TensorBoard
from torch.autograd import Function  # For custom autograd functions

# ===========================================
# Torchvision imports for models, transforms, and datasets
# ===========================================
import torchvision.models  # Pretrained models in torchvision
import torchvision.datasets as dset  # Standard datasets like MNIST, CIFAR10, etc.
from torchvision import transforms, utils  # Data augmentation and visualization utilities

# ===========================================
# Other scientific computing and utility imports
# ===========================================
from torchsummary import summary  # For printing model summaries like in Keras
import scipy.spatial.distance  # For computing spatial distances, e.g., Euclidean, cosine

# ===========================================
# PyTorch utility functions for convolution padding and dimension handling
# ===========================================
from torch.nn.modules.utils import _single, _pair, _triple  # For converting parameters to tuple form

# ===========================================
# Einops for tensor rearrangement and reduction operations
# ===========================================
from einops.layers.torch import Rearrange, Reduce  # For reshaping and reducing tensor dimensions

# ===========================================
# Plotting and visualization imports
# ===========================================
from matplotlib import pyplot as plt  # For plotting graphs and visualizations
from tqdm import tqdm  # For displaying progress bars in loops

# ===========================================
# Pathlib and collections for path handling and counter utilities
# ===========================================
from pathlib import Path  # For object-oriented file system paths
from collections import Counter  # For counting hashable objects efficiently

import time
time = time.strftime('%Y%m%d_%H%M%S')
# Get current script directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct directory path: ./data/{time}/
save_dir = os.path.join(current_dir, f'data/{time}')
# Check if the directory exists; if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def ball_query(xyz, new_xyz, radius, K):
    '''
    For each query point in new_xyz, find up to K neighboring points within a given radius from xyz.

    :param xyz: Tensor of shape (B, N, 3), original point cloud coordinates
    :param new_xyz: Tensor of shape (B, M, 3), query points
    :param radius: float, search radius
    :param K: int, maximum number of neighbors to sample
    :return: grouped_inds of shape (B, M, K), indices of sampled neighbors
    '''
    device = xyz.device
    B, N, C = xyz.shape  # Batch size, number of points, coordinate dim
    M = new_xyz.shape[1]  # Number of query points

    # Initialize grouped_inds as (B, M, N), containing all indices
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)

    # Compute pairwise distances between each new_xyz and xyz
    dists = get_dists(new_xyz, xyz)

    # Mask out points beyond radius by setting their indices to N (invalid index)
    grouped_inds[dists > radius] = N

    # Sort indices along the last dimension and take top K closest neighbors
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]

    # For any invalid index (N), replace with the first valid index in that group to avoid empty neighborhood
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]

    return grouped_inds

def get_dists(points1, points2):
    '''
    Compute Euclidean distances between two sets of points.

    :param points1: Tensor of shape (B, M, C), first set of points
    :param points2: Tensor of shape (B, N, C), second set of points
    :return: dists of shape (B, M, N), pairwise Euclidean distances
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape

    # Calculate squared norms and expand for broadcasting
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)

    # Subtract dot product times 2
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))

    # Correct for negative distances due to numerical error
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)

    # Return Euclidean distances
    return torch.sqrt(dists).float()

def gather_points(points, inds):
    '''
    Gather specific points from the point cloud using given indices.

    :param points: Tensor of shape (B, N, C), original point cloud
    :param inds: Tensor of shape (B, M) or (B, M, K), indices of points to gather
    :return: gathered points of shape (B, M, C) or (B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape

    # Prepare batch indices for advanced indexing
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1

    # Generate batch indices of shape compatible with inds
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)

    # Gather points using advanced indexing
    return points[batchlists, inds, :]

def setup_seed(seed):
    '''
    Set random seed for reproducibility across torch, numpy, and CUDA.

    :param seed: int, random seed value
    '''
    torch.backends.cudnn.deterministic = True  # Make cuDNN deterministic
    torch.manual_seed(seed)  # Set torch CPU seed
    torch.cuda.manual_seed_all(seed)  # Set torch GPU seed
    np.random.seed(seed)  # Set numpy seed


def fps(xyz, M):
    '''
    Perform Farthest Point Sampling (FPS) to downsample a point cloud to M points.

    :param xyz: Tensor of shape (B, N, 3), original point cloud coordinates
    :param M: int, number of points to sample
    :return: centroids of shape (B, M), indices of sampled points
    '''
    device = xyz.device
    B, N, C = xyz.shape

    # Initialize centroids tensor to store indices of sampled points
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)

    # Initialize distance tensor with very large values
    dists = torch.ones(B, N).to(device) * 1e5

    # Randomly choose the first sampled point for each batch
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)

    # Batch indices for indexing
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)

    for i in range(M):
        # Assign current sampled point index to centroids
        centroids[:, i] = inds

        # Get coordinates of currently sampled point (B, 3)
        cur_point = xyz[batchlists, inds, :]

        # Calculate distances from current point to all other points
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)

        # Update distances: keep the minimum distance to sampled set so far
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]

        # Select the next farthest point (max distance)
        inds = torch.max(dists, dim=1)[1]

    return centroids


def sample_and_group(xyz, points, M, radius, K, use_xyz=True):
    '''
    Perform sampling and grouping:
    - Sample M points using FPS
    - For each sampled point, group up to K neighbors within a given radius
    - Optionally concatenate XYZ coordinates to point features

    :param xyz: Tensor of shape (B, N, 3), input point cloud coordinates
    :param points: Tensor of shape (B, N, C), input point features
    :param M: int, number of sampled points
    :param radius: float, search radius for ball query
    :param K: int, maximum number of neighbors in each group
    :param use_xyz: bool, whether to concatenate XYZ to features
    :return:
        new_xyz: (B, M, 3), sampled point coordinates
        new_points: (B, M, K, C+3) or (B, M, K, C), grouped point features
        grouped_inds: (B, M, K), indices of grouped neighbors
        grouped_xyz: (B, M, K, 3), grouped neighbor coordinates relative to new_xyz
    '''
    # Sample M points from xyz using FPS
    new_xyz = gather_points(xyz, fps(xyz, M))

    # Find up to K neighbors within radius for each sampled point
    grouped_inds = ball_query(xyz, new_xyz, radius, K)

    # Gather coordinates of grouped neighbors
    grouped_xyz = gather_points(xyz, grouped_inds)

    # Normalize grouped_xyz by subtracting new_xyz (relative coordinates)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)

    if points is not None:
        # Gather features of grouped neighbors
        grouped_points = gather_points(points, grouped_inds)

        if use_xyz:
            # Concatenate relative coordinates with features
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, grouped_inds, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Group all points into a single group (used in global feature extraction).

    :param xyz: Tensor of shape (B, M, 3), input point cloud coordinates
    :param points: Tensor of shape (B, M, C), input point features
    :param use_xyz: bool, whether to concatenate XYZ to features
    :return:
        new_xyz: (B, 1, 3), dummy centroid (not used)
        new_points: (B, 1, M, C+3) or (B, 1, M, C), all point features
        grouped_inds: (B, 1, M), indices of all points
        grouped_xyz: (B, 1, M, 3), all point coordinates
    '''
    B, M, C = xyz.shape

    # Dummy new_xyz (unused in global pooling)
    new_xyz = torch.zeros(B, 1, C)

    # All point indices grouped together
    grouped_inds = torch.arange(0, M).long().view(1, 1, M).repeat(B, 1, 1)

    # Reshape xyz for grouped_xyz output
    grouped_xyz = xyz.view(B, 1, M, C)

    if points is not None:
        if use_xyz:
            # Concatenate xyz with features along feature dimension
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
        PointNet Set Abstraction (SA) Module for hierarchical feature learning.

        :param M: int, number of sampled points (None for group_all)
        :param radius: float, radius for ball query
        :param K: int, max number of neighbors in each group
        :param in_channels: int, input channel dimension
        :param mlp: list, output channel sizes for MLP layers
        :param group_all: bool, if True group all points into one group (global feature)
        :param bn: bool, use BatchNorm
        :param pooling: str, 'max' or 'avg' pooling
        :param use_xyz: bool, whether to concatenate xyz coordinates with point features
        '''
        super(PointNet_SA_Module, self).__init__()

        # Store module parameters
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz

        # Define MLP backbone as a Sequential container
        self.backbone = nn.Sequential()

        # Build MLP layers dynamically based on mlp list
        for i, out_channels in enumerate(mlp):
            # Add QuantizeConv2d layer (1x1 convolution)
            self.backbone.add_module('Conv{}'.format(i),
                                     QuantizeConv2d(in_channels, out_channels, 1,
                                                    stride=1, padding=0, bias=False))
            # Add BatchNorm if enabled
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            # Add ReLU activation
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())
            # Update input channel dimension for next layer
            in_channels = out_channels

    def forward(self, xyz, points):
        '''
        Forward pass of SA module.

        :param xyz: Tensor of shape (B, N, 3), point cloud coordinates
        :param points: Tensor of shape (B, N, C), input point features
        :return:
            new_xyz: (B, M, 3), sampled point coordinates
            new_points: (B, M, mlp[-1]), aggregated features for sampled points
        '''
        # Perform sampling and grouping
        if self.group_all:
            # Group all points into one group for global feature
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            # Sample M points and group neighbors within radius
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)

        # Permute dimensions to match Conv2d input format: (B, C_in, K, M)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())

        # Apply pooling operation across the neighborhood dimension (K)
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
        PointNet Set Abstraction (SA) Module Variant:
        Uses standard nn.Conv2d instead of QuantizeConv2d for feature extraction.

        :param M: int, number of sampled points (None for group_all)
        :param radius: float, ball query search radius
        :param K: int, max number of neighbors per group
        :param in_channels: int, input feature channel dimension
        :param mlp: list, output channel sizes for MLP layers
        :param group_all: bool, if True, group all points into a single group (global feature)
        :param bn: bool, whether to use BatchNorm
        :param pooling: str, 'max' or 'avg' pooling method
        :param use_xyz: bool, whether to concatenate xyz coordinates with point features
        '''
        super(PointNet_SA_Module_1, self).__init__()

        # Store module parameters
        self.M = M
        self.radius = radius
        self.K = K
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz

        # Define MLP backbone as a Sequential container
        self.backbone = nn.Sequential()

        # Dynamically build MLP layers using standard Conv2d
        for i, out_channels in enumerate(mlp):
            # Add Conv2d layer (1x1 convolution)
            self.backbone.add_module('Conv{}'.format(i),
                                     nn.Conv2d(in_channels, out_channels, 1,
                                               stride=1, padding=0, bias=False))
            # Add BatchNorm if enabled
            if bn:
                self.backbone.add_module('Bn{}'.format(i),
                                         nn.BatchNorm2d(out_channels))
            # Add ReLU activation
            self.backbone.add_module('Relu{}'.format(i), nn.ReLU())

            # Update input channel dimension for next layer
            in_channels = out_channels

    def forward(self, xyz, points):
        '''
        Forward pass of SA module.

        :param xyz: Tensor of shape (B, N, 3), point cloud coordinates
        :param points: Tensor of shape (B, N, C), input point features
        :return:
            new_xyz: (B, M, 3), sampled point coordinates
            new_points: (B, M, mlp[-1]), aggregated features for sampled points
        '''
        # Perform sampling and grouping
        if self.group_all:
            # Group all points into one group for global feature extraction
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            # Sample M points and group K neighbors within radius
            new_xyz, new_points, grouped_inds, grouped_xyz = sample_and_group(xyz=xyz,
                                                                              points=points,
                                                                              M=self.M,
                                                                              radius=self.radius,
                                                                              K=self.K,
                                                                              use_xyz=self.use_xyz)

        # Permute new_points to match Conv2d input format: (B, C_in, K, M)
        new_points = self.backbone(new_points.permute(0, 3, 2, 1).contiguous())

        # Apply pooling operation across neighborhood dimension (K)
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
        PointNet Set Abstraction Module with Multi-Scale Grouping (MSG).

        :param M: int, number of sampled points
        :param radiuses: list of float, radii for each scale's ball query
        :param Ks: list of int, number of neighbors per scale
        :param in_channels: int, input channel dimension
        :param mlps: list of lists, each inner list defines MLP output channels per scale
        :param bn: bool, whether to use BatchNorm
        :param pooling: str, pooling type ('max' or 'avg')
        :param use_xyz: bool, whether to concatenate xyz coordinates with point features
        '''
        super(PointNet_SA_Module_MSG, self).__init__()

        # Store module parameters
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz

        # Define backbones as a ModuleList, one MLP per scale
        self.backbones = nn.ModuleList()

        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels

            # Build MLP layers for scale j
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                    QuantizeConv2d(in_channels, out_channels, 1,
                                                   stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                        nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())

                # Update input channels for next layer
                in_channels = out_channels

            # Append the MLP backbone for this scale
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        '''
        Forward pass of MSG module.

        :param xyz: Tensor of shape (B, N, 3), point cloud coordinates
        :param points: Tensor of shape (B, N, C), input point features
        :return:
            new_xyz: (B, M, 3), sampled point coordinates
            new_points_concat: (B, M, sum of output dims per scale), concatenated features across scales
        '''
        # Sample M centroids using Farthest Point Sampling
        new_xyz = gather_points(xyz, fps(xyz, self.M))

        # List to store features from each scale
        new_points_all = []

        # Loop over each scale
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]

            # Ball query: group neighbors within radius
            grouped_inds = ball_query(xyz, new_xyz, radius, K)
            grouped_xyz = gather_points(xyz, grouped_inds)

            # Normalize grouped coordinates relative to new_xyz
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)

            if points is not None:
                # Gather point features of grouped neighbors
                grouped_points = gather_points(points, grouped_inds)

                if self.use_xyz:
                    # Concatenate grouped relative xyz with features
                    new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz

            # Permute to (B, C_in, K, M) for Conv2d
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())

            # Pooling over K neighbors
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]

            # Permute back to (B, M, C_out)
            new_points = new_points.permute(0, 2, 1).contiguous()

            # Append pooled features from this scale
            new_points_all.append(new_points)

        # Concatenate features from all scales along channel dimension
        return new_xyz, torch.cat(new_points_all, dim=-1)

    
class Quantize_8bit(Function):
    def __init__(self):
        # Initialize the custom autograd Function
        super(Quantize_8bit, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=2):
        '''
        Forward pass for 8-bit quantization.

        :param ctx: context object for backward
        :param input: input tensor to quantize
        :param nbits: number of bits (default=2, but function name implies 8-bit)
        :return: quantized output tensor
        '''
        ctx.save_for_backward(input)

        # Calculate scale factor s using input min and max range mapped to int8 range (-128 to 127)
        s = (input.max() - input.min()) / (127 - (-128))

        # Calculate zero point (bias)
        zero_point = 127 - input.max() / s

        # Quantize input: scale and shift then round
        out = torch.round(input / s + zero_point)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        '''
        Backward pass: Straight-Through Estimator (STE).
        Pass gradient directly to input without modification.

        :param ctx: context with saved tensors
        :param grad_outputs: gradients from upper layers
        :return: gradient w.r.t input, None for nbits (non-tensor)
        '''
        input = ctx.saved_tensors
        grad_input = grad_outputs

        # Optionally clip gradients (commented out)
        # grad_input[input[0].gt(1)] = 0
        # grad_input[input[0].lt(-1)] = 0

        return grad_input, None

class Quantize(Function):
    def __init__(self):
        # Initialize the custom autograd Function
        super(Quantize, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=4):
        '''
        Forward pass for generic n-bit symmetric quantization.

        :param ctx: context object for saving variables for backward computation
        :param input: input tensor to be quantized
        :param nbits: number of quantization bits (default=4)
        :return: quantized output tensor
        '''
        # Save input for backward pass
        ctx.save_for_backward(input)

        # Calculate quantization range based on nbits
        q_max = 2 ** (nbits - 1) - 1  # Maximum representable integer value
        q_min = -2 ** (nbits - 1)     # Minimum representable integer value

        # Calculate scale factor (s) to map float range to integer range
        s = (input.max() - input.min()) / (q_max - q_min)

        # Calculate zero point (z) to align minimum input value with q_min
        z = -(input.min() / s) + q_min

        # Quantize input by scaling, shifting, and rounding to nearest integer
        out = torch.round(input / s + z)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        '''
        Backward pass using Straight-Through Estimator (STE).
        Passes gradients unchanged through quantization.

        :param ctx: context with saved tensors from forward
        :param grad_outputs: gradients flowing from upper layers
        :return: gradient w.r.t input, None for nbits (since it's not a tensor)
        '''
        input = ctx.saved_tensors  # Retrieve saved input tensor
        grad_input = grad_outputs  # Gradient is passed through directly (STE)
        return grad_input, None

    
class Quantize_int8(Function):
    def __init__(self):
        # Initialize the custom autograd Function
        super(Quantize, self).__init__(self)

    @staticmethod
    def forward(ctx, input, nbits=8):
        '''
        Forward pass for 8-bit symmetric quantization.

        :param ctx: context object to save variables for backward computation
        :param input: input tensor to be quantized
        :param nbits: number of quantization bits (default=8)
        :return: quantized output tensor
        '''
        # Save input tensor for backward pass
        ctx.save_for_backward(input)

        # Calculate quantization integer range based on nbits
        q_max = 2 ** (nbits - 1) - 1  # Maximum representable value, e.g., 127 for 8-bit
        q_min = -2 ** (nbits - 1)     # Minimum representable value, e.g., -128 for 8-bit

        # Calculate scale factor (s) to map float range to integer range
        s = (input.max() - input.min()) / (q_max - q_min)

        # Calculate zero point (z) to align input.min() with q_min
        z = -(input.min() / s) + q_min

        # Quantize: scale input, shift by zero point, and round to nearest integer
        out = torch.round(input / s + z)

        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        '''
        Backward pass using Straight-Through Estimator (STE).
        Pass gradients directly through quantization operation.

        :param ctx: context with saved tensors from forward
        :param grad_outputs: gradients from upper layers
        :return: gradient w.r.t input, None for nbits (not a tensor)
        '''
        input = ctx.saved_tensors  # Retrieve saved input tensor
        grad_input = grad_outputs  # Pass gradients unchanged (STE)

        return grad_input, None


class TernaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        '''
        Forward pass for ternary quantization.

        Maps input values into {-1, 0, +1} based on their relative position within the input range.

        :param ctx: context object to save variables for backward computation
        :param input: input tensor to be quantized
        :return: quantized output tensor with values in {-1, 0, +1}
        '''
        # Save input tensor for backward pass
        ctx.save_for_backward(input)

        # Calculate max and min values of input tensor
        ctx_max, ctx_min = torch.max(input), torch.min(input)

        # Define quantization intervals:
        # Values below lower_interval -> -1
        # Values above higher_interval -> +1
        # Values in between -> 0
        lower_interval = ctx_min + (ctx_max - ctx_min) / 3
        higher_interval = ctx_max - (ctx_max - ctx_min) / 3

        # Initialize output tensor with ternary quantization
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

        :param ctx: context with saved tensors from forward
        :param grad_output: gradients flowing from upper layers
        :return: gradient w.r.t input
        '''
        input = ctx.saved_tensors  # Retrieve saved input tensor
        grad_input = grad_output   # Initialize gradient as received from upstream

        # Block gradients where input > +1 or input < -1
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
        # Convert parameters to tuple forms
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        # Initialize parent class (_ConvNd)
        super(TriConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # Note: weight parameter is inherited from _ConvNd
        # Optional reinitialization commented out

    def forward(self, input):
        '''
        Forward pass with ternary weight quantization.

        :param input: input tensor of shape (B, C_in, H, W)
        :return: output tensor after ternary quantized convolution
        '''
        bw = self.weight  # Get convolution weight
        ba = input        # Input tensor

        # Center weights by subtracting mean for better quantization stability
        bw = bw - bw.mean()

        # Apply ternary quantization: map weights to {-1, 0, +1}
        bw = TernaryQuantize().apply(bw)

        # Update self.weight.data with quantized weights (in-place)
        self.weight.data = bw

        # Optional input ternary quantization (commented out)
        # ba = TernaryQuantize().apply(ba)

        # Perform convolution operation based on padding mode
        if self.padding_mode == 'circular':
            # Calculate expanded padding for circular mode
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            # Apply padding then convolution
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            # Standard convolution
            return F.conv2d(ba, bw, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

def quantize_2bit(input_tensor, min_val=-1.5, max_val=1.5):
    '''
    Perform custom 2-bit quantization on input tensor.

    Maps input tensor values into discrete levels within specified min-max range.

    :param input_tensor: input tensor to be quantized
    :param min_val: float, minimum clamping value (default = -1.5)
    :param max_val: float, maximum clamping value (default = +1.5)
    :return: quantized tensor with values mapped to predefined levels
    '''

    # Clamp tensor values to [min_val, max_val] to avoid overflow
    clamped_tensor = torch.clamp(input_tensor, min=min_val, max=max_val)

    # Define scaling factor (s) and zero-point offset (z)
    # Here s=2.0 means quantization step is 0.5 (since 1/s=0.5), z=0.5 shifts before rounding
    s = 2.0
    z = 0.5

    # Quantize to nearest half-integer:
    # Multiply by s, subtract z to shift, round to nearest integer, then divide back by s
    quantized_values = torch.round(clamped_tensor * s - z) / s

    # Adjust quantized values to avoid exact values of 1, -1, 0, or -2
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
        Quantized 2D Convolution layer.

        Inherits from nn.Conv2d and applies quantization to both inputs and weights before convolution.
        '''
        # Initialize parent nn.Conv2d with standard arguments
        super(QuantizeConv2d, self).__init__(*args, **kwargs)

        # Optional: define quantization range (commented out)
        # self.qmin, self.qmax = -2 ** (8 - 1), 2 ** (8 - 1) - 1

    def forward(self, x):
        '''
        Forward pass with int8 quantization for both input and weight.

        :param x: input tensor of shape (B, C_in, H, W)
        :return: output tensor after quantized convolution
        '''

        # ====== INPUT QUANTIZATION ======
        # Apply 8-bit quantization to input tensor using Quantize_int8 function
        x_q = Quantize_int8.apply(x)

        # Alternative 2-bit quantization code (commented out):
        # x_q = torch.clamp(x, min=-1.5, max=1.5)
        # x_q = torch.round(x_q * 2) / 2

        # ====== WEIGHT QUANTIZATION ======
        weight = self.weight  # Get convolution weights

        # Apply 8-bit quantization to weight tensor using Quantize_int8 function
        weight_q = Quantize_int8.apply(weight)

        # Alternative 2-bit quantization code (commented out):
        # weight_q = torch.clamp(weight, min=-1.5, max=1.5)
        # weight_q = torch.round(weight_q * 2) / 2

        # ====== PERFORM QUANTIZED CONVOLUTION ======
        output = F.conv2d(x_q, weight_q, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        # Optional dequantization mapping back to float range (commented out):
        # s = (self.weight.max() - self.weight.min()) / (self.qmax - self.qmin)
        # output = s * (output - z)

        return output


class pointnet2_cls_ssg(nn.Module):
    def __init__(self, in_channels, nclasses):
        '''
        PointNet++ Classification Network with Single-Scale Grouping (SSG).

        :param in_channels: int, number of input feature channels (e.g. 3 for xyz coordinates)
        :param nclasses: int, number of output classes for classification
        '''
        super(pointnet2_cls_ssg, self).__init__()

        # ========= Set Abstraction Layer 1 =========
        # Samples 512 points, groups neighbors within radius 0.2, outputs feature dim 128
        self.pt_sa1 = PointNet_SA_Module(M=512, radius=0.2, K=32,
                                         in_channels=in_channels,
                                         mlp=[64, 64, 128],
                                         group_all=False)

        # ========= Set Abstraction Layer 2 =========
        # Samples another 512 points, processes combined features, outputs feature dim 256
        # Input channels: 131 = 128 (previous SA output) + 3 (xyz concatenation)
        self.pt_sa2 = PointNet_SA_Module(M=512, radius=0.2, K=32,
                                         in_channels=131,
                                         mlp=[128, 128, 256],
                                         group_all=False)

        # ========= Set Abstraction Layer 3 (Global) =========
        # Groups all remaining points to extract global features, outputs feature dim 1024
        # Input channels: 259 = 256 (previous SA output) + 3 (xyz concatenation)
        self.pt_sa3 = PointNet_SA_Module(M=None, radius=None, K=None,
                                         in_channels=259,
                                         mlp=[256, 512, 1024],
                                         group_all=True)

        # ========= Fully Connected Layers for Classification =========
        self.fc1 = nn.Linear(1024, 512, bias=False)   # FC layer: 1024 -> 512
        self.bn1 = nn.BatchNorm1d(512)                # BatchNorm for stability
        self.dropout1 = nn.Dropout(0.5)               # Dropout with 50% rate for regularization

        self.fc2 = nn.Linear(512, 256, bias=False)    # FC layer: 512 -> 256
        self.bn2 = nn.BatchNorm1d(256)                # BatchNorm
        self.dropout2 = nn.Dropout(0.5)               # Dropout

        self.cls = nn.Linear(256, nclasses)           # Final classifier layer: 256 -> nclasses

    def forward(self, xyz, points):
        '''
        Forward pass of PointNet++ SSG Classification Network.

        :param xyz: input coordinates tensor of shape (B, N, 3)
        :param points: input features tensor of shape (B, N, C)
        :return: logits tensor of shape (B, nclasses)
        '''
        batchsize = xyz.shape[0]

        # # Debug: print input shapes
        # print(xyz.shape)
        # print(points.shape)

        # Pass through Set Abstraction layers sequentially
        new_xyz, new_points = self.pt_sa1(xyz, points)       # SA1: local features
        new_xyz, new_points = self.pt_sa2(new_xyz, new_points)  # SA2: higher-level local features
        new_xyz, new_points = self.pt_sa3(new_xyz, new_points)  # SA3: global features

        # Flatten global feature vector
        net = new_points.view(batchsize, -1)

        # Pass through fully connected layers with BatchNorm, ReLU, and Dropout
        net = self.dropout1(F.relu(self.bn1(self.fc1(net))))
        net = self.dropout2(F.relu(self.bn2(self.fc2(net))))

        # Final classification output
        net = self.cls(net)

        return net


def read_off(file):
    '''
    Read an OFF (Object File Format) file to extract vertices and faces.

    :param file: opened OFF file object
    :return: verts (list of vertex coordinates), faces (list of face vertex indices)
    '''
    # Check if the file starts with 'OFF' header
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')

    # Read number of vertices, faces, and edges (edges count ignored here)
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])

    # Read vertex coordinates
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]

    # Read face indices, ignoring first index (which indicates number of vertices per face)
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]

    return verts, faces

class PointSampler(object):
    def __init__(self, output_size):
        '''
        Initialize the PointSampler.

        :param output_size: int, number of points to sample
        '''
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        '''
        Calculate area of a triangle given its three vertices.

        Uses Heron's formula.

        :param pt1: numpy array, first vertex (3,)
        :param pt2: numpy array, second vertex (3,)
        :param pt3: numpy array, third vertex (3,)
        :return: float, area of the triangle
        '''
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        '''
        Sample a random point within a triangle using barycentric coordinates.

        Reference: https://mathworld.wolfram.com/BarycentricCoordinates.html

        :param pt1: numpy array, first vertex (3,)
        :param pt2: numpy array, second vertex (3,)
        :param pt3: numpy array, third vertex (3,)
        :return: tuple, sampled point coordinates (x, y, z)
        '''
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        '''
        Sample self.output_size points uniformly from a mesh.

        :param mesh: tuple (verts, faces)
            verts: list of vertices
            faces: list of face indices
        :return: numpy array of shape (output_size, 3), sampled point cloud
        '''
        verts, faces = mesh
        verts = np.array(verts)

        # Calculate area of each face for weighted sampling
        areas = np.zeros((len(faces)))
        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        # Sample faces with probability proportional to their area
        sampled_faces = random.choices(faces,
                                       weights=areas,
                                       cum_weights=None,
                                       k=self.output_size)

        # Initialize sampled points array
        sampled_points = np.zeros((self.output_size, 3))

        # For each sampled face, sample a point within the triangle
        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points

class Normalize(object):
    def __call__(self, pointcloud):
        '''
        Normalize point cloud to zero mean and unit sphere.

        :param pointcloud: numpy array of shape (N, 3), input point cloud
        :return: normalized point cloud with mean at origin and max distance scaled to 1
        '''
        # Assert input is 2D (N, 3)
        assert len(pointcloud.shape) == 2

        # Subtract mean along each dimension to center the point cloud at the origin
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)

        # Scale point cloud so that its furthest point from origin is at distance 1
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        '''
        Apply random rotation to the point cloud around Z-axis.

        :param pointcloud: numpy array of shape (N, 3), input point cloud
        :return: rotated point cloud
        '''
        # Assert input is 2D (N, 3)
        assert len(pointcloud.shape) == 2

        # Generate a random rotation angle theta in [0, 2pi]
        theta = random.random() * 2. * math.pi

        # Define rotation matrix around Z-axis
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        # Apply rotation: (3,3) x (3,N) -> (3,N), then transpose back to (N,3)
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T

        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        '''
        Add Gaussian noise to each point in the point cloud.

        :param pointcloud: numpy array of shape (N, 3), input point cloud
        :return: noisy point cloud with added Gaussian noise
        '''
        # Assert input is 2D (N, 3)
        assert len(pointcloud.shape) == 2

        # Generate Gaussian noise with mean=0 and std=0.02, same shape as pointcloud
        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        # Add noise to point cloud
        noisy_pointcloud = pointcloud + noise

        return noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        '''
        Convert numpy array point cloud to PyTorch tensor.

        :param pointcloud: numpy array of shape (N, 3)
        :return: torch tensor of shape (N, 3)
        '''
        # Assert input is 2D (N, 3)
        assert len(pointcloud.shape) == 2

        # Convert numpy array to torch tensor
        return torch.from_numpy(pointcloud)


def default_transforms():
    '''
    Compose default transforms for point cloud preprocessing.

    Includes:
    - PointSampler: samples 1024 points uniformly from mesh
    - Normalize: centers point cloud at origin and scales to unit sphere
    - ToTensor: converts numpy array to PyTorch tensor

    :return: torchvision.transforms.Compose object
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


import kagglehub
from pathlib import Path
from torch.utils.data import DataLoader

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


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Optional debug setting to make CUDA operations synchronous for easier debugging of errors.

# Set device to GPU if available; otherwise fallback to CPU.
# Specifically selects cuda:1 (second GPU) if multiple GPUs are available.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# Instantiate the PointNet++ classification model with Single-Scale Grouping (SSG).
# Input channels = 3 (x, y, z), number of classes = 10 (e.g. ModelNet10).
model = pointnet2_cls_ssg(in_channels=3, nclasses=10)

# Move model to selected device (GPU or CPU).
model.to(device)


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

def train_model(model, epoch, traindata):
    '''
    Train the model for a specified number of epochs, with pruning and evaluation after each epoch.

    :param model: PyTorch model to train
    :param epoch: int, number of epochs to train
    :param traindata: DataLoader, training dataset loader (currently unused; uses train_loader globally)
    '''
    for epoch in range(epoch):  # Loop over epochs
        train_loss_all = []
        train_accur_all = []

        train_loss = 0  # Initialize total training loss
        train_num = 0.0  # Initialize total sample count
        train_accuracy = 0.0  # Initialize total correct prediction count

        model.train()  # Set model to training mode

        train_bar = tqdm(train_loader)  # Progress bar for training loop

        for step, data in enumerate(train_bar):  # Loop over batches
            # Get input point cloud and target labels
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)

            optimizer.zero_grad()  # Zero gradients

            # Split input into xyz coordinates and additional features (if any)
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]

            # Forward pass: obtain model outputs
            outputs = model(xyz.to(device), points.to(device))

            # Compute loss between model outputs and ground truth targets
            loss1 = loss(outputs, target.to(device))

            # Convert outputs to predicted class indices
            outputs = torch.argmax(outputs, 1)

            # Backward pass: compute gradients
            loss1.backward()

            # Gradient masking: zero out gradients for weights that are already pruned (equal to zero)
            for name, param in model.named_parameters():
                if 'weight' and 'Conv' in name:
                    if param.requires_grad:
                        if param.grad is not None:
                            grad_mask = param.data != 0  # Mask: True for non-zero weights
                            param.grad *= grad_mask     # Zero gradients for pruned weights

            optimizer.step()  # Update parameters using optimizer

            # Accumulate training loss (multiplied by batch size)
            train_loss += abs(loss1.item()) * inputs.size(0)

            # Calculate number of correct predictions
            accuracy = torch.sum(outputs == target.to(device))
            train_accuracy = train_accuracy + accuracy

            train_num += xyz.shape[0]  # Update total sample count

        # Print epoch training statistics
        print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(
            epoch + 1, train_loss / train_num, train_accuracy / train_num))

        # Append epoch loss and accuracy to lists for logging or plotting
        train_loss_all.append(train_loss / train_num)
        train_accur_all.append(train_accuracy.double().item() / train_num)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, f'data/{time}/weights_epoch_{epoch + 1}_before_pruning.pth')

        # ========== Save model weights before pruning ==========
        torch.save(model, save_path)

        # ========== Count number of zero weights before pruning ==========
        count_zero_weights_in_conv_layers_beforep(model)

        # ========== Test model before pruning ==========
        test_beforep_model(model, test_loader)

        # ========== Prune the model using quantization-based pruning ==========
        quantize_prune_model(model, threshold_c=0.45, threshold_b=0.01)

        # ========== Save model weights after pruning ==========
        pruned_weight_path = os.path.join(current_dir, f'data/{time}/weights_epoch_{epoch + 1}_after_pruning.pth')
        torch.save(model.state_dict(), pruned_weight_path)
        print(f"Pruned model weights saved to {pruned_weight_path}")

        # ========== Count number of zero weights after pruning ==========
        count_zero_weights_in_conv_layers_afterp(model)

def count_zero_weights_in_conv_layers_beforep(model):
    '''
    Count and log the number of zero weights in convolution layers before pruning.

    :param model: PyTorch model to analyze
    '''
    zero_count = 0    # Counter for zero weights
    total_count = 0   # Counter for total weights

    # Iterate through all model parameters by name
    for name, param in model.named_parameters():
        # Uncomment for debugging:
        # print(name)
        # print(param.shape)

        # Select parameters that are weights of Conv layers within pt_sa1 module
        if 'weight' in name and 'Conv' and 'pt_sa1' in name:
            # Count number of zero elements in current parameter tensor
            zero_count += torch.sum(param == 0).item()

            # Count total number of elements in current parameter tensor
            total_count += param.numel()

    # Calculate percentage of zero weights
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0

    # Print summary statistics
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/before_pruning_info.txt')

    # Write statistics to file for logging
    with open(save_path, 'a') as f:
        f.write(f'Total parameters: {total_count}\n')
        f.write(f'Zero parameters before pruning: {zero_count}\n')
        f.write(f'pruning rate: {zero_percentage}\n')


def count_zero_weights_in_conv_layers_afterp(model):
    '''
    Count and log the number of zero weights in convolution layers after pruning.

    :param model: PyTorch model to analyze
    '''
    zero_count = 0    # Counter for zero weights
    total_count = 0   # Counter for total weights

    # Iterate through all model parameters by name
    for name, param in model.named_parameters():
        # Select parameters that are weights of Conv layers within pt_sa1 module
        if 'weight' in name and 'Conv' and 'pt_sa1' in name:
            # Count number of zero elements in current parameter tensor
            zero_count += torch.sum(param == 0).item()

            # Count total number of elements in current parameter tensor
            total_count += param.numel()

    # Calculate percentage of zero weights
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0

    # Print summary statistics
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/after_pruning_info.txt')

    # Write statistics to file for logging
    with open(save_path, 'a') as f:
        f.write(f'Total parameters: {total_count}\n')
        f.write(f'Zero parameters after pruning: {zero_count}\n')
        f.write(f'pruning rate: {zero_percentage}\n')

def test_beforep_model(model, testdata):
    '''
    Evaluate the model on the test dataset before pruning.

    :param model: PyTorch model to evaluate
    :param testdata: DataLoader for test dataset
    '''
    # Initialize lists to store per-batch test loss and accuracy (optional use)
    test_loss_all = []
    test_accur_all = []

    test_loss = 0       # Initialize total test loss
    test_accuracy = 0.0 # Initialize total number of correct predictions
    test_num = 0        # Initialize total number of samples

    model.eval()  # Set model to evaluation mode (disables dropout, BN update)

    # Disable gradient computation for evaluation to save memory and computation
    with torch.no_grad():
        test_bar = tqdm(testdata)  # Progress bar for test loop

        for data in test_bar:
            # Extract inputs (point cloud) and target labels from batch
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)

            # Split inputs into xyz coordinates and additional features (if any)
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]

            # Forward pass: obtain model outputs
            outputs = model(xyz.to(device), points.to(device))

            # Calculate cross-entropy loss
            loss2 = loss(outputs, target.to(device))

            # Convert outputs to predicted class indices
            outputs = torch.argmax(outputs, 1)

            # Accumulate total test loss (multiplied by batch size)
            test_loss = test_loss + abs(loss2.item()) * inputs.size(0)

            # Calculate number of correct predictions in this batch
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy

            # Update total number of samples seen so far
            test_num += xyz.shape[0]

            # Calculate running test accuracy
            accuracy_before_pruning = test_accuracy / test_num

    # Print final test loss and accuracy after evaluation
    print("test-Loss：{} , test-accuracy-before-pruning：{}".format(
        test_loss / test_num, accuracy_before_pruning))

    # Append results to lists (optional logging use)
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/test-accuracy-before-pruning.txt')

    # Write evaluation results to file for logging
    with open(save_path, 'a') as f:
        f.write(f'Accuracy on test set before pruning: {accuracy_before_pruning}\n')


def test_afterp_model(model, testdata):
    '''
    Evaluate the model on the test dataset after pruning.

    :param model: PyTorch model to evaluate
    :param testdata: DataLoader for test dataset
    '''
    # Initialize lists to store per-batch test loss and accuracy (optional)
    test_loss_all = []
    test_accur_all = []

    test_loss = 0       # Initialize total test loss
    test_accuracy = 0.0 # Initialize total number of correct predictions
    test_num = 0        # Initialize total number of samples

    model.eval()  # Set model to evaluation mode (disables dropout, BN update)

    # Disable gradient computation for evaluation to save memory and computation
    with torch.no_grad():
        test_bar = tqdm(testdata)  # Progress bar for test loop

        for data in test_bar:
            # Extract inputs (point cloud) and target labels from batch
            inputs, target = data['pointcloud'].to(device).float(), data['category'].to(device)

            # Split inputs into xyz coordinates and additional features (if any)
            xyz, points = inputs[:, :, :3], inputs[:, :, 3:]

            # Forward pass: obtain model outputs
            outputs = model(xyz.to(device), points.to(device))

            # Calculate cross-entropy loss
            loss2 = loss(outputs, target.to(device))

            # Convert outputs to predicted class indices
            outputs = torch.argmax(outputs, 1)

            # Accumulate total test loss (multiplied by batch size)
            test_loss = test_loss + abs(loss2.item()) * inputs.size(0)

            # Calculate number of correct predictions in this batch
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy

            # Update total number of samples seen so far
            test_num += xyz.shape[0]

            # Calculate running test accuracy
            accuracy_after_pruning = test_accuracy / test_num

    # Print final test loss and accuracy after evaluation
    print("test-Loss：{} , test-accuracy-after-pruning：{}".format(
        test_loss / test_num, accuracy_after_pruning))

    # Append results to lists (optional logging use)
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/test-accuracy-after-pruning.txt')
    # Write evaluation results to file for logging
    with open(save_path, 'a') as f:
        # f.write(f'Accuracy on test set before pruning: {accuracy_before_pruning}\n')
        f.write(f'test_afterp_model: {accuracy_after_pruning}\n')

# Save model parameters with quantization-based pruning
def quantize_prune_model(model, threshold_c, threshold_b):
    '''
    Prune similar filters in convolution layers using quantization-based similarity.

    :param model: PyTorch model to prune
    :param threshold_c: float, pruning threshold based on filter occurrence frequency
    :param threshold_b: float, pruning threshold based on distance variance
    '''
    # Iterate over all model state_dict items (parameter tensors)
    for key, value in model.state_dict().items():
        # Select only convolution layer weights within 'pt_sa1' module
        if 'Conv' in key and 'weight' and 'pt_sa1' in key:
            print(key)

            weights = value  # Get current weight tensor

            close_filters = []  # List to store filters identified for pruning (not used further here)
            num_filters = weights.size(0)  # Number of filters in current layer
            hamming_distances = []  # Store pairwise filter distances

            # Calculate pairwise Euclidean distances between filters
            for i in range(num_filters):
                if torch.sum(weights[i]) != 0:  # Skip zero filters
                    for j in range(i + 1, num_filters):
                        if torch.sum(weights[j]) != 0:
                            hamming_distance_ij = euclidean_distance(weights[i], weights[j])
                            hamming_distances.append((i, j, hamming_distance_ij))

            # Extract all computed distances for statistical analysis
            distances = [distance.cpu() for _, _, distance in hamming_distances]

            # Calculate mean and variance of distances
            ean_distance = np.mean(distances)
            variance_distance = np.var(distances)

            filters_to_prune = []  # Initialize list of filters to prune

            # Identify filter pairs with distance below threshold
            for i, j, distance in hamming_distances:
                if distance < ean_distance - threshold_b * variance_distance:
                    filters_to_prune.append(i)
                    filters_to_prune.append(j)

            # Count how frequently each filter appears in close pairs
            filter_counts = Counter(filters_to_prune)
            print(filter_counts)

            # Finalize filters to prune based on occurrence frequency threshold_c
            filters_to_prune_final = [filter_num for filter_num, count in filter_counts.items()
                                      if count > threshold_c * (num_filters - 1)]

            # Prune selected filters by setting their weights to zero
            for filter_num in filters_to_prune_final:
                weights[filter_num] = 0

            # Update model state_dict with pruned weights
            model.state_dict()[key].copy_(weights)  # .copy_() ensures in-place update


# Set number of training epochs
epoch = 100  # Number of training iterations (epochs) to run

# Set learning rate for optimizer
learning = 0.001

# Initialize Adam optimizer with model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning)

# Define loss function as Cross Entropy Loss
# Suitable for multi-class classification tasks
loss = nn.CrossEntropyLoss()

# Start training the model using the defined training loop
train_model(model, epoch=epoch, traindata=train_loader)
