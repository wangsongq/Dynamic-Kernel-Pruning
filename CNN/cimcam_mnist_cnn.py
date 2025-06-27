# ===========================================
# Import necessary modules
# ===========================================

from torchsummary import summary
import torch
import torchvision
import torchvision.models
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import math
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.autograd import Function
from collections import Counter
import os

import time
time = time.strftime('%Y%m%d_%H%M%S')
# Get current script directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct directory path: ./data/{time}/
save_dir = os.path.join(current_dir, f'data/{time}')
# Check if the directory exists; if not, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ===========================================
# Define a custom autograd function called mask_dnnFunction
# This function implements a masked activation operation with a customized backward gradient.
# ===========================================

class mask_dnnFunction(Function):
    @staticmethod
    def forward(ctx, input_, weight):
        """
        Forward pass of the masked DNN function.
        Args:
            ctx: context to save tensors for backward computation.
            input_: input tensor to be masked.
            weight: threshold tensor for masking.
        Returns:
            out: masked output tensor.
        """
        # Initialize a mask tensor with zeros, same shape as input
        mask = torch.zeros_like(input_)
        
        # Update the mask: if input >= weight, set to 1.0; else set to 0.0
        # torch.where returns elements based on condition
        mask = torch.where(input_.ge(weight), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
        
        # Save input_, weight, mask for use in backward pass
        ctx.save_for_backward(input_, weight, mask)
        
        # Apply mask to input by element-wise multiplication
        out = input_.mul(mask)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradients w.r.t input and weight.
        Args:
            ctx: context with saved tensors from forward.
            grad_output: gradient of loss w.r.t output.
        Returns:
            grad_input: gradient of loss w.r.t input_.
            grad_weight: gradient of loss w.r.t weight.
        """
        # Retrieve saved tensors from context
        input_, weight, mask = ctx.saved_variables
        
        # Initialize gradients as None by default
        grad_input = grad_weight = None

        # Compute the difference between input and weight
        c_i = input_ - weight
        
        # Initialize auxiliary gradient tensor a_i with zeros
        a_i = torch.zeros_like(c_i)
        
        # Define piecewise gradient scaling rules for backpropagation
        # Each line applies conditions to c_i to compute a_i based on specific ranges:
        
        # Case 1: c_i in [0.4,1.0) => a_i = 0.4
        a_i = torch.where(c_i.lt(1.0) & c_i.ge(0.4), torch.tensor(0.4).to(device), a_i)
        
        # Case 2: c_i in [0.0,0.4) => a_i = -4*c_i + 2.0
        a_i = torch.where(c_i.lt(0.4) & c_i.ge(0.0), -4.0 * c_i + 2.0, a_i)
        
        # Case 3: c_i in (-0.4,0.0) => a_i = 4*c_i + 2.0
        a_i = torch.where(c_i.lt(0.0) & c_i.gt(-0.4), 4.0 * c_i + 2.0, a_i)
        
        # Case 4: c_i in [-1.0,-0.4] => a_i = 0.4
        a_i = torch.where(c_i.le(-0.4) & c_i.gt(-1.0), torch.tensor(0.4).to(device), a_i)

        # Compute gradient w.r.t input if required
        if ctx.needs_input_grad[0]:
            # Gradient formula combines the mask and scaled input
            grad_input_1 = mask + a_i.mul(input_)
            grad_input = grad_output.mul(grad_input_1)

        # Compute gradient w.r.t weight if required
        if ctx.needs_input_grad[1]:
            # Gradient formula with negative scaled input
            grad_weight_1 = -a_i.mul(input_)
            grad_weight = grad_output.mul(grad_weight_1)

        # Return gradients w.r.t input_ and weight
        return grad_input, grad_weight

# ===========================================
# Define a custom Linear (fully connected) layer with masking capability.
# It inherits from torch.nn.Linear.
# ===========================================
class maskLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        """
        Initializes the maskLinear layer.

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            bias (bool): If set to False, the layer will not learn an additive bias.
            binary_act (bool): Currently unused, but can be used to indicate binary activation usage.

        Additional:
            self.threshold: A learnable parameter tensor with shape [out_features, in_features],
                            initialized with random values. This acts as the per-weight threshold
                            for the mask operation in forward pass.
        """
        super(maskLinear, self).__init__(in_features, out_features, bias=bias)

        # Initialize output cache (if needed for analysis or debug)
        self.output_ = None

        # Define learnable thresholds for each weight connection
        self.threshold = torch.nn.Parameter(torch.rand(out_features, in_features), requires_grad=True)

        # Optionally, reset parameters using custom initialization
        # self.reset_parameters()

    # The default reset_parameters of torch.nn.Linear is already called via super().__init__()
    # You can override it if you want custom initialization.
    # def reset_parameters(self):
    #     torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass of maskLinear.

        Steps:
            1. Get current weight tensor.
            2. Apply mask_dnnFunction to perform masked thresholding on the weights.
            3. Perform standard linear transformation (input * weight^T + bias).
            4. Save output for potential debugging or visualization.
        
        Args:
            x: Input tensor of shape [batch_size, in_features].

        Returns:
            output: Result of linear transformation after masked weight application.
        """
        # Retrieve weight and input
        tw = self.weight
        ta = x

        # Apply custom masking function (with gradient support)
        tw = mask_dnnFunction.apply(tw, self.threshold)

        # Perform linear transformation using masked weights
        output = F.linear(ta, tw, self.bias)

        # Save output for external access if needed
        self.output_ = output

        # Return computed output
        return output

    # Note:
    # Backward function is handled by autograd due to usage of mask_dnnFunction.apply.
    # You do not need to define backward here unless overriding torch.nn.Linear gradients.

# ===========================================
# Define a custom Conv2d layer with masking capability.
# This layer inherits from torch.nn.modules.conv._ConvNd,
# which is the base class for all convolution layers in PyTorch.
# ===========================================
class maskConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        """
        Initializes the maskConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (filters).
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple): Stride of the convolution.
            padding (int or tuple): Zero-padding added to both sides of input.
            dilation (int or tuple): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): If True, adds a learnable bias to the output.
            padding_mode (string): 'zeros', 'reflect', 'replicate' or 'circular'.

        Additional:
            self.threshold: A learnable parameter tensor with shape [out_channels, in_channels/groups, kernel_height, kernel_width],
                            initialized with random values. This acts as per-weight threshold for masking.
        """
        # Convert parameters to tuple if needed (e.g. kernel_size=3 -> (3,3))
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        # Call parent constructor to initialize standard convolution parameters
        # Note: _ConvNd requires explicit flags for transposed conv (False here) and output padding (_pair(0))
        super(maskConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # Define learnable thresholds for each convolution kernel weight
        # Shape: [out_channels, in_channels/groups, kernel_height, kernel_width]
        self.threshold = torch.nn.Parameter(torch.rand(out_channels, in_channels // groups, *kernel_size),
                                            requires_grad=True)

    def forward(self, x):
        """
        Forward pass of maskConv2d.

        Steps:
            1. Retrieve current weights and input.
            2. Apply mask_dnnFunction to weights using learnable thresholds.
            3. Perform standard convolution operation using masked weights.
            4. Supports circular padding mode if specified.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            Output tensor after convolution with masked weights.
        """
        # Retrieve weight tensor (before masking)
        bw = self.weight

        # Retrieve input tensor
        ba = x

        # Apply custom mask function to weights (with custom backward gradient)
        bw = mask_dnnFunction.apply(bw, self.threshold)

        # Check if circular padding is specified
        if self.padding_mode == 'circular':
            # Compute expanded padding values for all sides
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            # Apply circular padding and perform convolution
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        # Perform standard convolution with masked weights
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# ===========================================
# Define a custom Conv2d module with filter pruning logic.
# This layer extends nn.Conv2d and integrates pruning based on Hamming distance between filters.
# ===========================================
class PrunedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, threshold_c=0.5, threshold_b=3, **kwargs):
        """
        Initializes the PrunedConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (filters).
            kernel_size (int or tuple): Size of the convolution kernel.
            threshold_c (float): Frequency threshold to decide filter pruning.
            threshold_b (float): Variance scaling factor for distance-based pruning criterion.
            **kwargs: Other keyword arguments for nn.Conv2d (e.g., stride, padding).
        """
        # Call the parent nn.Conv2d constructor
        super(PrunedConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

        # Store pruning thresholds as class attributes
        self.threshold_c = threshold_c
        self.threshold_b = threshold_b

    def forward(self, input):
        """
        Forward pass of PrunedConv2d.

        Steps:
            1. Binarize weights and input.
            2. Compute Hamming distances between filters to identify redundant filters.
            3. Prune (zero out) redundant filters based on frequency and variance-adjusted thresholds.
            4. Perform standard convolution using the potentially pruned weights.

        Args:
            input: Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            Output tensor after convolution using pruned filters.
        """
        # Retrieve current weight matrix
        weights = self.weight.data

        # Binarize weights: positive values -> 1.0; non-positive -> 0.0
        binary_weights = torch.where(weights > torch.tensor(0, device=device),
                                     torch.tensor(1.0, device=device),
                                     torch.tensor(0.0, device=device))

        # Binarize input similarly: positive -> 1.0; non-positive -> 0.0
        binary_input = torch.where(input > torch.tensor(0, device=device),
                                   torch.tensor(1.0, device=device),
                                   torch.tensor(0.0, device=device))

        # Number of convolution filters in this layer
        num_filters = weights.size(0)

        # Initialize containers to store filters marked for pruning and hamming distances
        filters_to_prune = set()
        hamming_distances = []

        # Compute pairwise Hamming distances between filters
        for filter_1 in range(num_filters):
            if torch.sum(self.weight[filter_1]) != 0:
                for filter_2 in range(filter_1 + 1, num_filters):
                    if torch.sum(self.weight[filter_2]) != 0:
                        # Calculate Hamming distance between two binary filters
                        hamming_distance_12 = calculate_hamming_distance(binary_weights[filter_1],
                                                                         binary_weights[filter_2])
                        hamming_distances.append((filter_1, filter_2, hamming_distance_12))

        # Extract only the distance values
        distances = [distance for _, _, distance in hamming_distances]

        # Compute mean and variance of the distances
        ean_distance = np.mean(distances)
        variance_distance = np.var(distances)

        # Identify filter pairs with distance below (mean - threshold_b * variance)
        for i, j, distance in hamming_distances:
            if distance < ean_distance - self.threshold_b * variance_distance:
                filters_to_prune.add(i)
                filters_to_prune.add(j)

        # Count frequency of filters in the pruning set using Counter
        filter_counts = Counter(filters_to_prune)

        # Select filters to prune based on frequency exceeding threshold_c * (num_filters - 1)
        filters_to_prune_final = [filter_num for filter_num, count in filter_counts.items()
                                  if count > self.threshold_c * (num_filters - 1)]

        # Perform pruning by zeroing out the weights of selected filters
        for filter_num in filters_to_prune_final:
            self.weight.data[filter_num].zero_()

        # Perform standard convolution with (potentially pruned) weights
        return nn.functional.conv2d(binary_input, self.weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

# ===========================================
# Define a function to calculate the Hamming distance between two tensors.
# ===========================================
def calculate_hamming_distance(weight1, weight2):
    """
    Calculates the Hamming distance between two tensors.

    Steps:
        1. Flatten both input tensors into 1D vectors.
        2. Compare each element to count the number of positions with different values.
    
    Args:
        weight1: First input tensor (e.g. a convolution filter tensor).
        weight2: Second input tensor (same shape as weight1).

    Returns:
        hamming_distance (int): Number of positions where weight1 and weight2 differ.
    """
    # Flatten both tensors to 1D vectors for element-wise comparison
    weights1 = weight1.view(-1)
    weights2 = weight2.view(-1)

    # Calculate Hamming distance as the count of unequal elements
    hamming_distance = (weights1 != weights2).sum()

    return hamming_distance

# ===========================================
# Define a function to calculate the Euclidean distance between two tensors.
# ===========================================
def euclidean_distance(A, B):
    """
    Calculates the Euclidean distance between two tensors.

    Steps:
        1. Compute the element-wise difference.
        2. Calculate the L2 norm of the difference vector.

    Args:
        A: First input tensor.
        B: Second input tensor (same shape as A).

    Returns:
        distance (float): Euclidean (L2) distance between A and B.
    """
    # Compute the L2 norm (Euclidean distance) between tensors A and B
    distance = torch.norm(A - B)

    return distance

# ===========================================
# Define a custom Conv2d module with binary weight and input functionality.
# ===========================================
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """
        Initializes the BinaryConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (filters).
            kernel_size (int or tuple): Size of the convolution kernel.
            **kwargs: Other keyword arguments for nn.Conv2d (e.g., stride, padding).
        """
        # Call the parent nn.Conv2d constructor to initialize standard parameters
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input):
        """
        Forward pass of BinaryConv2d.

        Steps:
            1. Retrieve weight matrix.
            2. Binarize weights to +1 / -1 based on sign.
            3. Binarize input to +1 / -1 based on sign.
            4. Perform standard convolution using binarized weights and input.

        Args:
            input: Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            Output tensor after binary convolution.
        """
        # Retrieve current weight tensor
        weights = self.weight.data

        # Binarize weights: positive values -> +1.0; non-positive -> -1.0
        binary_weights = torch.where(weights > torch.tensor(0, device=device),
                                     torch.tensor(1.0, device=device),
                                     torch.tensor(-1.0, device=device))

        # Binarize input similarly: positive -> +1.0; non-positive -> -1.0
        binary_input = torch.where(input > torch.tensor(0, device=device),
                                   torch.tensor(1.0, device=device),
                                   torch.tensor(-1.0, device=device))

        # Perform standard 2D convolution with binary weights and inputs
        return nn.functional.conv2d(binary_input, binary_weights, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)

# ===========================================
# Define a custom autograd Function for binary quantization with straight-through estimator (STE).
# ===========================================
class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of BinaryQuantize.

        Steps:
            1. Save input tensor for backward computation.
            2. Return sign of input tensor (+1 for positive, -1 for negative, 0 stays 0).

        Args:
            ctx: context to save tensors for backward pass.
            input: Input tensor to be quantized.

        Returns:
            out: Binarized tensor with sign function.
        """
        # Save input tensor to context for backward use
        ctx.save_for_backward(input)

        # Apply sign function: +1 if input>0, -1 if input<0, 0 if input==0
        out = torch.sign(input)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of BinaryQuantize with straight-through estimator.

        Steps:
            1. Retrieve saved input tensor.
            2. Pass through gradients for inputs in [-1,1]; set gradients to 0 outside this range.

        Args:
            ctx: context with saved tensors from forward.
            grad_output: Gradient of loss w.r.t output.

        Returns:
            grad_input: Gradient of loss w.r.t input.
            None: No gradient for other arguments.
        """
        # Retrieve saved input tensor
        input = ctx.saved_tensors

        # Initialize grad_input as grad_output (STE)
        grad_input = grad_output

        # Zero gradients where input > +1
        grad_input[input[0].gt(1)] = 0

        # Zero gradients where input < -1
        grad_input[input[0].lt(-1)] = 0

        # Return gradients w.r.t input and None for the second argument (no other inputs)
        return grad_input, None

# ===========================================
# Define a binary linear layer with optional binary activation.
# ===========================================
class BiLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=False):
        """
        Initializes the BiLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If True, includes a learnable bias.
            binary_act (bool): If True, input activations will be binarized.
        """
        # Call parent nn.Linear constructor
        super(BiLinear, self).__init__(in_features, out_features, bias=True)

        # Store binary activation flag
        self.binary_act = binary_act

        # Initialize output cache (useful for debugging or visualization)
        self.output_ = None

    def forward(self, input):
        """
        Forward pass of BiLinear.

        Steps:
            1. Retrieve weights and input.
            2. Apply binary quantization to weights.
            3. If binary_act is True, binarize input activations.
            4. Perform standard linear transformation.
            5. Cache and return output.

        Args:
            input: Input tensor of shape [batch_size, in_features].

        Returns:
            output: Result of linear transformation with binary weights and optionally binary input.
        """
        # Retrieve weight and input tensors
        bw = self.weight
        ba = input

        # Apply binary quantization to weights
        bw = BinaryQuantize().apply(bw)

        # Optionally binarize input activations if binary_act flag is True
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)

        # Perform linear transformation
        output = F.linear(ba, bw, self.bias)

        # Cache output for external access if needed
        self.output_ = output

        # Return computed output
        return output

# ===========================================
# Define a binary convolution layer with weight binarization and optional mean-centering.
# ===========================================
class BiConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        """
        Initializes the BiConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (filters).
            kernel_size (int or tuple): Size of convolution kernel.
            stride (int or tuple): Stride of convolution.
            padding (int or tuple): Padding added to input.
            dilation (int or tuple): Spacing between kernel elements.
            groups (int): Number of blocked connections from input to output channels.
            bias (bool): If True, includes learnable bias.
            padding_mode (str): Padding mode ('zeros', 'reflect', 'replicate', 'circular').
        """
        # Call parent nn.Conv2d constructor
        super(BiConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):
        """
        Forward pass of BiConv2d.

        Steps:
            1. Retrieve weight and input tensors.
            2. Subtract mean from weights for mean-centering.
            3. Apply binary quantization to weights.
            4. Perform convolution (supports circular padding if specified).

        Args:
            input: Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            Output tensor after binary convolution.
        """
        # Retrieve weight and input tensors
        bw = self.weight
        ba = input

        # Mean-centering of weights (improves binarization performance)
        bw = bw - bw.mean()

        # Apply binary quantization to weights
        bw = BinaryQuantize().apply(bw)

        # Uncomment below if ternary quantization of input is desired
        # ba = TernaryQuantize().apply(ba)

        # Check if padding mode is circular
        if self.padding_mode == 'circular':
            # Compute expanded padding
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)

            # Apply circular padding and perform convolution
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)

        # Perform standard convolution
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# ===========================================
# Define a fully connected (linear) layer with binary weights and binary input.
# ===========================================
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the BinaryLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
        """
        super(BinaryLinear, self).__init__()

        # Define standard linear layer as submodule
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input):
        """
        Forward pass of BinaryLinear.

        Steps:
            1. Retrieve weight matrix.
            2. Binarize weights to +1/-1.
            3. Binarize input to +1/-1.
            4. Perform standard linear transformation using binarized weights and inputs.

        Args:
            input: Input tensor of shape [batch_size, in_features].

        Returns:
            output: Output tensor after binary linear transformation.
        """
        # Retrieve weights of the linear layer
        weights = self.fc.weight.data

        # Binarize weights: positive -> +1.0; non-positive -> -1.0
        binary_weights = torch.where(weights > 0,
                                     torch.tensor(1.0, device=weights.device),
                                     torch.tensor(-1.0, device=weights.device))

        # Binarize input: positive -> +1.0; non-positive -> -1.0
        binary_input = torch.where(input > 0,
                                   torch.tensor(1.0, device=input.device),
                                   torch.tensor(-1.0, device=input.device))

        # Perform linear transformation with binary weights and inputs
        output = F.linear(binary_input, binary_weights, self.fc.bias)

        return output

# ===========================================
# Define a VGG16-like convolutional neural network using binary convolution layers.
# ===========================================
class VGG16(nn.Module):
    def __init__(self):
        """
        Initializes the VGG16 model architecture.

        Architecture:
            - 3 binary convolutional layers
            - 3 fully connected layers
            - Uses ReLU activations and max pooling between convolutional layers.
        """
        super(VGG16, self).__init__()

        # First convolutional layer:
        # Input channels = 1 (grayscale), output channels = 32, kernel size = 3x3, padding=1 to preserve spatial size.
        self.conv1 = BinaryConv2d(1, 32, kernel_size=3, padding=1)  # Output: 32@28x28

        # Second convolutional layer:
        # Input channels = 32, output channels = 64, kernel size = 3x3, padding=1.
        self.conv2 = BinaryConv2d(32, 64, kernel_size=3, padding=1)  # Output: 64@14x14

        # Third convolutional layer:
        # Input channels = 64, output channels = 16, kernel size = 3x3, padding=1.
        self.conv3 = BinaryConv2d(64, 16, kernel_size=3, padding=1)  # Output: 16@7x7

        # Fully connected layers:
        # First FC layer: input features = 16*7*7 (flattened feature map), output = 256
        self.fc1 = nn.Linear(16 * 7 * 7, 256)

        # Second FC layer: input = 256, output = 128
        self.fc2 = nn.Linear(256, 128)

        # Third FC layer (classification output): input = 128, output = 10 classes (e.g. MNIST)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass of VGG16.

        Steps:
            1. Pass through first binary convolution -> ReLU -> MaxPool2d (downsample by 2).
            2. Pass through second binary convolution -> ReLU -> MaxPool2d.
            3. Pass through third binary convolution -> ReLU.
            4. Flatten feature map.
            5. Pass through fully connected layers with ReLU activations.
            6. Output classification logits.

        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28].

        Returns:
            x: Output logits tensor of shape [batch_size, 10].
        """
        # First convolution block: Conv -> ReLU -> MaxPool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Downsample to 14x14

        # Second convolution block: Conv -> ReLU -> MaxPool
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Downsample to 7x7

        # Third convolution block: Conv -> ReLU (no pooling)
        x = F.relu(self.conv3(x))

        # Flatten the feature map for fully connected layers
        x = x.view(-1, 16 * 7 * 7)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Final output layer (logits)
        x = self.fc3(x)

        return x

def train_model(model, epoch, traindata):
    """
    Trains the model with integrated pruning per epoch.

    Args:
        model: PyTorch model to train.
        epoch: Number of training epochs.
        traindata: DataLoader for training dataset.

    Notes:
        - This function performs training, evaluates pre-pruning performance, prunes filters, 
          saves model weights before and after pruning, and reports sparsity statistics.
    """
    for epoch in range(epoch):  # Loop over epochs
        train_loss_all = []  # List to store loss values for plotting (if needed)
        train_accur_all = []  # List to store accuracy values

        train_loss = 0  # Initialize cumulative training loss
        train_num = 0.0  # Initialize total number of samples
        train_accuracy = 0.0  # Initialize cumulative training accuracy

        model.train()  # Set model to training mode

        train_bar = tqdm(traindata)  # Initialize progress bar for training data

        for step, data in enumerate(train_bar):  # Iterate through batches
            img, target = data  # Unpack batch into images and labels

            optimizer.zero_grad()  # Clear previous gradients

            outputs = model(img.to(device))  # Forward pass with input moved to device

            loss1 = loss(outputs, target.to(device))  # Compute loss between output and target

            outputs = torch.argmax(outputs, 1)  # Convert model outputs to predicted class labels

            loss1.backward()  # Backward pass (compute gradients)

            # Apply gradient mask to preserve only non-zero weight gradients for convolution layers
            for name, param in model.named_parameters():
                if 'weight' and 'conv' in name:
                    if param.requires_grad:
                        if param.grad is not None:
                            grad_mask = param.data != 0  # Create mask for non-zero weights
                            param.grad *= grad_mask  # Apply mask to gradients

            optimizer.step()  # Update model parameters with optimizer (e.g. Adam)

            # Accumulate absolute loss scaled by batch size
            train_loss += abs(loss1.item()) * img.size(0)

            # Count number of correct predictions in the batch
            accuracy = torch.sum(outputs == target.to(device))

            # Accumulate correct predictions
            train_accuracy += accuracy

            # Increment total number of samples seen
            train_num += img.size(0)

        # Print epoch training statistics
        print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(epoch + 1, train_loss / train_num,
                                                                     train_accuracy / train_num))

        # Save loss and accuracy statistics for plotting (if needed)
        train_loss_all.append(train_loss / train_num)
        train_accur_all.append(train_accuracy.double().item() / train_num)

        # ================================
        # Save unpruned model weights
        # ================================
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        unpruned_weight_path = os.path.join(current_dir, f'data/{time}/weights_epoch_{epoch + 1}_before_pruning.pth')

        torch.save(model, unpruned_weight_path)  # Save entire model object

        # Count number of zero weights in convolution layers before pruning
        count_zero_weights_in_conv_layers_beforep(model)

        # Evaluate model performance before pruning
        test_beforep_model(model, testdata)

        # ================================
        # Prune the model
        # ================================
        binary_prune_model(model, threshold_c=0.15, threshold_b=0.146)

        # ================================
        # Save pruned model weights
        # ================================
        pruned_weight_path = os.path.join(current_dir, f'data/{time}/weights_epoch_{epoch+1}_after_pruning.pth')
        torch.save(model.state_dict(), pruned_weight_path)  # Save only model parameters (state dict)
        print(f"Pruned model weights saved to {pruned_weight_path}")

        # Count number of zero weights in convolution layers after pruning
        count_zero_weights_in_conv_layers_afterp(model)

        # # Evaluate model performance after pruning (optional)
        # test_afterp_model(model, testdata)

# ===========================================
# Define function to perform binary pruning on convolution layers of the model.
# ===========================================
def binary_prune_model(model, threshold_c, threshold_b):
    """
    Prunes similar filters in convolution layers based on Hamming distance similarity.

    Args:
        model: PyTorch model to be pruned.
        threshold_c (float): Frequency threshold for pruning (controls pruning aggressiveness).
        threshold_b (float): Variance scaling factor for distance-based pruning criterion.

    Steps:
        1. Iterate over model state_dict to find convolution layer weights.
        2. Binarize weights using sign function (+1/-1).
        3. Compute pairwise Hamming distances between filters.
        4. Identify filters with distance < (mean - threshold_b * variance).
        5. Prune filters that frequently appear in pruning candidates (based on threshold_c).
        6. Update pruned weights back to the model.
    """

    for key, value in model.state_dict().items():
        # Check if current parameter is a convolution layer weight
        if 'conv' in key and 'weight' in key:
            # Binarize weights: positive -> +1, negative/zero -> -1
            binary_value = torch.sign(value)

            # Clone binary weights for pruning manipulation
            weights = binary_value.clone()

            num_filters = weights.size(0)  # Number of filters in this conv layer
            hamming_distances = []  # List to store pairwise Hamming distances

            # Calculate pairwise Hamming distances between all unique filter pairs
            for i in range(num_filters):
                if torch.sum(weights[i]) != 0:  # Skip fully pruned filters
                    for j in range(i + 1, num_filters):
                        if torch.sum(weights[j]) != 0:
                            # Calculate Hamming distance between filter i and filter j
                            hamming_distance_ij = calculate_hamming_distance(weights[i], weights[j])
                            hamming_distances.append((i, j, hamming_distance_ij))

            # Extract distance values and convert to CPU for numpy operations
            distances = [distance.cpu() for _, _, distance in hamming_distances]

            # Calculate mean and variance of all pairwise distances
            ean_distance = np.mean(distances)
            variance_distance = np.var(distances)

            filters_to_prune = []  # List to store filters selected for pruning

            # Select filter pairs with distance below (mean - threshold_b * variance)
            for i, j, distance in hamming_distances:
                if distance < ean_distance - threshold_b * variance_distance:
                    filters_to_prune.append(i)
                    filters_to_prune.append(j)

            # Count frequency of each filter appearing in pruning candidates
            filter_counts = Counter(filters_to_prune)

            # Determine final filters to prune based on frequency exceeding threshold_c*(num_filters - 1)
            filters_to_prune_final = [filter_num for filter_num, count in filter_counts.items()
                                      if count > threshold_c * (num_filters - 1)]

            # Perform pruning: set weights of selected filters to zero
            for filter_num in filters_to_prune_final:
                weights[filter_num] = 0

            # Update pruned weights back to model state_dict using .copy_() to ensure in-place update
            model.state_dict()[key].copy_(weights)

# ===========================================
# Define function to count number of zero weights in convolution layers BEFORE pruning.
# ===========================================
def count_zero_weights_in_conv_layers_beforep(model):
    """
    Counts and logs the number of zero-valued weights in convolution layers before pruning.

    Args:
        model: PyTorch model whose convolution layers are analyzed.

    Outputs:
        Prints total weight count, zero weight count, and sparsity percentage.
        Appends results to 'before_pruning_info_1112_2.txt'.
    """
    zero_count = 0  # Initialize counter for zero weights
    total_count = 0  # Initialize counter for total weights

    # Iterate through model parameters
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            # Count number of zero weights in this parameter tensor
            zero_count += torch.sum(param == 0).item()

            # Count total number of weights in this parameter tensor
            total_count += param.numel()

    # Calculate percentage of zero weights (sparsity)
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0

    # Print sparsity statistics
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/before_pruning_info.txt')

    # Write statistics to file 'before_pruning_info_1112_2.txt' (append mode)
    with open(save_path, 'a') as f:
        f.write(f'Total parameters: {total_count}\n')
        f.write(f'Zero parameters before pruning: {zero_count}\n')
        f.write(f'pruning rate: {zero_percentage}\n')

# ===========================================
# Define function to count number of zero weights in convolution layers AFTER pruning.
# ===========================================
def count_zero_weights_in_conv_layers_afterp(model):
    """
    Counts and logs the number of zero-valued weights in convolution layers after pruning.

    Args:
        model: PyTorch model whose convolution layers are analyzed.

    Outputs:
        Prints total weight count, zero weight count, and sparsity percentage.
        Appends results to 'after_pruning_info.txt' in the current script directory.
    """
    zero_count = 0  # Initialize counter for zero weights
    total_count = 0  # Initialize counter for total weights

    # Iterate through model parameters
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()

    # Calculate percentage of zero weights (sparsity)
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0

    # Print sparsity statistics
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")

    # Write statistics to file 'after_pruning_info.txt' (append mode) in current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/after_pruning_info.txt')

    with open(save_path, 'a') as f:
        f.write(f'Total parameters: {total_count}\n')
        f.write(f'Zero parameters after pruning: {zero_count}\n')
        f.write(f'pruning rate: {zero_percentage}\n')

# ===========================================
# Define function to test model performance BEFORE pruning.
# ===========================================
def test_beforep_model(model, testdata):
    """
    Evaluates model performance on the test dataset before pruning.

    Args:
        model: PyTorch model to be evaluated.
        testdata: DataLoader for test dataset.

    Outputs:
        Prints and logs test loss and accuracy before pruning.
        Appends results to 'test-accuracy-before-pruning.txt'.
    """
    test_loss_all = []  # List to store test loss for potential plotting
    test_accur_all = []  # List to store test accuracy

    test_loss = 0  # Initialize cumulative test loss
    test_accuracy = 0.0  # Initialize cumulative correct prediction count
    test_num = 0  # Initialize total number of samples

    model.eval()  # Set model to evaluation mode (e.g. disables dropout, batchnorm updates)

    with torch.no_grad():  # Disable gradient calculation for testing (reduces memory and computation)
        test_bar = tqdm(testdata)  # Progress bar for test data

        for data in test_bar:
            img, target = data  # Unpack batch into images and labels

            outputs = model(img.to(device))  # Forward pass with input moved to device

            loss2 = loss(outputs, target.to(device))  # Compute loss for this batch

            outputs = torch.argmax(outputs, 1)  # Convert model outputs to predicted class labels

            # Accumulate absolute loss scaled by batch size
            test_loss += abs(loss2.item()) * img.size(0)

            # Count number of correct predictions in this batch
            accuracy = torch.sum(outputs == target.to(device))

            # Accumulate correct predictions
            test_accuracy += accuracy

            # Increment total number of samples seen
            test_num += img.size(0)

            # Calculate running accuracy
            accuracy_before_pruning = test_accuracy / test_num

    # Print overall test loss and accuracy before pruning
    print("test-Loss：{} , test-accuracy-before-pruning：{}".format(test_loss / test_num, accuracy_before_pruning))

    # Save test loss and accuracy to lists for potential plotting
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)

    # Write accuracy result to file 'test-accuracy-before-pruning.txt' in current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'data/{time}/test-accuracy-before-pruning.txt')

    with open(save_path, 'a') as f:
        f.write(f'Accuracy on test set before pruning: {accuracy_before_pruning}\n')


# ===========================================
# Define function to test model performance AFTER pruning.
# ===========================================
def test_afterp_model(model, testdata):
    """
    Evaluates model performance on the test dataset after pruning.

    Args:
        model: PyTorch model to be evaluated.
        testdata: DataLoader for test dataset.

    Outputs:
        Prints and logs test loss and accuracy after pruning.
        Appends results to 'test-accuracy-after-pruning.txt'.
    """
    test_loss_all = []  # List to store test loss for potential plotting
    test_accur_all = []  # List to store test accuracy

    test_loss = 0  # Initialize cumulative test loss
    test_accuracy = 0.0  # Initialize cumulative correct prediction count
    test_num = 0  # Initialize total number of samples

    model.eval()  # Set model to evaluation mode (e.g. disables dropout, batchnorm updates)

    with torch.no_grad():  # Disable gradient calculation for testing
        test_bar = tqdm(testdata)  # Progress bar for test data

        for data in test_bar:
            img, target = data  # Unpack batch into images and labels

            outputs = model(img.to(device))  # Forward pass with input moved to device

            loss2 = loss(outputs, target.to(device))  # Compute loss for this batch

            outputs = torch.argmax(outputs, 1)  # Convert model outputs to predicted class labels

            # Accumulate absolute loss scaled by batch size
            test_loss += abs(loss2.item()) * img.size(0)

            # Count number of correct predictions in this batch
            accuracy = torch.sum(outputs == target.to(device))

            # Accumulate correct predictions
            test_accuracy += accuracy

            # Increment total number of samples seen
            test_num += img.size(0)

            # Calculate running accuracy after pruning
            accuracy_after_pruning = test_accuracy / test_num

    # Print overall test loss and accuracy after pruning
    print("test-Loss：{} , test-accuracy-after-pruning_1112_2：{}".format(test_loss / test_num, accuracy_after_pruning))

    # Save test loss and accuracy to lists for potential plotting
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)

    # Write accuracy result to file 'test-accuracy-after-pruning.txt' in current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, f'/data/{time}/test-accuracy-after-pruning.txt')

    with open(save_path, 'a') as f:
        f.write(f'test_afterp_model: {accuracy_after_pruning}\n')

# ===========================================
# Define data transformation pipeline for MNIST dataset.
# ===========================================
data_transform = transforms.Compose([
    transforms.Grayscale(1),  # Convert image to grayscale with 1 channel
    # transforms.Resize((7, 7)),  # Resize image to (7,7) if needed (currently commented out)
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to torch.FloatTensor and scale to [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST dataset mean and std deviation
])

# ===========================================
# Load MNIST training dataset.
# ===========================================
train_data = torchvision.datasets.MNIST(
    root="./mnist_data/train",  # Directory to store downloaded dataset
    train=True,  # Load training set
    download=True,  # Download if dataset not present
    transform=data_transform  # Apply defined transforms
)

# Wrap training dataset in DataLoader for batch loading and shuffling
traindata = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# ===========================================
# Load MNIST test dataset.
# ===========================================
test_data = torchvision.datasets.MNIST(
    root="./mnist_data/val",  # Directory to store downloaded dataset
    train=False,  # Load test set
    download=True,  # Download if dataset not present
    transform=data_transform  # Apply defined transforms
)

# Wrap test dataset in DataLoader for batch loading
# Note: num_workers=0 by default for Windows; can set >0 in Linux for multi-process loading
testdata = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# ===========================================
# Set device to GPU if available, otherwise use CPU.
# ===========================================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# ===========================================
# Initialize VGG16 model and move to specified device.
# ===========================================
model = VGG16().to(device)

# Optional: Load pre-trained model weights if needed
# param = torch.load("vgg16_dnn_wtom_fashionmnist_wtom.pth")
# model.load_state_dict(param)

# ===========================================
# Define hyperparameters.
# ===========================================
beta = 1e-3  # Unused in current snippet, could be for regularization elsewhere
learning = 0.001  # Learning rate

# ===========================================
# Setup optimizer (Adam) with model parameters and specified learning rate.
# ===========================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning)

# Define loss function (CrossEntropyLoss) for multi-class classification tasks
loss = nn.CrossEntropyLoss()

# ===========================================
# Start training model for specified number of epochs.
# ===========================================
train_model(model, epoch=2, traindata=traindata)
