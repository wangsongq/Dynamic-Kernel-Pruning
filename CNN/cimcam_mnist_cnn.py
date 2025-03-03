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




class mask_dnnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight):
        mask = torch.zeros_like(input_)
        mask = torch.where(input_.ge(weight), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
        ctx.save_for_backward(input_, weight, mask)
        out = input_.mul(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, mask = ctx.saved_variables
        grad_input = grad_weight = None

        c_i = input_ - weight
        a_i = torch.zeros_like(c_i)
        a_i = torch.where(c_i.lt(1.0) & c_i.ge(0.4), torch.tensor(0.4).to(device), a_i)
        a_i = torch.where(c_i.lt(0.4) & c_i.ge(0.0), -4.0 * c_i + 2.0, a_i)
        a_i = torch.where(c_i.lt(0.0) & c_i.gt(-0.4), 4.0 * c_i + 2.0, a_i)
        a_i = torch.where(c_i.le(-0.4) & c_i.gt(-1.0), torch.tensor(0.4).to(device), a_i)

        if ctx.needs_input_grad[0]:
            grad_input_1 = mask + a_i.mul(input_)
            grad_input = grad_output.mul(grad_input_1)

        if ctx.needs_input_grad[1]:
            grad_weight_1 = -a_i.mul(input_)
            grad_weight = grad_output.mul(grad_weight_1)

        return grad_input, grad_weight


class maskLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(maskLinear, self).__init__(in_features, out_features, bias=bias)
        self.output_ = None
        self.threshold = torch.nn.Parameter(torch.rand(out_features, in_features), requires_grad=True)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        tw = self.weight
        ta = x
        tw = mask_dnnFunction.apply(tw, self.threshold)
        output = F.linear(ta, tw, self.bias)
        self.output_ = output
        # self.weight.data = tw
        return output
    # def backward(self, )


class maskConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(maskConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.threshold = torch.nn.Parameter(torch.rand(out_channels, in_channels // groups, *kernel_size),
                                            requires_grad=True)

    def forward(self, x):
        bw = self.weight
        ba = x
        bw = mask_dnnFunction.apply(bw, self.threshold)
        # self.weight.data = bw
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# 定义带有滤波器剪枝逻辑的Conv2d模块
class PrunedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, threshold_c=0.5, threshold_b=3, **kwargs):
        super(PrunedConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.threshold_c = threshold_c
        self.threshold_b = threshold_b

    def forward(self, input):
        # 获取权重矩阵
        weights = self.weight.data
        # 二值化权重
        binary_weights = torch.where(weights > torch.tensor(0, device=device), torch.tensor(1.0, device=device),
                                     torch.tensor(0.0, device=device))
        # 二值化输入
        binary_input = torch.where(input > torch.tensor(0, device=device), torch.tensor(1.0, device=device),
                                   torch.tensor(0.0, device=device))

        num_filters = weights.size(0)
        filters_to_prune = set()
        hamming_distances = []

        for filter_1 in range(num_filters):
            if torch.sum(self.weight[filter_1]) != 0:
                for filter_2 in range(filter_1 + 1, num_filters):
                    if torch.sum(self.weight[filter_2]) != 0:
                        hamming_distance_12 = calculate_hamming_distance(binary_weights[filter_1],
                                                                         binary_weights[filter_2])  # 计算汉明距离
                        hamming_distances.append((filter_1, filter_2, hamming_distance_12))
        distances = [distance for _, _, distance in hamming_distances]
        ean_distance = np.mean(distances)
        variance_distance = np.var(distances)
        for i, j, distance in hamming_distances:
            if distance < ean_distance - self.threshold_b * variance_distance:
                filters_to_prune.add(i)
                filters_to_prune.add(j)

        # 使用Counter计算集合中每个元素的频率
        filter_counts = Counter(filters_to_prune)
        # 计算每个元素出现的次数
        filters_to_prune_final = [filter_num for filter_num, count in filter_counts.items() if
                                  count > self.threshold_c * (num_filters - 1)]

        # 进行滤波器剪枝操作
        for filter_num in filters_to_prune_final:
            # 将需要剪枝的滤波器权重设置为0
            self.weight.data[filter_num].zero_()

        return nn.functional.conv2d(binary_input, self.weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


# 计算汉明距离的函数
def calculate_hamming_distance(weight1, weight2):
    # 实现汉明距离的计算逻辑
    # 将卷积核的权重矩阵展开为一维向量
    weights1 = weight1.view(-1)
    weights2 = weight2.view(-1)
    # 计算汉明距离
    hamming_distance = (weights1 != weights2).sum()
    return hamming_distance



def euclidean_distance(A, B):
    # 计算欧几里得距离
    distance = torch.norm(A - B)
    # 返回距离
    return distance


# 定义带有滤波器剪枝逻辑的Conv2d模块
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input):
        # 获取权重矩阵
        weights = self.weight.data
        # 二值化权重
        binary_weights = torch.where(weights > torch.tensor(0, device=device), torch.tensor(1.0, device=device),
                                     torch.tensor(-1.0, device=device))
        # 二值化输入
        binary_input = torch.where(input > torch.tensor(0, device=device), torch.tensor(1.0, device=device),
                                   torch.tensor(-1.0, device=device))

        return nn.functional.conv2d(binary_input, binary_weights, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input, None


class BiLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=False):
        super(BiLinear, self).__init__(in_features, out_features, bias=True)
        self.binary_act = binary_act
        self.output_ = None

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = BinaryQuantize().apply(bw)
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)
        output = F.linear(ba, bw, self.bias)
        self.output_ = output
        return output


class BiConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = BinaryQuantize().apply(bw)
        # ba = TernaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinaryLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input):
        # 获取权重矩阵
        weights = self.fc.weight.data
        # 二值化权重
        binary_weights = torch.where(weights > 0, torch.tensor(1.0, device=weights.device),
                                     torch.tensor(-1.0, device=weights.device))

        # 二值化输入
        binary_input = torch.where(input > 0, torch.tensor(1.0, device=input.device),
                                   torch.tensor(-1.0, device=input.device))

        # 进行二值全连接操作
        output = F.linear(binary_input, binary_weights, self.fc.bias)

        return output


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 第一层卷积层
        self.conv1 = BinaryConv2d(1, 32, kernel_size=3, padding=1)  # 输出: 32@28x28
        # 第二层卷积层
        self.conv2 = BinaryConv2d(32, 64, kernel_size=3, padding=1)  # 输出: 64@14x14
        # 第三层卷积层
        self.conv3 = BinaryConv2d(64, 16, kernel_size=3, padding=1)  # 输出: 16@7x7
        # 最终的全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 256)  # 假设是10类分类问题
        self.fc2 = nn.Linear(256, 128)  # 假设是10类分类问题
        self.fc3 = nn.Linear(128, 10)  # 假设是10类分类问题


    def forward(self, x):
        # 应用第一层卷积并激活
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 池化到14x14
        # 应用第二层卷积并激活
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 池化到7x7
        # 应用第三层卷积并激活
        x = F.relu(self.conv3(x))
        # 展平
        x = x.view(-1, 16 * 7 * 7)
        # 应用全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x








def train_model(model, epoch, traindata):
    for epoch in range(epoch):  # 开始迭代
        train_loss_all = []
        train_accur_all = []
        train_loss = 0  # 训练集的损失初始设为0
        train_num = 0.0  #
        train_accuracy = 0.0  # 训练集的准确率初始设为0
        model.train()  # 将模型设置成 训练模式
        train_bar = tqdm(traindata)  # 用于进度条显示，没啥实际用处
        for step, data in enumerate(train_bar):  # 开始迭代跑， enumerate这个函数不懂可以查查，将训练集分为 data是序号，data是数据
            img, target = data  # 将data 分位 img图片，target标签
            optimizer.zero_grad()  # 清空历史梯度
            outputs = model(img.to(device))  # 将图片打入网络进行训练,outputs是输出的结果

            loss1 = loss(outputs, target.to(device))  # 计算神经网络输出的结果outputs与图片真实标签target的差别-这就是我们通常情况下称为的损失
            outputs = torch.argmax(outputs, 1)  # 会输出10个值，最大的值就是我们预测的结果 求最大值
            loss1.backward()  # 神经网络反向传播
            for name, param in model.named_parameters():
                if 'weight' and 'conv' in name:
                    if param.requires_grad:
                        if param.grad is not None:
                            grad_mask = param.data != 0  # 创建一个掩码，只有非零的权重的梯度才被保留
                            param.grad *= grad_mask

            optimizer.step()  # 梯度优化 用上面的abam优化
            train_loss += abs(loss1.item()) * img.size(0)  # 将所有损失的绝对值加起来
            accuracy = torch.sum(outputs == target.to(device))  # outputs == target的 即使预测正确的，统计预测正确的个数,从而计算准确率
            train_accuracy = train_accuracy + accuracy  # 求训练集的准确率
            train_num += img.size(0)  #

        print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(epoch + 1, train_loss / train_num,  # 输出训练情况
                                                                    train_accuracy / train_num))
        train_loss_all.append(train_loss / train_num)  # 将训练的损失放到一个列表里 方便后续画图
        train_accur_all.append(train_accuracy.double().item() / train_num)  # 训练集的准确率

        # 保存训练完的权重
        torch.save(model, f'weights_epoch{epoch + 1}_before_pruning.pth')

        # 统计卷积层中权重为0的个数
        count_zero_weights_in_conv_layers_beforep(model)

        # 测试剪枝后的模型
        test_beforep_model(model, testdata)
        
        # 剪枝
        binary_prune_model(model, threshold_c=0.15, threshold_b=0.146)
        
        # 保存剪枝后的权重
        pruned_weight_path = f'weights_epoch_{epoch+1}_after_pruning.pth'
        torch.save(model.state_dict(), pruned_weight_path)
        print(f"Pruned model weights saved to {pruned_weight_path}")

        # 统计剪枝后卷积层中权重为0的个数
        count_zero_weights_in_conv_layers_afterp(model)
        
        # # 测试剪枝后的模型
        # test_afterp_model(model, testdata)
    
# 保存模型参数为二进制形式
def binary_prune_model(model, threshold_c, threshold_b):
    for key, value in model.state_dict().items():
        if 'conv' in key and 'weight' in key:  # 对卷积层进行pruning
            # print(key)
            binary_value = torch.sign(value)
            weights = binary_value.clone() 
            close_filters = []
            num_filters = weights.size(0)
            hamming_distances = []

            for i in range(num_filters):
                if torch.sum(weights[i]) != 0:
                    for j in range(i + 1, num_filters):
                        if torch.sum(weights[j]) != 0:
                            hamming_distance_ij = calculate_hamming_distance(weights[i], weights[j])
                            # print(hamming_distance_ij)
                            hamming_distances.append((i, j, hamming_distance_ij))
            distances = [distance.cpu() for _, _, distance in hamming_distances]
            # distances = [distance for _, _, distance in hamming_distances]
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
            # print(len(filter_counts))
            # # top_310_elements = filter_counts.most_common(300)
            # top_310_counter = Counter(dict(filter_counts.most_common(10)))
            # print(len(top_310_counter))
            # print(filter_counts)
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


def count_zero_weights_in_conv_layers_beforep(model):
    zero_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            # print(param)
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")
    # 写入信息到文件
    with open('before_pruning_info_1112_2.txt', 'a') as f:
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
        if 'weight' in name and 'conv' in name:
            zero_count += torch.sum(param == 0).item()
            total_count += param.numel()
    zero_percentage = 100 * zero_count / total_count if total_count > 0 else 0
    print(f"Total convolution weights: {total_count}, Zero weights: {zero_count}, Percentage: {zero_percentage:.2f}%")
    # 写入信息到文件
    with open('after_pruning_info_1112_2.txt', 'a') as f:
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
            img, target = data

            outputs = model(img.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)
            accuracy_before_pruning = test_accuracy / test_num

    print("test-Loss：{} , test-accuracy-before-pruning：{}".format(test_loss / test_num, accuracy_before_pruning))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)
    # 写入信息到文件
    with open('test-accuracy-before-pruning_1112_2.txt', 'a') as f:
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
            img, target = data

            outputs = model(img.to(device))
            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)
            accuracy_after_pruning = test_accuracy / test_num

    print("test-Loss：{} , test-accuracy-after-pruning_1112_2：{}".format(test_loss / test_num, accuracy_after_pruning))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)
    # 写入信息到文件
    with open('test-accuracy-after-pruning_1112_2.txt', 'a') as f:
        # f.write(f'Accuracy on test set before pruning: {accuracy_before_pruning}\n')
        f.write(f'test_afterp_model: {accuracy_after_pruning}\n')



data_transform = transforms.Compose([
    transforms.Grayscale(1),
    # transforms.Resize((7, 7)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))])

train_data = torchvision.datasets.MNIST(root="./mnist_data/train", train=True, download=True,
                                        transform=data_transform)

traindata = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# test_data = torchvision.datasets.CIFAR10(root = "./data" , train = False ,download = False,
#                                           transform = trans)
test_data = torchvision.datasets.MNIST(root="./mnist_data/val", train=False, download=True,
                                       transform=data_transform)
testdata = DataLoader(dataset=test_data, batch_size=64, shuffle=True
                      )  # windows系统下，num_workers设置为0，linux系统下可以设置多进程

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

model = VGG16().to(device)
# param = torch.load("vgg16_dnn_wtom_fashionmnist_wtom.pth")
# model.load_state_dict(param)
# train
beta = 1e-3

learning = 0.001  # 学习率



# setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()  # 损失函数


train_model(model, epoch=100, traindata=traindata)


    # # 加载剪枝前的权重
    # pruned_state_dict = torch.load(f'weights01_epoch_{epoch+1}_before_pruning.pth')
    # model.load_state_dict(pruned_state_dict)
    # test_beforep_model(model, testdata)

    # # 加载剪枝后的权重
    # pruned_state_dict = torch.load(f'weights01_epoch_{epoch+1}_after_pruning.pth')
    # model.load_state_dict(pruned_state_dict)
    # test_afterp_model(model, testdata)