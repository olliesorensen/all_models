import torch
import torch.nn as nn
import torch.nn.functional as F

""" Implementation of the EdgeSegNet network,
adapted form https://github.com/lorenmt/auto-lambda """
# --------------------------------------------------------------------------------
# Define Modules
# --------------------------------------------------------------------------------
class ResidualBottleneckModule(nn.Module):

    def __init__(self):
        
        super(ResidualBottleneckModule, self).__init__()
        self.bn_0 = nn.BatchNorm2d(13)
        self.relu_0 = nn.ReLU()
        
        self.conv_1x1_0 = nn.Conv2d(in_channels=13, out_channels=193, kernel_size=1, stride=1)

        self.bn_1 = nn.BatchNorm2d(193)
        self.relu_1 = nn.ReLU()

        self.conv_3x3_0 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=3, stride=2, padding=1)

        self.bn_2 = nn.BatchNorm2d(193)
        self.relu_2 = nn.ReLU()

        self.conv_1x1_1 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=1, stride=1)

        self.conv_3x3_1 = nn.Conv2d(in_channels=13, out_channels=193, kernel_size=3, stride=2, padding=1)

    def forward(self, input_batch):

        x_bn_0 = self.bn_0(input_batch)
        x_relu_0 = self.relu_0(x_bn_0)
        x_conv_3x3_1 = self.conv_3x3_1(x_relu_0)

        x = self.conv_1x1_0(x_relu_0)
        
        x = self.bn_1(x)
        x = self.relu_1(x)
        
        x = self.conv_3x3_0(x)
        
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv_1x1_1(x)

        output = x_conv_3x3_1 + x

        return output


class BottleneckReductionModule(nn.Module):

    def __init__(self):

        super(BottleneckReductionModule, self).__init__()

        self.conv_3x3_0 = nn.Conv2d(in_channels=13, out_channels=193, kernel_size=3, stride=8)
        self.relu_0 = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=1, stride=1)
        self.relu_1 = nn.ReLU()
        self.conv_3x3_1 = nn.Conv2d(in_channels=193, out_channels=193, kernel_size=3, stride=1, padding=1)

    def forward(self, input_batch):

        x = self.conv_3x3_0(input_batch)

        x = self.relu_0(x)
        x = self.conv_1x1(x)
        x = self.relu_1(x)
        x = self.conv_3x3_1(x)

        return x


class RefineModule(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(RefineModule, self).__init__()
        
        self.conv_1x1_0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv_1x1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.relu_0 = nn.ReLU()
        self.conv_3x3_0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv_3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input_batch):

        x_conv_1x1_0 = self.conv_1x1_0(input_batch)

        x = self.conv_1x1_1(x_conv_1x1_0)
        x = self.relu_0(x)
        x = self.conv_3x3_0(x)
        x = self.relu_1(x)
        x = self.conv_3x3_1(x)

        return x + x_conv_1x1_0


class BilinearResizeModule(nn.Module):

    def __init__(self, scale_factor):

        super(BilinearResizeModule, self).__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, input_batch):

        x = self.upsample(input_batch)
        return x


# --------------------------------------------------------------------------------
# Define EdgeSegNet
# --------------------------------------------------------------------------------
class EdgeSegNet(nn.Module):

    def __init__(self, tasks):

        super(EdgeSegNet, self).__init__()

        self.tasks = tasks

        self.conv_7x7 = nn.Conv2d(in_channels=3, out_channels=13, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Top part of the graph, before the first addition of (32, 32, 193)
        self.bottleneck_module = BottleneckReductionModule()
        self.bilinear_resize_x2_0 = BilinearResizeModule(scale_factor=2)

        # Bottom part of the graph, before the first addition of (32, 32, 193)
        self.residual_module = ResidualBottleneckModule()
        self.refine_module_0 = RefineModule(in_channels=193, out_channels=193)

        # Top part of the graph, after the first addition of (32, 32, 193)
        self.refine_module_1 = RefineModule(in_channels=193, out_channels=217)
        self.bilinear_resize_x2_1 = BilinearResizeModule(scale_factor=2)

        # Bottom part of the graph, maxpool to (64, 64, 217) addition
        self.refine_module_2 = RefineModule(in_channels=13, out_channels=217)

        # After the (64, 64, 217) addition
        self.refine_module_3 = RefineModule(in_channels=217, out_channels=217)
        self.bilinear_resize_x4 = BilinearResizeModule(scale_factor=4)
        self.conv_1x1 = nn.Conv2d(in_channels=217, out_channels=32, kernel_size=1, stride=1)

        # define task specific decoders
        if all (k in tasks for k in ('seg', 'depth', 'normal')):
        
            self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=217, out_channels=217, kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=217, out_channels=13, kernel_size=1, padding=0))
            self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=217, out_channels=217, kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=217, out_channels=1, kernel_size=1, padding=0))
            self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=217, out_channels=217, kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=217, out_channels=3, kernel_size=1, padding=0))
        else:
            self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=217, out_channels=217, kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=217, out_channels=19, kernel_size=1, padding=0))
            self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=217, out_channels=217, kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=217, out_channels=10, kernel_size=1, padding=0))
            self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=217, out_channels=217, kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=217, out_channels=1, kernel_size=1, padding=0))
        
        self.decoders = nn.ModuleList([self.pred_task1, self.pred_task2, self.pred_task3])

    def forward(self, input_batch):

        _, _, im_h, im_w = input_batch.shape
        
        x_conv_7x7 = self.conv_7x7(input_batch)
        x_maxpool = self.maxpool(x_conv_7x7)

        x_top = self.bottleneck_module(x_conv_7x7)
        x_top = self.bilinear_resize_x2_0(x_top)

        x_bottom = self.residual_module(x_maxpool)
        x_bottom = self.refine_module_0(x_bottom)

        # (32, 32, 193) addition
        x_top = x_top + x_bottom
        x_top = self.refine_module_1(x_top)
        x_top = self.bilinear_resize_x2_1(x_top)

        x_bottom = self.refine_module_2(x_maxpool)

        # (64, 64, 217) addition
        x_top = x_top + x_bottom

        x_top = self.refine_module_3(x_top)
        x_top = self.bilinear_resize_x4(x_top)
        #x_top = self.conv_1x1(x_top) # Replace this with three decoders?

        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x_top), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        
        return out