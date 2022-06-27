from cgi import test
from configparser import Interpolation

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model_ResNet import MTANDeepLabv3, MTLDeepLabv3
from model_SegNet import SegNetMTAN, SegNetSplit
from model_EdgeSegNet import EdgeSegNet
from model_GuideDepth import GuideDepth
from model_DDRNet import DualResNet, BasicBlock
from utils import *
from torch2trt import torch2trt, tensorrt_converter, get_arg, trt, add_missing_trt_tensors, torch_dtype_to_trt


""" Script for testing various networks at inference time. All timings are given in milliseconds """


# REGISTER NEW CONVERTERS
@tensorrt_converter('torch.nn.functional.silu')
def convert_silu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    layer = ctx.network.add_elementwise(input_trt, layer.get_output(0), trt.ElementWiseOperation.PROD)
    
    output._trt = layer.get_output(0)
    
@tensorrt_converter('torch.nn.functional.linear')
def convert_Linear(ctx):
    input = ctx.method_args[0]
    weight = get_arg(ctx, 'weight', 1, None)
    bias = get_arg(ctx, 'bias', 2, None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    # reshape to ...xNx1x1
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(input_trt.shape) + (1, 1) 

    bias_trt = trt.Weights(torch_dtype_to_trt(weight.dtype))
    if bias is not None:
        bias_trt = bias.detach().cpu().numpy()
        
    # add fully connected
    layer = ctx.network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=int(weight.shape[0]),
        kernel=weight.detach().cpu().numpy(),
        bias=bias_trt)

    # reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(output.shape[1:])

    output._trt = layer.get_output(0)


def inference_test(model_name, data_set, trt=True):

    # Define tasks based on dataset
    train_tasks = create_task_flags('all', data_set)

    # Load CUDA
    device = torch.device("cuda")

    # Load model
    if model_name == 'ResNet_split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif model_name == 'ResNet_mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
    elif model_name == "SegNet_split":
        model = SegNetSplit(train_tasks).to(device)
    elif model_name == "SegNet_mtan":
        model = SegNetMTAN(train_tasks).to(device)
    elif model_name == "EdgeSegNet":
        model = EdgeSegNet(train_tasks).to(device)
    elif model_name == "GuidedDepth":
        model = GuideDepth(train_tasks).to(device) 
    elif model_name == "DDRNet":
        model = DualResNet(BasicBlock, [2, 2, 2, 2], train_tasks, planes=32, spp_planes=128, head_planes=64).to(device)
    else:
        raise ValueError 

    if trt:
        # create example data
        if data_set == "nyuv2":
            x = torch.ones((1, 3, 288, 384)).cuda()
        else:
            x = torch.ones((1, 3, 256, 512)).cuda()
        model.eval()

        # convert to TensorRT feeding sample data as input
        model_trt = torch2trt(model, [x])
        model = model_trt
    
    model.eval()

    # Define start event, end event and number of repetitions
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))

    # GPU warm-up
    for _ in range(20):

        if data_set == "nyuv2":
            test_data = torch.randint(0, 256, (1, 3, 288, 384)).to(device)
            test_data = test_data.to(torch.float32)
        else:
            test_data = torch.randint(0, 256, (1, 3, 256, 512)).to(device)
            test_data = test_data.to(torch.float32)
        
        _ = model(test_data)

    # Measure performance 
    with torch.no_grad():
        for rep in range(repetitions):
            
            if data_set == "nyuv2":
                test_data = torch.randint(0, 256, (1, 3, 288, 384)).to(device)
                test_data = test_data.to(torch.float32)
            else:
                test_data = torch.randint(0, 256, (1, 3, 256, 512)).to(device)
                test_data = test_data.to(torch.float32)

            starter.record()
            _ = model(test_data)
            ender.record()

            # Wait for GPU to sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    # Calculate mean and std
    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)
    return mean_time, std_time

    
if __name__ == "__main__":

    models = ["ResNet_split", "ResNet_mtan", "SegNet_split", "SegNet_mtan", "EdgeSegNet", "GuidedDepth", "DDRNet"]

    nyuv2_timings = {}
    cityscapes_timings = {}
    
    # Create timings for NYUV2 dataset
    for model in models:
        model_mean, model_std = inference_test(model, "nyuv2", trt=False)
        nyuv2_timings[model] = [model_mean, model_std]
    
    print("Halfway there")

    # Create timings for CityScapes dataset
    for model in models:
        model_mean, model_std = inference_test(model, "cityscapes", trt=False)
        cityscapes_timings[model] = [model_mean, model_std]

    print("Timings for all models on NYUV2 given in milliseconds")
    print(nyuv2_timings)

    print("Timings for all models on CityScapes given in milliseconds")
    print(cityscapes_timings)