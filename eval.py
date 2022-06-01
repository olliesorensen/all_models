from cgi import test
import cmath
from configparser import Interpolation
from webbrowser import BackgroundBrowser
from PIL import Image

import torch
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model_ResNet import MTANDeepLabv3, MTLDeepLabv3
from model_SegNet import SegNetMTAN, SegNetSplit
from model_EdgeSegNet import EdgeSegNet
from create_dataset import *
from utils import *


parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan, EdgeSegNet')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
opt = parser.parse_args()

def main():

    model_name = opt.network
    data_set = opt.dataset

    batch_size = 1

    # Define tasks based on dataset
    train_tasks = create_task_flags('all', opt.dataset, with_noise=False)

    # Define dataset
    if opt.dataset == 'nyuv2':
        dataset_path = 'dataset/nyuv2'
        test_set = NYUv2(root=dataset_path, train=False)
        batch_size = 1

    elif opt.dataset == 'cityscapes':
        dataset_path = 'dataset/cityscapes'
        test_set = CityScapes(root=dataset_path, train=False)
        batch_size = 1

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False
    )

    # Load CUDA
    device = torch.device("cpu")

    # Load model
    if opt.network == 'ResNet_split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif opt.network == 'ResNet_mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
    elif opt.network == "SegNet_split":
        model = SegNetSplit(train_tasks).to(device)
    elif opt.network == "SegNet_mtan":
        model = SegNetMTAN(train_tasks).to(device)
    elif opt.network == "EdgeSegNet":
        model = EdgeSegNet(train_tasks).to(device)   

    model.load_state_dict(torch.load(f"models/{opt.weight}/model_{model_name}_{data_set}.pth", map_location=device))
    model.eval()

    # Create iteratable object
    test_dataset = iter(test_loader)

    # Make prediction
    with torch.no_grad():
        test_data, test_target = test_dataset.next()
        test_data = test_data.to(device)
        test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}
        test_pred = model(test_data)

    test_target = list(test_target.values())

    if opt.dataset == "nyuv2":

        # Reshape input data
        test_data = torch.squeeze(test_data, 0)
        test_target[0] = torch.squeeze(test_target[0], 0)
        test_target[1] = torch.squeeze(test_target[1], 0)
        test_target[2] = torch.squeeze(test_target[2], 0)
        test_data = test_data.permute(1, 2, 0)
        test_target[1] = test_target[1].permute(1, 2, 0)
        test_target[2] = test_target[2].permute(1, 2, 0)   

        # Reshape output data 
        test_pred[0] = torch.squeeze(test_pred[0], 0)
        test_pred[1] = torch.squeeze(test_pred[1], 0)
        test_pred[2] = torch.squeeze(test_pred[2], 0)
        test_pred[1] = test_pred[1].permute(1, 2, 0)
        test_pred[2] = test_pred[2].permute(1, 2, 0)

        # From Tensor to Numpy
        test_data = test_data.cpu().detach().numpy()

        test_target[0] = test_target[0].cpu().detach().numpy()    
        test_target[1] = test_target[1].cpu().detach().numpy()       
        test_target[2] = test_target[2].cpu().detach().numpy()
        
        test_pred[0] = test_pred[0].cpu().detach().numpy()    
        test_pred[1] = test_pred[1].cpu().detach().numpy()    
        test_pred[2] = test_pred[2].cpu().detach().numpy()  

        seg_mask2 = visualize_segmentation(test_target[0], opt.dataset, num_classes=13)
        seg_mask = visualize_segmentation(np.argmax(test_pred[0], axis=0), opt.dataset, num_classes=13) 

        # Create subplots
        fig, axarr = plt.subplots(2, 4)
        # Remove values between 0 and -1
        #test_data = test_data + 1
        axarr[0, 0].imshow(test_data) # Original
        axarr[0, 1].imshow(seg_mask2) # Segmentation truth
        axarr[0, 2].imshow(test_target[1], cmap="gray") # Depth truth
        axarr[0, 3].imshow(test_target[2]) # Normals truth
        axarr[1, 0].imshow(test_data) # Original
        axarr[1, 1].imshow(seg_mask) # Segmentation prediction
        axarr[1, 2].imshow(test_pred[1], cmap="gray") # Depth prediction
        axarr[1, 3].imshow(test_pred[2]) # Normals prediction
        plt.show()

    elif opt.dataset == "cityscapes":

        # Reshape input data
        test_data = torch.squeeze(test_data, 0)
        test_target[0] = torch.squeeze(test_target[0], 0)
        test_target[1] = torch.squeeze(test_target[1], 0)
        test_target[2] = torch.squeeze(test_target[2], 0)
        test_data = test_data.permute(1, 2, 0)
        test_target[2] = test_target[2].permute(1, 2, 0)   

        # Reshape output data 
        test_pred[0] = torch.squeeze(test_pred[0], 0)
        test_pred[1] = torch.squeeze(test_pred[1], 0)
        test_pred[2] = torch.squeeze(test_pred[2], 0)
        test_pred[2] = test_pred[2].permute(1, 2, 0)

        # From Tensor to Numpy
        test_data = test_data.cpu().detach().numpy()

        test_target[0] = test_target[0].cpu().detach().numpy()    
        test_target[1] = test_target[1].cpu().detach().numpy()       
        test_target[2] = test_target[2].cpu().detach().numpy()
        
        test_pred[0] = test_pred[0].cpu().detach().numpy()    
        test_pred[1] = test_pred[1].cpu().detach().numpy()    
        test_pred[2] = test_pred[2].cpu().detach().numpy() 

        print("Input and targets")
        print(test_data.shape)
        print(test_target[0].shape)
        print(test_target[1].shape)
        print(test_target[2].shape)

        print("Predictions")
        print(test_pred[0].shape)
        print(test_pred[1].shape)
        print(test_pred[2].shape)

        # Target segmentation
        seg_mask2 = visualize_segmentation(test_target[0], opt.dataset, num_classes=19) 
        seg_mask3 = visualize_segmentation(test_target[1], opt.dataset, num_classes=10) 

        # Predicted segmentation
        seg_mask0 = visualize_segmentation(np.argmax(test_pred[0], axis=0), opt.dataset, num_classes=19) 
        seg_mask1 = visualize_segmentation(np.argmax(test_pred[1], axis=0), opt.dataset, num_classes=10) 

        # Create subplots
        fig, axarr = plt.subplots(2, 4)
        fig.suptitle(f"Visual Evaluation\n {model_name}, {data_set}")
        # Remove values between 0 and -1
        #test_data = test_data + 1
        axarr[0, 0].imshow(test_data) # Original
        axarr[0, 1].imshow(seg_mask2) # Semantic Segmentation truth
        axarr[0, 2].imshow(seg_mask3) # Part Segmentation Truth
        axarr[0, 3].imshow(test_target[2], cmap="gray") # Disparity Truth
        axarr[1, 0].imshow(test_data) # Original
        axarr[1, 1].imshow(seg_mask0) # Semantic Segmentation prediction
        axarr[1, 2].imshow(seg_mask1) # Part Segmentation Prediction
        axarr[1, 3].imshow(test_pred[2], cmap="gray") # Disparity Prediction
        fig.tight_layout()
        plt.show()


def visualize_segmentation(temp, dataset, num_classes):
    
    if dataset == "nyuv2":
        background = [128, 128, 128]
        bed = [128, 0, 0]
        books = [192, 192, 128]
        ceiling = [255, 69, 0]
        chair = [128, 64, 128]
        floor = [60, 40, 222]
        furniture = [128, 128, 0]
        objects = [192, 128, 128]
        painting = [64, 64, 128]
        sofa = [64, 0, 128]
        table = [0, 128, 192]
        tv = [70, 192, 0]
        wall = [50, 90, 128]
        window = [0, 128, 0]

        label_colours = np.array([background, bed, books, ceiling, chair, floor, furniture, objects, 
                                painting, sofa, table, tv, wall, window])
    
    elif dataset == "cityscapes" and num_classes == 19:
        road = [128,64,128]
        sidewalk = [244,35,232]
        building = [70,70,70]
        wall = [102,102,156]
        fence = [100,40,40]
        pole = [153,153,153]
        traffic_light = [250,170,30] 
        traffic_sign = [220,220,0] 
        vegetation = [107,142, 35]
        terrain = [145,170,100]
        sky = [70,130,180]
        person = [220,20,60]
        rider = [255,0,0]
        carm_truck = [0,0,142]
        bus = [0,60,100]
        caravan = [0,0,90]
        trailer = [0,0,110]
        train = [0,80,100]
        motorcycle = [0,0,230]

        label_colours = np.array([road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign, 
                                vegetation, terrain, sky, person, rider, carm_truck, bus, caravan, trailer,
                                train, motorcycle])

    elif dataset == "cityscapes" and num_classes == 10:
        one = [128,64,128]
        two = [244,35,232]
        three = [70,70,70]
        four = [102,102,156]
        five = [190,153,153]
        six = [153,153,153]
        seven = [250,170,30] 
        eight = [220,220,0] 
        nine = [107,142, 35]
        ten = [152,251,152]

        label_colours = np.array([one, two, three, four, five, six, seven, eight, nine, ten])

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()

    for l in range(0, num_classes):
        r[temp==l] = label_colours[l, 0]
        g[temp==l] = label_colours[l, 1]
        b[temp==l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    
    return rgb


if __name__ == "__main__":

    main()