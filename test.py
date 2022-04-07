from model_ResNet import MTLDeepLabv3
from model_SegNet import SegNetSplit


tasks = {'seg': 13, 'depth': 1, 'normal': 3}

resnet = MTLDeepLabv3(tasks)

segnet = SegNetSplit()

print(resnet)

print("---------")

print(segnet)