import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from model_ResNet import *
from model_SegNet import *
from model_EdgeSegNet import *
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Single-task Learning: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--network', default='SegNet_split', type=str, help='SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan, EdgeSegNet')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--task', default='seg', type=str, help='choose task for single task learning: seg, depth, normal, part_seg, disp')
parser.add_argument('--seed', default=0, type=int, help='gpu ID')

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
train_tasks = create_task_flags(opt.task, opt.dataset)

print('Training Task: {} - {} in Single Task Learning Mode with {}'
      .format(opt.dataset.title(), opt.task.title(), opt.network.upper()))

# define model
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

total_epoch = 200

# define optimizer and scheduler
if "ResNet" in opt.network:
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
elif "SegNet" in opt.network:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif "EdgeSegNet" in opt.network:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)

# define dataset
if opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 4

elif opt.dataset == 'cityscapes':
    dataset_path = 'dataset/cityscapes'
    train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
    test_set = CityScapes(root=dataset_path, train=False)
    batch_size = 4

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)
train_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset)
for index in range(total_epoch):

    # evaluating train data
    model.train()
    train_dataset = iter(train_loader)
    for k in range(train_batch):
        train_data, train_target = train_dataset.next()
        train_data = train_data.to(device)
        train_target = {task_id: train_target[task_id].to(device) for task_id in train_tasks.keys()}

        train_pred = model(train_data)
        optimizer.zero_grad()

        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
        train_loss[0].backward()
        optimizer.step()

        print("train target:", train_target["seg"].size())
        print("train_pred", train_pred[0].size())
        train_metric.update_metric(train_pred, train_target, train_loss)

    train_str = train_metric.compute_metric()
    train_metric.reset()

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = test_dataset.next()
            test_data = test_data.to(device)
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    task_dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric}
    np.save('logging/stl_{}_{}_{}_{}.npy'.format(opt.network, opt.dataset, opt.task, opt.seed), task_dict)