import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# Define VGG16
# --------------------------------------------------------------------------------

class VGG16(nn.Module):
    def __init__(self, tasks):
        super(VGG16, self).__init__()

        self.tasks = tasks







        # define task specific decoders
        if all (k in tasks for k in ('seg', 'depth', 'normal')):
        
            self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=filter[0], out_channels=13, kernel_size=1, padding=0))
            self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
            self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))
        else:
            self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=filter[0], out_channels=19, kernel_size=1, padding=0))
            self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=filter[0], out_channels=10, kernel_size=1, padding=0))
            self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0], out_channels=filter[0], kernel_size=3, padding=1),
                                            nn.Conv2d(in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        
        self.decoders = nn.ModuleList([self.pred_task1, self.pred_task2, self.pred_task3])


        def forward(self, x):
            pass