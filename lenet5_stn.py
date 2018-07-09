from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
from stn_module import STN


DATASET_DIR = "./DATA"
NUM_CLASSES = len(os.listdir(DATASET_DIR))


class LeNet5STN(nn.Module):
    def __init__(self, num_classes=-1):
        super(LeNet5STN, self).__init__()
        if num_classes == -1:
            self.num_classes = NUM_CLASSES
        else:
            self.num_classes = num_classes
        self.stn = STN()
        # Feature Extraction
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),
            nn.Linear(120, 84),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # lenet5
        x = self.conv(x)
        # torch.Size([128, 16, 53, 53])
        x = x.view(-1, 16 * 53 * 53)
        x = self.clf(x)
        # x = F.log_softmax(x)
        return x
