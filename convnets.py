import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os


DATASET_DIR = "./DATA"


class ConvNetA(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetA, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 96, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetB(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetB, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetC(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetC, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 96, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetD(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetD, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 48, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetE(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetE, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 72, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(72),
            nn.Conv2d(72, 144, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(144),
            nn.Conv2d(144, 96, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96)
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 6 * 6, 768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(768, 768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(768, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 96 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetF(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetF, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 24, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetG(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetG, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 96, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.classifier(x)
        return x


class ConvNetH(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetH, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 32, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 13 * 13, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x


class ConvNetI(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetI, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 13 * 13, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x


class ConvNetJ(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetJ, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 31 * 31, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x


class ConvNetK(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetK, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 32, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 256, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 13 * 13, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x


class ConvNetL(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetL, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 13 * 13, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x


class ConvNetM(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetM, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x


class ConvNetN(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNetN, self).__init__()
        if num_classes == -1:
            self.num_classes = len(os.listdir(DATASET_DIR))
        else:
            self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, 500, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(500, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        _, d, w, h = x.size()
        x = x.view(-1, d * w * h)
        x = self.classifier(x)
        return x

