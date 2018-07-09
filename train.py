import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from alex_stn import STNAlexNet
from lenet5 import LeNet5
import argparse
from lenet5_stn import LeNet5STN
from alexnet import AlexNet
from hdf5dataset import Hdf5Dataset, Hdf5DatasetMPI
from capsnet import CapsuleNet, CapsuleLoss, augmentation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from convnets import ConvNetA, ConvNetB, ConvNetC, ConvNetD, ConvNetE, ConvNetF, ConvNetG, ConvNetH, ConvNetI, ConvNetJ, ConvNetK, ConvNetL, ConvNetM, ConvNetN
from datetime import datetime
import torchvision.models as models
from rq import get_current_job
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_generic_model(model_name="alexnet",
                        dataset="custom",
                        num_classes=-1,
                        batch_size=8,
                        is_transform=1,
                        num_workers=2,
                        lr_decay=1,
                        l2_reg=0,
                        hdf5_path="dataset-224x224.hdf5",
                        trainset_dir="./TRAIN_data_224",
                        testset_dir="./TEST_data_224",
                        convert_grey=False,
                        learning_rate=0.001,
                        num_epochs=4000):
    CHKPT_PATH = "./checkpoint_{}.PTH".format(model_name)
    print("CUDA:")
    print(torch.cuda.is_available())
    if is_transform:

        trans_ls = []
        if convert_grey:
            trans_ls.append(transforms.Grayscale(num_output_channels=1))
        trans_ls.extend(
            [
                transforms.Resize((512, 512)),
                # transforms.RandomCrop((224, 224)),
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        transform = transforms.Compose(trans_ls)
    else:
        transform = None

    print("DATASET FORMAT: {}".format(dataset))
    print("TRAINSET PATH: {}".format(trainset_dir))
    print("TESTSET PATH: {}".format(testset_dir))
    print("HDF5 PATH: {}".format(hdf5_path))
    if dataset == "custom":
        trainset = torchvision.datasets.ImageFolder(root=trainset_dir, transform=transform)
        train_size = len(trainset)
        testset = torchvision.datasets.ImageFolder(root=testset_dir, transform=transform)
        test_size = len(testset)
    elif dataset == "cifar":
        trainset = torchvision.datasets.CIFAR10(root="CIFAR_TRAIN_data", train=True, download=True, transform=transform)
        train_size = len(trainset)
        testset = torchvision.datasets.CIFAR10(root="CIFAR_TEST_data", train=False, download=True, transform=transform)
        test_size = len(testset)
    elif dataset == "hdf5":
        if num_workers == 1:
            trainset = Hdf5Dataset(hdf5_path, transform=transform, is_test=False)
        else:
            trainset = Hdf5DatasetMPI(hdf5_path, transform=transform, is_test=False)
        train_size = len(trainset)
        if num_workers == 1:
            testset = Hdf5Dataset(hdf5_path, transform=transform, is_test=True)
        else:
            testset = Hdf5DatasetMPI(hdf5_path, transform=transform, is_test=True)
        test_size = len(testset)

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers
                                               )

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers
                                              )
    if model_name == "alexnet":
        net = AlexNet(num_classes=num_classes)
    elif model_name == "lenet5":
        net = LeNet5(num_classes=num_classes)
    elif model_name == "stn-alexnet":
        net = STNAlexNet(num_classes=num_classes)
    elif model_name == "stn-lenet5":
        net = LeNet5STN(num_classes=num_classes)
    elif model_name == "capsnet":
        net = CapsuleNet(num_classes=num_classes)
    elif model_name == "convneta":
        net = ConvNetA(num_classes=num_classes)
    elif model_name == "convnetb":
        net = ConvNetB(num_classes=num_classes)
    elif model_name == "convnetc":
        net = ConvNetC(num_classes=num_classes)
    elif model_name == "convnetd":
        net = ConvNetD(num_classes=num_classes)
    elif model_name == "convnete":
        net = ConvNetE(num_classes=num_classes)
    elif model_name == "convnetf":
        net = ConvNetF(num_classes=num_classes)
    elif model_name == "convnetg":
        net = ConvNetG(num_classes=num_classes)
    elif model_name == "convneth":
        net = ConvNetH(num_classes=num_classes)
    elif model_name == "convneti":
        net = ConvNetI(num_classes=num_classes)
    elif model_name == "convnetj":
        net = ConvNetJ(num_classes=num_classes)
    elif model_name == "convnetk":
        net = ConvNetK(num_classes=num_classes)
    elif model_name == "convnetl":
        net = ConvNetL(num_classes=num_classes)
    elif model_name == "convnetm":
        net = ConvNetM(num_classes=num_classes)
    elif model_name == "convnetn":
        net = ConvNetN(num_classes=num_classes)
    elif model_name == "resnet18":
        net = models.resnet18(pretrained=False, num_classes=num_classes)

    print(net)

    if torch.cuda.is_available():
        net = net.cuda()

    if model_name == "capsnet":
        criterion = CapsuleLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=l2_reg
                          )

    if lr_decay:
        scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_acc = 0
    from_epoch = 0

    if os.path.exists(CHKPT_PATH):
        print("Checkpoint Found: {}".format(CHKPT_PATH))
        state = torch.load(CHKPT_PATH)
        net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        best_acc = state['best_accuracy']
        from_epoch = state['epoch']

    for epoch in range(from_epoch, num_epochs):
        #print("Epoch: {}/{}".format(epoch + 1, NUM_EPOCHS))
        epoch_loss = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            #print("Train \t Epoch: {}/{} \t Batch: {}/{}".format(epoch + 1,
            #                                            NUM_EPOCHS,
            #                                            i + 1,
            #                                            ceil(train_size / BATCH_SIZE)))
            inputs, labels = data
            inputs, labels = Variable(inputs).type(torch.FloatTensor),\
                             Variable(labels).type(torch.LongTensor)

            if model_name == "capsnet":
                inputs = augmentation(inputs)
                ground_truth =  torch.eye(num_classes).index_select(dim=0, index=labels)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            if model_name == "capsnet":
                classes, reconstructions = net(inputs, ground_truth)
                loss = criterion(inputs, ground_truth, classes, reconstructions)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
            if model_name != "capsnet":
                log_outputs = F.softmax(outputs, dim=1)
            else:
                log_outputs = classes
            pred = log_outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

        print("Epoch: {} \t Training Loss: {:.4f} \t Training Accuracy: {:.2f} \t {}/{}".format(
            epoch+1,
            epoch_loss/train_size,
            100 * correct/train_size,
            correct,
            train_size
        )
        )

        loss_train = epoch_loss
        correct_train = correct

        correct = 0
        test_loss = 0
        for i, data in enumerate(test_loader, 0):
            # print("Test \t Epoch: {}/{} \t Batch: {}/{}".format(epoch + 1,
            #                                             NUM_EPOCHS,
            #                                             i + 1,
            #                                             ceil(test_size / BATCH_SIZE)))
            inputs, labels = data
            inputs, labels = Variable(inputs).type(torch.FloatTensor), Variable(labels).type(torch.LongTensor)

            if model_name == "capsnet":
                inputs = augmentation(inputs)
                ground_truth =  torch.eye(num_classes).index_select(dim=0, index=labels)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            if model_name == "capsnet":
                classes, reconstructions = net(inputs)
                loss = criterion(inputs, ground_truth, classes, reconstructions)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            test_loss += loss.data[0]

            if model_name != "capsnet":
                log_outputs = F.softmax(outputs, dim=1)
            else:
                log_outputs = classes

            pred = log_outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
        print("Epoch: {} \t Testing Loss: {:.4f} \t Testing Accuracy: {:.2f} \t {}/{}".format(
            epoch+1,
            test_loss/test_size,
            100 * correct/test_size,
            correct,
            test_size
        )
        )
        if correct >= best_acc:
            if not os.path.exists("./models"):
                os.mkdir("./models")
            torch.save(net.state_dict(), "./models/model-{}-{}-{}-{}-val-acc-{:.2f}-train-{}-test-{}-epoch-{}.pb".format(
                model_name,
                dataset,
                hdf5_path,
                str(datetime.now()),
                100 * correct / test_size,
                trainset_dir.replace(" ", "_").replace("/", "_"),
                testset_dir.replace(" ", "_").replace("/", "_"),
                epoch+1
            )
                       )
        best_acc = max(best_acc, correct)

        # save checkpoint path
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_acc
        }
        torch.save(state, CHKPT_PATH)

        if lr_decay:
            # Note that step should be called after validate()
            scheduler.step(test_loss)

        job = get_current_job()
        job.meta['progress'] = {
            "epoch": int(epoch+1),
            "best_accuracy": float(best_acc/test_size),
            "testing": {
                "loss": float(test_loss / test_size),
                "accuracy": float(100 * correct / test_size),
                "correct": int(correct),
                "total": int(test_size)
            },
            "training": {
                "loss": float(loss_train / train_size),
                "accuracy": float(100 * correct_train / train_size),
                "correct": int(correct_train),
                "total": int(train_size)
            },
        }

        job.save_meta()
        with open("{}.npy".format(job.id), 'a') as f_handle:
            f_handle.write("{},{},{},{}\n".format(
                float(loss_train / train_size),
                float(100 * correct_train / train_size),
                float(test_loss / test_size),
                float(100 * correct / test_size)
            ))
        f_handle.close()

    print('Finished Training')

    print("")
    print("")


def wrap_train(
        q,
        model_name,
        dataset,
        num_classes,
        batch_size,
        is_transform,
        num_workers,
        lr_decay,
        l2_reg,
        hdf5_path,
        trainset_dir,
        testset_dir,
        convert_grey,
        learning_rate,
        num_epochs
):
    qj = q.enqueue(
        train_generic_model,
        args=(
            model_name,
            dataset,
            num_classes,
            batch_size,
            is_transform,
            num_workers,
            lr_decay,
            l2_reg,
            hdf5_path,
            trainset_dir,
            testset_dir,
            convert_grey,
            learning_rate,
            num_epochs,
        ),
        timeout="3000h"
        )
    return qj.id
