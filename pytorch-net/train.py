import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from dataloader.transform import RandomScaleCrop, RandomHorizontalFlip, Normalize, ToTensor
from dataloader.vocdataset import VOCDataset
from loss.focal_loss import FocalLoss
from loss.lovasz_loss import MultiLovaszLoss
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from net.unet.res_unet import Res34Unet
from net.PAN.PAN import PAN

base_dir = "data/train"
model_dir = "model"
total_epochs = 180
batch_size = 16
lr = 0.001


def parse_args(base_dir, model_dir, total_epochs, batch_size, lr):
    """
    parse arguements to the training
    Args:
        base_dir(str): Directory containing images and mask
        model_dir(str): Directory to be saved model
        total_epochs(int): total epochs for training
        batch_size(int): total epochs for training
        lr(float): Learning rate
    return
    """
    parser = argparse.ArgumentParser(description="arguements of the training")
    
    parser.add_argument('--base', dest='base_dir', help='Directory containing image and mask',
                        default=base_dir, type=str)
    parser.add_argument('--model', dest='model', help='Directory to be saved model',
                        default=model_dir, type=str)
    parser.add_argument('--epoch', dest='epoch', help='total epochs for training',
                        default=total_epochs, type=int)
    parser.add_argument('--batch', dest='batch', help='batch for training',
                        default=batch_size, type=int)
    parser.add_argument('--lr', dest='lr', help='Learning rate', default=lr)
    parser.add_argument('--cuda', dest='cuda', help='whether use gpu or not', 
                        default=True)
    parser.add_argument('--gpus', dest='gpus', help='such as --gpus 0,1', default="0")
    parser.add_argument('--resume_from', dest='resume_from', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--eval', dest='eval', help='eval validate dataset', default=False)

    return parser.parse_args()


def train():
    args = parse_args(base_dir, model_dir, total_epochs, batch_size, lr)

    trainset = VOCDataset(base_dir=args.base_dir, split="train",
                            transform=transforms.Compose([
                                RandomScaleCrop(550, 512),
                                RandomHorizontalFlip(),
                                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensor()
                            ]))
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4)

    print("starting loading the net and model")
    # net = Res34Unet(3, 21)
    net = PAN(3, 21)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        args.gpus = [int(x) for x in args.gpus.split(",")]
        net = nn.DataParallel(net, device_ids=args.gpus)

    net.to(device)

    # loss and optimizer
    criterion = FocalLoss()
    # criterion = MultiLovaszLoss()
    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, 
    #                                 min_lr=0.00001)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    start_epoch = 1
    # Resuming training
    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume_from)
        #args.start_epoch = checkpoint['epoch']
        if args.cuda:
            net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint['model_state_dict'])
        print("resuming training from {}, epoch:{}"\
        	.format(args.resume_from, checkpoint['epoch']))
        start_epoch = checkpoint['epoch'] + 1
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("finishing loading the net and model")

    print("start training")
    for epoch in range(start_epoch, args.epoch + start_epoch):
        scheduler.step()
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data["image"].to(device), data["mask"].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print("epoch %2d, [%5d / %5d], lr: %5g, loss: %.3f" % 
                    (epoch, (i + 1) * args.batch, len(trainset), scheduler.get_lr()[0], running_loss / 10))
                # print("epoch %2d, [%5d / %5d], loss: %.5f" % 
                #     (epoch, (i + 1) * args.batch, len(trainset), running_loss / 10))
                running_loss = 0.0

        # scheduler.step(epoch_loss / math.ceil(len(trainset) / args.batch))

        # save model
        torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": net.module.state_dict() if args.cuda else net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "lr": scheduler.get_lr()[0]
                    },
                    os.path.join(model_dir, "epoch_{}.pth".format(epoch))
                )
    print("Finished training")


if __name__ == "__main__":
    train()
