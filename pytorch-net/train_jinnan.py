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
from dataloader.jndataset import JinNanDataset
from loss.dice_loss import MultiDiceLoss, multi_dice_coef
from loss.focal_loss import FocalLoss
from loss.lovasz_loss import MultiLovaszLoss
from loss.combinedloss import CombinedLoss
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from net.unet.res_unet import Res34Unet, Res50Unet, Res101Unet
from net.PAN.PAN import PAN34, PAN50, PAN101
from tensorboardX import SummaryWriter


images_dir = "data/jinnan/restricted"
maskes_dir = "data/jinnan/mask"
images_list = "data/jinnan/cross_validation/train_1.txt"
checkpoint_dir = "checkpoint"
total_epochs = 150
batch_size = 8
lr = 0.001


def parse_args():
    parser = argparse.ArgumentParser(description="arguements of the training")
    
    parser.add_argument('--images_dir', dest='images_dir', help='Directory containing image and mask',
                        default=images_dir, type=str)
    parser.add_argument('--maskes_dir', dest='maskes_dir', help='Directory containing image and mask',
                        default=maskes_dir, type=str)
    parser.add_argument('--images_list', dest='images_list', help='Directory containing image and mask',
                        default=images_list, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint_dir', help='Directory to be saved checkpoint',
                        default=checkpoint_dir, type=str)
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

    return parser.parse_args()


def train():
    args = parse_args()

    writer = SummaryWriter(log_dir="log", comment="net")

    trainset = JinNanDataset(images_dir=args.images_dir,
                            maskes_dir=args.maskes_dir,
                            images_list=args.images_list,
                            transform=transforms.Compose([
                                RandomScaleCrop(550, 512),
                                RandomHorizontalFlip(),
                                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensor()
                            ]))
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)

    print("starting loading the net and model")
    net = Res34Unet(3, 6, pretrained=False)
    # net = PAN34(3, 6)
    # net = Res50Unet(3, 6)
    # net = PAN50(3, 6)

    writer.add_graph(net, (torch.rand(2, 3, 512, 512),))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        args.gpus = [int(x) for x in args.gpus.split(",")]
        net = nn.DataParallel(net, device_ids=args.gpus)

    net.to(device)

    # loss and optimizer
    # criterion = FocalLoss()
    # criterion = MultiLovaszLoss()
    #class_weight = torch.tensor([0.0067, 1.649, 1.0, 1.334, 0.646, 1.047]).to(device) # ignore the class of background
    #class_weight = torch.tensor([1.45, 42.2, 38.16, 40.64, 33.68, 38.59]).to(device)
    class_weight = torch.tensor([0.067, 1.649, 1.0, 1.334, 0.646, 1.047]).to(device)
    # criterion = nn.NLLLoss(weight=class_weight)
    # criterion = FocalLoss(weight=class_weight)
    criterion = MultiDiceLoss(weights=class_weight)
    #criterion = MultiDiceLoss()
    # criterion = CombinedLoss()


    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.99, weight_decay=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, 
    #                                 min_lr=0.00001)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

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
            
            dice_coef = multi_dice_coef(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print("epoch %2d, [%5d / %5d], lr: %5g, loss: %.3f, dice coef: %.5f" % 
                    (epoch, (i + 1) * args.batch, len(trainset), scheduler.get_lr()[0], running_loss / 10, dice_coef))
                # print("epoch %2d, [%5d / %5d], loss: %.3f" % 
                #     (epoch, (i + 1) * args.batch, len(trainset), running_loss / 10))
                running_loss = 0.0

        
        epoch_loss = epoch_loss / math.ceil(len(trainset) / args.batch)
        # scheduler.step(epoch_loss / math.ceil(len(trainset) / args.batch))

        writer.add_scalar("train", epoch_loss, epoch)
        # save model

        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": net.module.state_dict() if args.cuda else net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr": scheduler.get_lr()[0],
                        "loss": epoch_loss
                    },
                    os.path.join(args.checkpoint_dir, "epoch_{}.pth".format(epoch))
                )

    print("Finished training")
    writer.close()


if __name__ == "__main__":
    train()
