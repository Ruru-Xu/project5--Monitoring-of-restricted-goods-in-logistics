import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from dataloader.transform import FixScaleCrop, Normalize, ToTensor
from dataloader.vocdataset import VOCDataset
from dataloader.jndataset import JinNanDataset
from net.unet.res_unet import Res34Unet, Res50Unet, Res101Unet
from net.PAN.PAN import PAN34, PAN50, PAN101
from utils.metrics import Evaluator

images_dir = "data/jinnan/restricted"
maskes_dir = "data/jinnan/mask"
images_list = "data/jinnan/cross_validation/val_1.txt"
batch_size = 2
num_classes = 6

def parse_args():
    parser = argparse.ArgumentParser(description="arguements of the training")
    
    parser.add_argument('--images_dir', dest='images_dir', help='Directory containing image and mask',
                        default=images_dir, type=str)
    parser.add_argument('--maskes_dir', dest='maskes_dir', help='Directory containing image and mask',
                        default=maskes_dir, type=str)
    parser.add_argument('--images_list', dest='images_list', help='Directory containing image and mask',
                        default=images_list, type=str)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='path of checkpoint', type=str)
    parser.add_argument('--batch', dest='batch', help='batch for training',
                        default=batch_size, type=int)
    parser.add_argument('--num_classes', dest='num_classes', help='batch for training',
                        default=num_classes, type=int)
    parser.add_argument('--cuda', dest='cuda', help='whether use gpu or not', 
                        default=True)
    parser.add_argument('--gpus', dest='gpus', help='such as --gpus 0,1', default="0")

    return parser.parse_args()


def eval():
    args = parse_args()

    valset = JinNanDataset(images_dir=args.images_dir,
                            maskes_dir=args.maskes_dir,
                            images_list=args.images_list,
                            transform=transforms.Compose([
                                FixScaleCrop(512),
                                Normalize(mean=(0.485, 0.456, 0.406), 
                                        std=(0.229, 0.224, 0.225)),
                                ToTensor()
                            ]))
    valloader = DataLoader(valset, batch_size=args.batch, shuffle=True, num_workers=0)

    print("starting loading the net and model")
    net = Res34Unet(3, 6)
    # net = PAN34(3, 6)
    #net = PAN50(3, 6)

    net.load_state_dict(torch.load(args.checkpoint_path)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda:
        args.gpus = [int(x) for x in args.gpus.split(",")]
        net = nn.DataParallel(net, device_ids=args.gpus)
    net.to(device)
    net.eval()
    print("finishing loading the net and model")

    print("start validating")
    evaluator = Evaluator(args.num_classes)
    evaluator.reset()
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            print("calculate %d batch" % (i+1))
            # get the inputs
            inputs = data["image"].to(device)
            labels = data["mask"]
            
            # forward
            outputs = net(inputs)
            outputs = outputs.cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            # add batch
            evaluator.add_batch(labels.numpy(), outputs)
    ACC = evaluator.pixel_accuracy_class()
    MIoU = evaluator.mean_intersection_over_union()
    print("pixel accuracy class:", ACC)
    print("mean intersection over union:", MIoU)
    print("Finished validating")


if __name__ == "__main__":
    eval()
