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
from net.unet.res_unet import Res34Unet
from net.PAN.PAN import PAN
from utils.metrics import Evaluator


def parse_args():
    """parse arguements to the validating"""

    parser = argparse.ArgumentParser(description="arguements of the training")
    
    parser.add_argument('--base', dest='base_dir', 
                        help='Directory containing image and mask',
                        default="data/train", type=str)
    parser.add_argument('--model_path', dest='model_path',
                        help='Directory to be saved model', type=str)
    parser.add_argument('--batch', dest='batch', help='batch for training',
                        default=2, type=int)
    parser.add_argument('--num_classes', dest='num_classes', 
                        help='number of classes',
                        default=21, type=int)
    parser.add_argument('--cuda', dest='cuda', help='whether use gpu or not', 
                        default=True)
    parser.add_argument('--gpus', dest='gpus', help='such as --gpus 0,1', 
                        default="0")

    return parser.parse_args()


def train():
    args = parse_args()

    val_set = VOCDataset(base_dir=args.base_dir, split="val",
                            transform=transforms.Compose([
                                FixScaleCrop(512),
                                Normalize(mean=(0.485, 0.456, 0.406), 
                                        std=(0.229, 0.224, 0.225)),
                                ToTensor()
                            ]))
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=True, 
                            num_workers=4)

    print("starting loading the net and model")
    #net = Res34Unet(3, args.num_classes, True)
    net = PAN(3, 21)
    net.load_state_dict(torch.load(args.model_path)["model_state_dict"])
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
        for i, data in enumerate(val_loader, 0):
            print("calculate %d batch" % (i+1))
            # get the inputs
            inputs = data["image"].to(device)
            labels = data["mask"]
            
            # forward
            outputs = net(inputs)
            outputs = outputs.cpu().numpy().astype(np.uint8)
            outputs = np.argmax(outputs, axis=1)

            evaluator.add_batch(labels.numpy(), outputs)
    ACC = evaluator.pixel_accuracy_class()
    MIoU = evaluator.mean_intersection_over_union()
    print("pixel accuracy class:", ACC)
    print("mean intersection over union:", MIoU)
    print("Finished validating")


if __name__ == "__main__":
    train()
