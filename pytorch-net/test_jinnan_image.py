"""test image"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from net.unet.res_unet import Res34Unet, Res50Unet, Res101Unet
from net.PAN.PAN import PAN34, PAN50, PAN101


checkpoint_path = "checkpoint/epoch_5.pth"
image_path = "data/jinnan/restricted/1.jpg"


def test(checkpoint_path, image_path):
    net = Res34Unet(3, 6)
    #net = PAN50(3, 6)
    net.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    with torch.no_grad():
        
        image = Image.open(image_path).resize((512, 512), Image.BILINEAR)
        image_np = np.array(image).copy()
        image = np.array(image).astype(np.float32)
        
        image /= 255.0
        image -= (0.485, 0.456, 0.406)
        image /= (0.229, 0.224, 0.225)
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float().to(device)

        predict_mask = net(image)
        predict_mask = torch.squeeze(predict_mask.cpu()).numpy()

        predict_mask = np.argmax(predict_mask, axis=0)# ont-hot convert to label

        for i in range(0, 6):
            print("%d:" % i, np.sum(predict_mask == i))

        mask = predict_mask * 20
        
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(image_np)
        plt.subplot(2, 2, 2)
        plt.imshow(mask)
        plt.show()


if __name__ == '__main__':
    test(checkpoint_path, image_path)


