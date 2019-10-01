"""test image"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from net.unet.res_unet import Res34Unet
from net.PAN.PAN import PAN
from utils.utils import decode_segmap


model_path = "model/epoch_195.pth"
image_path = "data/test/2.jpg"


def test(model_path, image_path):
    #net = Res34Unet(3, 21, False)
    net = PAN(3, 21)
    net.load_state_dict(torch.load(model_path)["model_state_dict"])
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

        np.set_printoptions(threshold=np.inf)

        predict_mask = net(image)
        predict_mask = torch.squeeze(predict_mask.cpu()).numpy()

        predict_mask = np.argmax(predict_mask, axis=0)# ont-hot convert to label

        for i in range(0, 21):
            print("%d:" % i, np.sum(predict_mask == i))

        mask = decode_segmap(predict_mask)
        
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(image_np)
        plt.subplot(2, 2, 2)
        plt.imshow(mask)
        plt.show()


if __name__ == '__main__':
    test(model_path, image_path)


