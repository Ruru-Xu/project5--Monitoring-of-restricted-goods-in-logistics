"""test image"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from net.unet.res_unet import Res34Unet, Res50Unet, Res101Unet
from net.PAN.PAN import PAN34, PAN50, PAN101


def test(model_path, test_dir, prediction_dir=None):

    net = PAN50(3, 6)
    net.load_state_dict(torch.load(model_path)["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    with torch.no_grad():

        for image_name in os.listdir(test_dir):
            print(image_name)
            image_id = image_name.split(".")[0]
            image_path = os.path.join(test_dir, image_name)
            image = Image.open(image_path)
            original_size = image.size
            image = image.resize((512, 512), Image.BILINEAR)
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

            for i in range(1, 6):
                mask_np = np.zeros((predict_mask.shape[0], predict_mask.shape[1]), np.uint8) 
                mask_np[predict_mask == i] = 1
                mask = Image.fromarray(mask_np)
                mask = mask.resize(original_size, Image.NEAREST)
                mask_np = np.array(mask)
                np.save(os.path.join(prediction_dir, "%d_%d" % (int(image_id), i)), mask_np)


if __name__ == '__main__':
    import os
    model_path = "model/epoch_58.pth"
    test_dir = "data/jinnan/test"
    prediction_dir = "prediction"

    test(model_path, test_dir, prediction_dir)
    


