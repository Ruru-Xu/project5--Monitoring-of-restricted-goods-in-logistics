"""
the dataset for jinnan
"""
import os
from torch.utils.data import Dataset
from PIL import Image


class JinNanDataset(Dataset):
    """docstring for JinNanDataset"""

    def __init__(self, images_dir, maskes_dir, images_list, transform):
        self.images_dir = images_dir
        self.maskes_dir = maskes_dir
        self.transform = transform
        # self.images_list = [mask_name.split(".")[0] \
        #                     for mask_name in os.listdir(self.maskes_dir)]
        self.images_list = []
        with open(images_list, "r") as f:
        	line = f.readline()
        	while line:
        		self.images_list.append(line.split(".")[0])
        		line = f.readline()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images_list[idx] + ".jpg")
        mask_path = os.path.join(self.maskes_dir, self.images_list[idx] + '.png')
        image = Image.open(image_path)
        mask =  Image.open(mask_path)
        sample = {"image": image, "mask": mask}
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    import sys
    sys.path.append("../")
    import numpy as np
    import matplotlib.pyplot as plt
    from transform import Normalize, RandomScaleCrop, RandomGaussianBlur, RandomHorizontalFlip, ToTensor
    from torchvision import transforms
    from torch.utils.data import DataLoader


    trainset = JinNanDataset(images_dir="../data/jinnan/restricted",
                            maskes_dir="../data/jinnan/mask",
                            images_list="../data/jinnan/cross_validation/train_1.txt",
                            transform=transforms.Compose([
                            	# RandomGaussianBlur(),
                                RandomScaleCrop(550, 512),
                                RandomHorizontalFlip(),
                                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensor()
                            ]))
    print(len(trainset))
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    for i, sample in enumerate(trainloader):
        for j in range(sample["image"].size()[0]):
            image = sample["image"][j].numpy()
            mask = sample["mask"][j].numpy()
            image = image.transpose([1, 2, 0])
            image *= (0.229, 0.224, 0.225)
            image += (0.485, 0.456, 0.406)
            image = image * 255
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)

            mask = mask * 20

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.subplot(2, 2, 2)
            plt.imshow(mask)
        if i == 0:
            break
    plt.show()
    
        