"""
the dataset for pascal voc
Reference:
    https://github.com/jfzhang95/pytorch-deeplab-xception
"""
import os
from torch.utils.data import Dataset
from PIL import Image
from utils.utils import encode_segmap


class VOCDataset(Dataset):
    """Pacal VOC Dateset"""

    def __init__(self, base_dir, split='train', transform=None):
        """
        Args:
             base_dir(str): Directory with all the images ande mask
             split(str): one of train and val or trainval
             transform (callable): transform to be applied on a sample.
        """
        self.image_dir = os.path.join(base_dir, "JPEGImages")
        self.mark_dir = os.path.join(base_dir, "SegmentationClass")
        self.split = split
        assert os.path.exists(self.image_dir) and os.path.exists(self.mark_dir)

        self.image_list = []
        with open(os.path.join(base_dir, self.split + ".txt"), "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            self.image_list.append(line)

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, "{}.jpg".format(self.image_list[idx]))
        image = Image.open(image_path)
        mask_path = os.path.join(self.mark_dir, "{}.png".format(self.image_list[idx]))
        mask = Image.open(mask_path).convert("RGB")
        mask = encode_segmap(mask) # encode segmentation label images as pascal classes
        mask = Image.fromarray(mask) # convert numpy array to Image array
        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    """
    test VOCDataset
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from transform import Normalize, RandomScaleCrop, RandomGaussianBlur, RandomHorizontalFlip, ToTensor
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from utils import utils


    trainset = VOCDataset("data/train",  
            transform=transforms.Compose(
                [
                    RandomScaleCrop(550, 512),
                    RandomHorizontalFlip(),
                    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensor()
                ]
                )
        )
    dataloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader):
        for j in range(sample["image"].size()[0]):
            image = sample["image"][j].numpy()
            mask = sample["mask"][j].numpy()
            image = image.transpose([1, 2, 0])
            image *= (0.229, 0.224, 0.225)
            image += (0.485, 0.456, 0.406)
            image = image * 255
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)

            mask = utils.decode_segmap(mask)

            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(image)
            plt.subplot(2, 2, 2)
            plt.imshow(mask)
        if i == 0:
            break
    plt.show()



