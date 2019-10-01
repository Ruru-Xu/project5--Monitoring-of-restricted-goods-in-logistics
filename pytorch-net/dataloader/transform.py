"""
Reference:
    https://github.com/jfzhang95/pytorch-deeplab-xception
"""
import random
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        if random.random() < self.p:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {"image": image, "mask": mask}


# class Rescale(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
#         w, h = image.size
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         img = image.resize((new_h, new_w), Image.BILINEAR)
#         label = mask.resize((new_h, new_w), Image.NEAREST)
#         return {'image': img, 'mask': label}



# class RandomCrop(object):
#     """Crop randomly the image in a sample.

#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size

#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']

#         h, w = image.size
#         new_h, new_w = self.output_size

#         top = random.randint(0, h - new_h)
#         left = random.randint(0, w - new_w)

#         image = image.crop((left, top, left + new_w, top + new_h))
#         mask = mask.crop((left, top, left + new_w, top + new_h))
#         return {'image': image, 'mask': mask}


class RandomScaleCrop(object):

    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        # random scale 
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = image.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': image, 'mask': mask}


class FixScaleCrop(object):
    """Scale and Crop the image in a sample.

    Args:
        crop_size (int): Desired output size. 
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        w, h = image.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = image.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': image, 'mask': mask}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample : Image and mask to be flipped.
        Returns:
            sample: Randomly flipped image and mask.
        """
        if random.random() < self.p:
            if isinstance(sample["image"], Image.Image):
                sample["image"] = sample["image"].transpose(Image.FLIP_LEFT_RIGHT)
                sample["mask"] = sample["mask"].transpose(Image.FLIP_LEFT_RIGHT)
            else:
                TypeError('img should be PIL Image. Got {}'.format(type(sample["image"])))
        return sample

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        return {"image": image, "mask": mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.uint8)
        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).long()}



