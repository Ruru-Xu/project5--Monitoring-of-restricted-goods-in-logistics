"""data augment"""
import os
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def random_bright_flip(image_dir, mask_dir, image_name):
    """亮度"""
    if random.random() < 0.5:
        image_path = os.path.join(image_dir, image_name + ".jpg")
        mask_path = os.path.join(mask_dir, image_name + ".png")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        brightness = float("%.2f" % random.uniform(0.8, 1.2))
        enh_bri = ImageEnhance.Brightness(image)
        image_enh_bri = enh_bri.enhance(brightness)
        if random.random() < 0.5:
            image_enh_bri = image_enh_bri.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image_enh_bri.save(os.path.join(image_dir, image_name + "_bf.jpg"))
        mask.save(os.path.join(mask_dir, image_name + "_bf.png"))


def random_contrast_flip(image_dir, mask_dir, image_name):
    """对比度"""
    if random.random() < 0.5:
        image_path = os.path.join(image_dir, image_name + ".jpg")
        mask_path = os.path.join(mask_dir, image_name + ".png")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        contrast = float("%.2f" % random.uniform(0.8, 1.5))
        enh_con = ImageEnhance.Contrast(image)
        image_enh_con = enh_con.enhance(contrast)
        if random.random() < 0.5:
            image_enh_con = image_enh_con.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image_enh_con.save(os.path.join(image_dir, image_name + "_cf.jpg"))
        mask.save(os.path.join(mask_dir, image_name + "_cf.png"))

def random_sharp_flip(image_dir, mask_dir, image_name):
    """锐度"""
    if random.random() < 0.5:
        image_path = os.path.join(image_dir, image_name + ".jpg")
        mask_path = os.path.join(mask_dir, image_name + ".png")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        sharpness = float("%.2f" % random.uniform(0.8, 1.5))
        enh_sha = ImageEnhance.Sharpness(image)
        image_enh_sha = enh_sha.enhance(sharpness)
        if random.random() < 0.5:
            image_enh_sha = image_enh_sha.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image_enh_sha.save(os.path.join(image_dir, image_name + "_sf.jpg"))
        mask.save(os.path.join(mask_dir, image_name + "_sf.png"))

def random_horizontal_flip(image_dir, mask_dir, image_name):
    """以一定概率左右旋转"""
    if random.random() < 0.5:
        image_path = os.path.join(image_dir, image_name + ".jpg")
        mask_path = os.path.join(mask_dir, image_name + ".png")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image.save(os.path.join(image_dir, image_name + "_f.jpg"))
        mask.save(os.path.join(mask_dir, image_name + "_f.png"))
      

def data_augment(image_dir, mask_dir, image_list):
    with open(image_list, "r") as f:
        line = f.readline().strip()
        while line:
            image_path = os.path.join(image_dir, line + ".jpg")
            mask_path = os.path.join(mask_dir, line + ".png")
            random_bright_flip(image_dir, mask_dir, line)
            random_contrast_flip(image_dir, mask_dir, line)
            random_sharp_flip(image_dir, mask_dir, line)
            random_horizontal_flip(image_dir, mask_dir, line)
            line = f.readline().strip()




if __name__ == '__main__':
    image_dir = "jinnan/image"
    mask_dir = "jinnan/mask"
    image_list = "jinnan/index/trainval.txt"
    data_augment(image_dir, mask_dir, image_list)

