"""convert polygon to mask"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pycocotools.mask as maskutil


def pologon2mask(json_path, mask_dir, num_classes=5):
	"""convert pologon to mask
	Args:
		json_path(str): path of coco mask 
		mask_dir(str): directory of mask 
		num_classes(int): number of classes
	"""
	if not os.path.exists(mask_dir):
		os.makedirs(mask_dir)

	with open(json_path, "r", encoding="utf-8") as f:
		json_data = json.load(f)

	images = json_data["images"]
	for image in json_data["images"]:
		image_id = image["id"]
		image_name = image["file_name"]
		print("converting %s" % image_name)
		width = image["width"]
		height = image["height"]
		annotation_ids = [annotation["id"] for annotation in json_data["annotations"]\
							if image_id == annotation["image_id"]]
		mask_np = np.zeros((height, width), np.uint8) # (height, width)
		for annot_id in annotation_ids:
			try:
				segmentation = json_data["annotations"][annot_id]["segmentation"][0]
			except IndexError as e:
				break

			compactedRLE = maskutil.frPyObjects([segmentation], height, width)
			mask = maskutil.decode(compactedRLE)
			category_id = json_data["annotations"][annot_id]["category_id"]
			mask = np.reshape(mask, (height, width))

			mask_np[mask == 1] = category_id * 20
		mask_im = Image.fromarray(mask_np)
		mask_im.save(os.path.join(mask_dir, str(image_id) + ".png"))


def vis_mask(mask_dir):
	"""visaulize mask
	Args:
		mask_dir(str): path of mask
	"""
	for file_name in os.listdir(mask_dir):
		mask_path  = os.path.join(mask_dir, file_name)
		mask_np = np.load(mask_path)
		height, width, num_classes = mask_np.shape
		# (width, height, channel) to (channel, height, width)
		mask_np = mask_np.transpose((2, 0, 1)) 
		mask = np.zeros((height, width), np.uint8)
		for i in range(num_classes):
			mask += mask_np[i][:][:] * (i + 1) * 20
		print(file_name, mask.shape)
		plt.imshow(mask)
		plt.show()


if __name__ == '__main__':
	json_path = "train_restriction.json"
	mask_dir = "mask"
	pologon2mask(json_path, mask_dir)
	# vis_mask(mask_dir)