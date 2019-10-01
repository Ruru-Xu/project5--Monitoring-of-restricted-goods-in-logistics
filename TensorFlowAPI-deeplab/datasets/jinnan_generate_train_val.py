import os
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val(train_dir, index):
	"""split train val
	Args:
		train_dir(str): directory of training dataset
		index(str): directory to save the result of spliting
	"""
	if not os.path.exists(index):
		os.makedirs(index)
		
	images_list = [image_name.split(".")[0] for image_name in os.listdir(train_dir)]
	with open(os.path.join(index, "trainval.txt"), "w") as f:
			for i_l in images_list:
				f.write(i_l + "\n")

	train_list, val_list = train_test_split(images_list, test_size=0.2, random_state=123)
	with open(os.path.join(index, "train.txt"), "w") as f:
		for t_l in train_list:
			f.write(t_l + "\n")

	with open(os.path.join(index, "val.txt"), "w") as f:
		for v_l in val_list:
			f.write(v_l + "\n")



if __name__ == '__main__':
	train_dir = "jinnan/mask"
	index = "jinnan/index"
	split_train_val(train_dir, index)