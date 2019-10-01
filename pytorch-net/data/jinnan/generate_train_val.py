import os
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val(train_dir, cross_val_dir):
	"""split train val
	Args:
		train_dir(str): directory of training dataset
		cross_val_dir(str): directory to save the result of spliting
	"""
	if not os.path.exists(cross_val_dir):
		os.makedirs(cross_val_dir)
		
	images_list = [image_name for image_name in os.listdir(train_dir)]
	with open(os.path.join(cross_val_dir, "train.txt"), "w") as f:
			for i_l in images_list:
				f.write(i_l + "\n")

	random_stat = [123, 321, 156, 456, 987]
	for i, r_s in enumerate(random_stat):
		np.random.seed(r_s)
		train_list, val_list = train_test_split(images_list, test_size=0.2, random_state=r_s)
		with open(os.path.join(cross_val_dir, "train_%d.txt" % (i + 1)), "w") as f:
			for t_l in train_list:
				f.write(t_l + "\n")

		with open(os.path.join(cross_val_dir, "val_%d.txt" % (i + 1)), "w") as f:
			for v_l in val_list:
				f.write(v_l + "\n")



if __name__ == '__main__':
	train_dir = "restricted"
	cross_val_dir = "cross_validation"
	split_train_val(train_dir, cross_val_dir)