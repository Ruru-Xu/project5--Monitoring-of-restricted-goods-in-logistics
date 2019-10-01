import os
with open("jinnan/index_aug/train_aug.txt", "w") as f:
	for image in os.listdir("jinnan/mask"):
		if image.find("_") != -1:
			f.write(image.split(".")[0] + "\n")

