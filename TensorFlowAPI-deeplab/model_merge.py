import os
import numpy as np

npy_dir_1 = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan-4-16/prediction-4-15-1"
npy_dir_2 = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/prediction-4-21-0"
saved_npy_dir = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/prediction-4-23-0"

test_dir = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/test"

for image_name in os.listdir(test_dir):
    img_id = image_name.split(".")[0]
    print(img_id)
    for i in range(1, 6):
        name = "{}_{}.npy".format(img_id, i)
        npy_path_1 = os.path.join(npy_dir_1, name)
        npy_path_2 = os.path.join(npy_dir_2, name)
        np_1 = np.load(npy_path_1)
        np_2 = np.load(npy_path_2)
        np_2[np_1 != np_2] = 0
        np.save(os.path.join(saved_npy_dir, "{}_{}".format(img_id, i)), np_2)

