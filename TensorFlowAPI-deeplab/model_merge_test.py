import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

image_dir = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/test"
prediction_dir = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/prediction-4-23-0"

model_0_path = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/model_jinnan/frozen_inference_graph-4-15-1.pb"
model_1_path = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/model_jinnan/frozen_inference_graph-4-21-0.pb"

img = tf.placeholder(tf.uint8, [1, None, None, 3], name="img")
with open(model_0_path, "rb") as f:
    graph_def_0 = tf.GraphDef()
    graph_def_0.ParseFromString(f.read())
    output_0 = tf.import_graph_def(graph_def_0, input_map={"ImageTensor:0": img}, return_elements=["SemanticPredictions:0"])

with open(model_1_path, "rb") as f:
    graph_def_1 = tf.GraphDef()
    graph_def_1.ParseFromString(f.read())
    output_1 = tf.import_graph_def(graph_def_1, input_map={"ImageTensor:0": img}, return_elements=["SemanticPredictions:0"])

sess = tf.Session()

for i, image_name in enumerate(os.listdir(image_dir)):
    print(i)
    image_id = image_name.split(".")[0]
    image_path = os.path.join(image_dir, image_name)
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0).astype(np.uint8)
    # result = sess.run(output, feed_dict = {img:image})
    result_0, result_1 = sess.run([output_0, output_1], feed_dict = {img:image})

    
    result_0 = np.squeeze(result_0[0])
    result_1 = np.squeeze(result_1[0])
    # merge
    result_1[result_0 != result_1] = 0
    for i in range(1, 6):
        mask_np = np.zeros((result_1.shape[0], result_1.shape[1]), np.uint8) 
        mask_np[result_1 == i] = 1
        np.save(os.path.join(prediction_dir, "%d_%d" % (int(image_id), i)), mask_np)
