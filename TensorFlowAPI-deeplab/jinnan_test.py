import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

image_dir = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/test-4-24"
prediction_dir = "/home/imc/XR/models/TensorFlow-Models/research/deeplab/datasets/jinnan/prediction-4-24-0"

img = tf.placeholder(tf.uint8, [1, None, None, 3], name="img")
with open("/home/imc/XR/models/TensorFlow-Models/research/deeplab/model_jinnan/frozen_inference_graph-4-21-0.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    output = tf.import_graph_def(graph_def, input_map={"ImageTensor:0": img}, return_elements=["SemanticPredictions:0"])
sess = tf.Session()

for image_name in os.listdir(image_dir):
    image_id = image_name.split(".")[0]
    image_path = os.path.join(image_dir, image_name)
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0).astype(np.uint8)
    result = sess.run(output, feed_dict = {img:image})
    
    result = np.squeeze(result[0])

    for i in range(1, 6):
        mask_np = np.zeros((result.shape[0], result.shape[1]), np.uint8) 
        mask_np[result == i] = 1
        np.save(os.path.join(prediction_dir, "%d_%d" % (int(image_id), i)), mask_np)
