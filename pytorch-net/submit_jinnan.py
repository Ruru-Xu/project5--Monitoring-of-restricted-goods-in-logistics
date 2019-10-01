import os
import json
import numpy as np
# import pycocotools.mask as maskutil
from pycocotools.mask import encode


def make_submit(image_name,preds):
    '''
    Convert the prediction of each image to the required submit format
    :param image_name: image file name
    :param preds: 5 class prediction mask in numpy array
    :return:
    '''

    submit = dict()
    submit['image_name'] = image_name
    submit['size'] = (preds.shape[1],preds.shape[2])  #(height,width)
    submit['mask'] = dict()

    for cls_id in range(0,5):      # 5 classes in this competition

        mask = preds[cls_id,:,:]
        cls_id_str = str(cls_id+1)   # class index from 1 to 5,convert to str
        fortran_mask = np.asfortranarray(mask)
        # rle = maskutil.encode(fortran_mask) #encode the mask into rle, for detail see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        rle = encode(fortran_mask)
        submit['mask'][cls_id_str] = rle

    return submit



def dump_2_json(submits,save_p):
    '''

    :param submits: submits dict
    :param save_p: json dst save path
    :return:
    '''
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(save_p, 'w', encoding='utf-8');
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()




if __name__=="__main__":

    test_dir = "data/jinnan/test"
    prediction_dir = "./prediction"
    json_p = "result/submission_4_5_PAN50.json"

    submits_dict = dict()
    for image_name in os.listdir(test_dir):
        img_id = image_name.split(".")[0]
        preds = []
        for cls_id in range(1,6): # 5 classes in this competition
            cls_pred_name = "%d_%d.npy" % (int(img_id), cls_id)
            pred_p = os.path.join(prediction_dir, cls_pred_name)
            pred = np.load(pred_p)
            preds.append(pred)

        preds_np = np.array(preds) #fake prediction
        submit = make_submit(image_name,preds_np)
        submits_dict[image_name] = submit

    dump_2_json(submits_dict,json_p)

