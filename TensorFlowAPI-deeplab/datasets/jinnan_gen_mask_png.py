from pycocotools.mask import *
import pycocotools.mask as mask_util
import json
import numpy as np
import os
import os.path as osp
from PIL import Image
from random import randint

my_color_map = [
0,0,0,		# 0 background
128,0,0,	# 1 铁壳打火机
0,128,0,	# 2 黑钉打火机
128,128,0,	# 3 刀具
0,0,128,	# 4 电池电容
128,0,128	# 5 剪刀
]

def id2name(images_list_dict):
	"""
	return id name map dict:
	{
		"0":{"img_name":xxx,height:000,width:000},
		"1":{"img_name":xxx,height:000,width:000},
		...
	}
	"""
	rst_dict = dict()
	for img in images_list_dict:
		img_id = str(img["id"])
		rst_dict[img_id] = dict()
		rst_dict[img_id]["file_name"] = img["file_name"]
		rst_dict[img_id]["height"] = img["height"]
		rst_dict[img_id]["width"] = img["width"]
	return rst_dict

def getImagesegmsFromImageIds(annotations):
	"""
	para: annotations is annotations list
	return: id dict
	{
		img_id:[(cate_id,segm),(cate_id,segm),...,(cate_id,segm)]
	}
	"""
	rst_dict = dict()
	stastic_rst = dict()

	for img_segm in annotations:
		img_id = str(img_segm["image_id"])
		if img_id not in rst_dict.keys():
			rst_dict[img_id] = []

		iscrowd = img_segm["iscrowd"]
		segm = img_segm["segmentation"]
		category_id = img_segm["category_id"]
		if str(category_id) not in stastic_rst.keys():
			stastic_rst[str(category_id)] = 0
		stastic_rst[str(category_id)] += 1
		if iscrowd == 1:
			print("segm is crowd")
		else:
			"""
			segm format:
			[[xx,xx,xx,...,xxx],[...]]
			"""
			rst_dict[img_id].append((category_id,segm))
	return rst_dict,stastic_rst

def segm2bimask(category_id,segms):
	pass



def rel2bimask(coco_json,save_png_path):
	json_val = json.load(open(coco_json,'r',encoding='utf-8'))
	categories = json_val["categories"]
	images = json_val["images"]
	annotations = json_val["annotations"]
	id_name = id2name(images)

	img_segm_info,stastic = getImagesegmsFromImageIds(annotations)
	for img_id,img_info in img_segm_info.items():
		img_name = id_name[str(img_id)]["file_name"]
		name,img_type = img_name.split(".")
		h,w = id_name[str(img_id)]["height"],id_name[str(img_id)]["width"]
		mymask = np.zeros([h,w])
		# print("raw-mask shpae",mymask.shape)
		for id_segm_tupplu in img_info:
			category_id = id_segm_tupplu[0]
			segm = id_segm_tupplu[1]
			
			rles = mask_util.frPyObjects(segm,h,w)
			mask = mask_util.decode(rles)

			reshape_mask = mask[:,:,0]
			rows,clos = reshape_mask.shape

			# with overlap
			# idx = np.where(reshape_mask == 1,category_id,0)
			# mymask += idx

			idx_temp = np.where(reshape_mask*mymask >0,0,1)
			idx = idx_temp * np.where(reshape_mask == 1,category_id,0)
			
			mymask += idx

		my_img = Image.fromarray(mymask).convert("P")	
		# my_img.putpalette(my_color_map)
		my_img.save(osp.join(save_png_path,name+".png"))
		

	print(len(id_name.keys()))
	print(len(img_segm_info.keys()))
	diff = set(id_name.keys() - img_segm_info.keys())

	print("unlabeed images: \n",diff)
	print("object distribute: \n",stastic)

if __name__ == '__main__':
	json_path = "jinnan/train_restriction.json"
	save_png = "jinnan/mask"
	if not osp.exists(save_png):
		os.makedirs(save_png)
	rel2bimask(json_path,save_png)
	"""
	{'1255', '1735', '1307', '1900'}
	{'1': 1068, '4': 5254, '2': 4359, '3': 1895, '5': 1825}
	"""
	print("done!")