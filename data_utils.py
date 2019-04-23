
import json
import cv2
import numpy as np
import os
from fn import vis_frame as vis_frame


annotation_file = './data/pose_walking/alphapose-results.json'
img_path = './data/split_walking/'
results = json.load(open(annotation_file, 'r'))
out_path = './data/debug_input'



'''
A for loop that reads a list of keypoints and rearrange in a directory whose keys are named as the input frames.
This is done for visualization and more clear annotation format.
'''
anns = {}
last_image_name = ' '
for i in range(len(results)):
    imgpath = results[i]['image_id']
    if last_image_name != imgpath:
        anns[imgpath] = []
        anns[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
    else:
        anns[imgpath].append({'keypoints':results[i]['keypoints'],'scores':results[i]['score']})
    last_image_name = imgpath



limgs = os.listdir(img_path)
limgs.sort()


for name in limgs:
    p = os.path.join(img_path, name)
    img = cv2.imread(p)
    shape = img.shape
    # create a black Image
    empty = np.zeros(shape, dtype=np.uint8)

    kd = name.split('.png')[0]
    key_img = vis_frame(empty, anns[kd])

    # concatenate images for visualization
    img_cat = np.concatenate((img, key_img), axis=1)

    name_out = os.path.join(out_path, name)
    cv2.imwrite(name_out, img_cat)
