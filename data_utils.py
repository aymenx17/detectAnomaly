

import json
import cv2
import numpy as np
import os
from fn import vis_frame as vis_frame


annotation_file = './data/pose_walking/alphapose-results-forvis.json'
img_path = './data/split_walking/'
anns = json.load(open(annotation_file, 'r'))

out_path = './data/debug_input'


limgs = os.listdir(img_path)
limgs.sort()


for name in limgs:
    p = os.path.join(img_path, name)
    img = cv2.imread(p)
    shape = img.shape
    print(name)
    print(shape)
    # create a black Image
    empty = np.zeros(shape, dtype=np.uint8)

    kd = name.split('.png')[0]
    key_img = vis_frame(empty, anns[kd])

    # concatenate images for visualization
    img_cat = np.concatenate((img, key_img), axis=1)

    name_out = os.path.join(out_path, name)
    cv2.imwrite(name_out, img_cat)
