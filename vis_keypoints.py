# code implemented by Aymen Sayed. Git: aymenx17

import json
import cv2
import numpy as np
import os
from fn import vis_frame as vis_frame
from load_dataset import load_json

'''

This module has to be executed at the same working directory of ucf_poseAnalysis.py.

'''



def main ():

    # root directory
    data_root = os.path.join(os.getcwd(), 'data')

    # trainval is the main dataset folder for the processed videos
    out_splitted = os.path.join(data_root, 'trainval')

    for i,(dire, folds, fils) in enumerate(os.walk(out_splitted)):
        if i ==0:
            print('Main dataset directory: {}\n  Subdirectories within it: {}\n'.format(dire, folds))
            continue

        # run command on the frames
        if len(folds) == 0 and len(fils) > 0:

            # loop over video frames
            print('Reading frames from: {}'.format(dire))

            # read the annotations specific to this folder
            p_json = os.path.join(dire.replace('trainval', 'trainval_anns'))
            anns = load_json(p_json)

            # read frames
            p_imgs = [os.path.join(dire, frame_name) for frame_name in fils if frame_name in anns]
            frames_name = [p.split('/')[-1] for p in p_imgs ]
            imgs = [ cv2.imread(p) for p in p_imgs]

            # create a list of black images
            shapes = [ img.shape for img in imgs]
            empty = [np.zeros(shape, dtype=np.uint8) for shape in shapes]


            # apply annotation
            key_imgs = [ vis_frame(empty[i], anns[kd]) for i, kd in enumerate(frames_name)]

            # concatenate
            img_cat = [np.concatenate((img, key_imgs[i]), axis=1) for i, img in enumerate(imgs)]

            p_out = os.path.join(dire.replace('trainval', 'keypoint_frames'))
            if not os.path.isdir(p_out):
                os.makedirs(p_out)

            # save result to disk
            for i, im in enumerate(img_cat):
                p = os.path.join(p_out, frames_name[i])
                cv2.imwrite(p, im)


if __name__ == "__main__":
    main()
