# code implemented by Aymen Sayed. Git: aymenx17

import os
import glob as gb
import cv2
import numpy as np
import torch
from torch.utils import data
from random import shuffle
import json

'''

- This module prepares the data for a spatio-temporal deep model based on 3D convolutions.
- Each video is split in 32 segments. At this stage, videos are saved in directories of frames.
- Given the different video lengths, a function called find_size is implemented to calculate the right
  number of frames per video, in order to read and encode data into non-operlapping segments.

'''


# def find_size():
#     return


def load_dset():

    ''' Load video frames '''

    path = '/home/aymen/work/pytorch/intuition/detectAnomaly/data'

    dset_path = os.path.join(path, 'trainval_anns')
    vid_list = []

    norm_list = []
    anom_list = []

    for i, (dire, folds, fils) in enumerate(os.walk(dset_path)):
        if dire.split('/')[-1] == 'normal' and len(fils) == 0:
            norm_list = [ os.path.join(dset_path, 'normal', fold) for fold in folds]
        elif i > 0 and len(fils) == 0:
            name_dir = dire.split('/')[-1]
            anom_list += [ os.path.join(dset_path, name_dir, fold) for fold in folds]

    # for comodity I am working with half normal and half abnormal
    # mi = min(len(norm_list), len(anom_list))
    # image_list = norm_list[:mi] + anom_list[:mi]
    image_list = norm_list + anom_list

    return image_list


def load_json(p_ann):
    '''
    A for loop that reads a list of keypoints and rearrange in a directory whose keys are named as the input frames.
    This is done to obtain a suitable format to work with.
    '''

    # load json
    p_ann = os.path.join(p_ann, 'alphapose-results.json')
    if os.path.isfile(p_ann):
        results = json.load(open(p_ann, 'r'))

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
    else:
        anns = {}

    return anns


def get_data(image_list, index):

    try:
        # path to one video annotation
        p_ann = image_list[index]
        anns = load_json(p_ann)
        length = len(anns)

        keyframe = None

    except Exception  as e:
        print('Exception in get_data()')


    return keyframe

class custom_dset(data.Dataset):
    def __init__(self):
        self.image_list  = load_dset()
    def __getitem__(self, index):
        img, anns = get_data(self.image_list,  index)
        return img, anns
    def __len__(self):
        return len(self.image_list)

def collate_fn(batch):
    keyframes = zip(*batch)

    return keyframes
