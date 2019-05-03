# code implemented by Aymen Sayed. Git: aymenx17
import os
import subprocess
import json
import csv
import argparse


'''

This module is adpated to UCF-Anomaly-Detection dataset.

Three functions:
    - split_videos     ---> It runs ffmpeg on the dataset with fixed fps.
    - pose_estimation  ---> It runs demo.py from AlphaPose framework on folder frames.
    - vis_stats        ---> It prints and save statistichs about pose estimation on the dataset.

Note:   Each of the first two functions create and new directory structure.
        You can select target videos to work with by changing the python list named 'target'.
        Example : target = ['Normal_Videos_event', 'Fighting', 'Robbery']



Expected dataset folder structure:

Videos: {
        Normal_Videos_event:
         {
          vid1.mp4, vid2.mp4 . .. .
          },

        Fighting:
         {
          vid1.mp4, vid2.mp4 . .. .
          },
        Robbery:
        {
        vid1.mp4, vid2.mp4 . .. .
        }
        }

'''

parser = argparse.ArgumentParser('Pose estimation analysis on UCF-Anomaly-Detection dataset')

parser.add_argument('--split', action='store_true', default=False, help='Store true flag to split videos')
parser.add_argument('--pose', action='store_true', default=False, help='Store true flag to pose estimation')
parser.add_argument('--stats', action='store_true', default=False, help='Store true flag to print and save statistichs')
parser.add_argument('--target', action='store_true', default=False, help='Store true flag if you have choosed targets')
parser.add_argument('--limit', action='store_true',  default=False, help='Store true flag to process only videos below certain length')

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

def split_videos(dset_root, outpath, target):

    for i, (dire, folds, fils) in enumerate(os.walk(dset_root)):
        if i ==0:
            print('Main dataset directory: {}\n  Subdirectories within it: {}\n'.format(dire, folds))
            continue

        if i > 0 and len(fils) > 0 and (dire.split('/')[-1] not in target):
            continue


        # loop over videos
        print('Processing videos in: {}'.format(dire))
        for vid_name in fils:

            # path to video
            pvid = os.path.join(dire, vid_name)

            # create a folder for each video
            fname = vid_name.split('.mp4')[0]
            path_fold = os.path.join(outpath,  dire.split('/')[-1], fname)
            if not os.path.isdir(path_fold):
                os.makedirs(path_fold)

            # executing ffmpeg -i file.mp4 -vf fps=5 path/%04d.jpg
            print('*' *100)
            print(pvid)
            print(path_fold)
            cmd = "ffmpeg -i {}  -vf fps=5  {}/%04d.jpg".format(pvid, path_fold)
            subprocess.call(cmd, shell=True)



def pose_estimation(outpath, outanns, target):

    max_length = 2e03
    args = parser.parse_args()
    for i,(dire, folds, fils) in enumerate(os.walk(outpath)):
        if i ==0:
            print('Main dataset directory: {}\n  Subdirectories within it: {}\n'.format(dire, folds))
            continue

        # run command on the frames
        if len(folds) == 0 and len(fils) > 0 and (dire.split('/')[-2] in target):

            if args.limit and len(fils) > max_length:
                continue

            # loop over video frames
            print('Processing video frames in: {}'.format(dire))

            # run a python module and output the json file at the correpondent annotation folder
            ldir = dire.split('/')
            out_json = os.path.join(outanns, ldir[-2], ldir[-1])
            if not os.path.isdir(out_json):
                os.makedirs(out_json)
            cmd = "python demo.py --indir {}  --outdir {}".format(dire, out_json)
            subprocess.call(cmd , shell=True)


def vis_stats(out_splitted):

    '''
    The function will print and save statistichs and ratios related of detecting pose in UCF Anomaly dataset.
    '''


    csv_file = 'stats.csv'
    with open(csv_file, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['VideoName' ,'NumberOfFramesWithKeypointDetections', 'TotalFrames', 'Percentage'])
        writer.writeheader()
        for (dire, folds, fils) in os.walk(out_splitted):

            if len(folds) == 0 and len(fils) > 0:
                num_frames = len(fils)
                p_json = os.path.join(dire.replace('trainval', 'trainval_anns'))
                anns = load_json(p_json)
                if len(anns) >0:
                    # number of frames with at least one keypoint detection
                    num_preds = len(anns)
                    perc = round(num_preds/num_frames, 1) * 100

                    vn = dire.split('/')[-1]
                    writer.writerow({ 'VideoName': vn,'NumberOfFramesWithKeypointDetections': num_preds, 'TotalFrames': num_frames, 'Percentage':perc})

                    print('Number of pose detections over total frames per video: {}/{}   Percentage: {}%'.format(num_preds, num_frames, perc))



def main():
    # root directory
    data_root = '/media/sdc1'
    # UCF_Anomalies dataset path. Replace the name trial_dataset with the proper name. Default: Videos
    dset_root = os.path.join(data_root, 'Videos')
    args = parser.parse_args()

    # trainval will be the main dataset folder for the processed videos
    out_splitted = os.path.join(data_root, 'trainval')
    if not os.path.isdir(out_splitted):
        os.mkdir(out_splitted)

    # you can select target folders you want to process using this list
    if args.target:
        target = ['Training-Normal-Videos-Part-1']
    else:
        target = os.listdir(dset_root)

    print('targets: {}'.format(target))
    # create new directory structure and write frames at path out_splitted
    if args.split:
        split_videos(dset_root, out_splitted, target)


    # output annotation path (json files)
    outanns = os.path.join(data_root, 'trainval_anns')
    if not os.path.isdir(outanns):
        os.mkdir(outanns)

    # set working directory for AlphaPose framework
    os.chdir('/home/ubuntu/work/pytorch/intuition/AlphaPose')
    # create new directory structure and write annotation files at outanns
    if args.pose:
        print('\n'*5)
        print('Running pose estimation\n\n')
        pose_estimation(out_splitted, outanns, target)

    # print statistichs
    if args.stats:
        os.chdir(data_root)
        vis_stats(out_splitted)
        print('stats.csv is saved in: {}'.format(data_root))


if __name__ == "__main__":
    main()
