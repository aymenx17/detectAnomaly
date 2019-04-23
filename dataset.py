import os
import subprocess


'''

This module is adpated to UCF_Anomalies dataset.

Two functions:
    - split_videos     ---> It runs ffmpeg on the dataset with fixed fps.
    - pose_estimation  ---> It runs demo.py from AlphaPose framework on folder frames.

Note: Each of these functions will create and new directory structure.



'''


def split_videos(dset_root, outpath):
    
    for i, (dire, folds, fils) in enumerate(os.walk(dset_root)):
        if i ==0:
            print('Main dataset directory: {}\n  Subdirectories within it: {}\n Files within it: {}'.format(dire, folds, fils))

            # copy folder's structure in outpath
            for f in folds:
                if not os.path.isdir(os.path.join(outpath, f)):
                    os.mkdir(os.path.join(outpath, f))
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
                os.mkdir(path_fold)

            # executing ffmpeg -i walking.mp4 -vf fps=1 %04d.jpg
            print('*' *100)
            print(pvid)
            print(path_fold)
            cmd = "ffmpeg -i {}  -vf fps=5  {}/%04d.jpg".format(pvid, path_fold)
            subprocess.call(cmd, shell=True)



def pose_estimation(outpath, outanns):

    for i,(dire, folds, fils) in enumerate(os.walk(outpath)):
        if i ==0:
            print('Main dataset directory: {}\n  Subdirectories within it: {}\n Files within it: {}'.format(dire, folds, fils))

            # copy folder's structure in outpath
            for f in folds:
                if not os.path.isdir(os.path.join(outanns, f)):
                    os.mkdir(os.path.join(outanns, f))
            continue

        # loop over video frames
        print('Processing video frames in: {}'.format(dire))
        print(folds)
        print(fils)

        # replicate same directory structure for the json files
        if len(folds) > 0 and len(fils) == 0:
                for f in folds:
                    if not os.path.isdir(os.path.join(dire, f)):
                        os.mkdir(os.path.join(dire, f))
                continue

        # run command on the frames
        if len(folds) == 0 and len(fils) > 0:

            # run a python module and output the json file at the correpondent annotation folder
            ldir = dire.split('/')
            out_json = os.path.join(outanns, ldir[-2], ldir[-1])
            cmd = "python demo.py --indir {}  --outdir {}".format(dire, out_json)
            subprocess.call(cmd , shell=True)



def main():
    # root directory
    data_root = os.path.join(os.getcwd(), 'data')
    # UCF_Anomalies dataset path. Replace the name trial_dataset with the proper name. Default: Videos
    dset_root = os.path.join(data_root, 'trial_dataset')

    # trainval will be the main dataset folder for the processed videos
    out_splitted = os.path.join(data_root, 'trainval')
    if not os.path.isdir(out_splitted):
        os.mkdir(out_splitted)


    # create new directory structure and write frames at path out_splitted
    split_videos(dset_root, out_splitted)

    print('\n'*5)
    print('Running pose estimation\n\n')

    # output annotation path (json files)
    outanns = os.path.join(data_root, 'trainval_anns')
    if not os.path.isdir(outanns):
        os.mkdir(outanns)

    # set working directory for AlphaPose framework
    os.chdir('/home/paperspace/work/pytorch/intuition/AlphaPose')
    # create new directory structure and write annotation files at outanns
    pose_estimation(out_splitted, outanns)


if __name__ == "__main__":
    main()
