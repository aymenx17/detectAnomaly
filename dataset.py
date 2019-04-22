import os
import subprocess


# root directory
data_root = os.path.join(os.getcwd(), 'data')
dset_root = os.path.join(data_root, 'trial_dataset')

# trainval will be the main dataset folder for the processed videos
outpath = os.path.join(data_root, 'trainval')
if not os.path.isdir(outpath):
    os.mkdir(outpath)



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
