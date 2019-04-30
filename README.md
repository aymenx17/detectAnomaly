
### Intro
This repository supports the UCF-Anomaly-Detection dataset, which has been constructed for two main tasks: Anomaly detection, video activity classification. The current code implementation allows to make a pose estimation analysis on the dataset, in order to better evaluate the idea to feed pose info as input for the task anomaly detection.



### Environment

The code has been tested on this setup:

- Pytorch 1.0
- Python 3.7.1
- opencv 4.0
- numpy  1.15.4

For a more comprehensive list of required packages you may refer to the file env.yaml
You may also use conda package manager to recreate that specific working environment, with the following command:
```python
conda env create -f env.yaml
```



### DATASET

The dataset can be also downloaded from the following link: https://visionlab.uncc.edu/download/summary/60-data/477-ucf-anomaly-detection-dataset

It covers 13 realworld anomalies, including Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism.


### Run code

#### ucf_poseAnalysis.py

This module has been implemented to automate and perform several tasks on the dataset.
The code appears with three tasks: Splitting the videos into frames (ffmpeg), running pose estimation on the frames,
saving few statistics in csv format. The code allows to run these tasks on a selected target of video classes
or on the whole dataset (95 GB).

To set targets in the python code, you need to act on the list named 'target'.
At first run, you need to set the path for pose framework: In the code under the comment 'set working directory for AlphaPose framework'. For the dataset, you can see a sample structure commented in the code, and even here you may change path, (default: 'working-directory/data').


```python
python ucf_poseAnalysis.py --help # will also show available targets
python  \-\-split        # only to split on the whole dataset
python \-\-pose \-\-stats  # runs pose on the previously split videos
python `--`split `--`pose `--`stats `--`target # split and runs pose on target videos, saving statistics
```



##### Sample of statistics from stats.csv

![statistics](https://github.com/aymenx17/blob/master/project_images/stats.png)


#### vis_keypoints.py

This module will read the annotation files and it will apply the key-points upon black frames and
concatenates these with the original frames in order to debug the actual joint-detections. This gives
a clearer image on the quality of pose-estimation on the typology of these videos.

You need to set the same dataset path and it has to be executed at the same path as in ucf_poseAnalysis.py.  

```python
python vis_keypoints.py
```


##### Sample visualization

![2320.png](https://github.com/aymenx17/blob/master/project_images/2320.jpg)
![2321.png](https://github.com/aymenx17/blob/master/project_images/2321.jpg)

### Citations and References

##### For the dataset:

Citation:  
@InProceedings{Sultani_2018_CVPR,  
author = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},  
title = {Real-World Anomaly Detection in Surveillance Videos},  
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2018}  
}  

##### For pose estimation:

[Code](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch)  
@inproceedings{fang2017rmpe,
  title={{RMPE}: Regional Multi-person Pose Estimation},
  author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
  booktitle={ICCV},
  year={2017}
}
@inproceedings{xiu2018poseflow,
  author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
  title = {{Pose Flow}: Efficient Online Pose Tracking},
  booktitle={BMVC},
  year = {2018}
}
