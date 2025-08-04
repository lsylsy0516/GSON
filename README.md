# GSON  Group-based Social Navigation Framework with Large Multimodal Model

<p align="center">
    <img src="imgs/teaser.jpg" alt="GSON Framework Overview" style="width:60%;" />
</p>

-----------------

> This paper introduces GSON, a novel group-based social navigation framework that leverages Large Multimodal Models (LMMs) to enhance robots’ social perception capabilities. Our approach uses visual prompting to enable zero-shot extraction of social relationships among pedestrians and integrates these results with robust pedestrian detection and tracking pipelines to overcome the inherent inference speed limitations of LMMs. 


<p align="center">
  [<a href="https://arxiv.org/abs/2409.18084">Read our arXiv Paper</a>]
</p>

## News
- **2025.07.29**: Our work has been accepted to RA-L ! 
## Quick Start
GSON is developed based on `ubuntu 20.04`, `python=3.8`,`pytorch=1.11.0`,and `yolov5_ros`

```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive https://github.com/lsylsy0516/GSON.git && cd GSON
```

`conda` is recommand to setup virtual python environment:
```
conda create -n gson python=3.8
conda activate gson
pip install -r ./yolov5_ros/src/yolov5/requirements.txt
cd ./2D_lidar_person_detection/dr_spaam
python setup.py install
```

After that,change our code from `/home/orin/miniconda3` to your `$MINICONDA PATH` 
then you can compile gson with 
```
catkin_make
```

Also , you need to download the necessary weight file for **2D LiDAR detection**:
https://drive.google.com/drive/folders/1Wl2nC8lJ6s9NI1xtWwmxeAUnuxDiiM4W

In GSON ,we use `ckpt_jrdb_ann_drow3_e40.pth`


## Third-party Code and Licenses

This repository incorporates and modifies code from the following open-source projects:

- [`yolov5_ros`](https://github.com/mats-robotics/yolov5_ros) (GPLv3):  
  Used for YOLOv5-based pedestrian detection in ROS.  
  We modified internal modules (e.g., `my_plotting.py`) and integrated it into our pipeline.  
  See `third_party/yolov5_ros/` for details.

- [`2D_lidar_person_detection`](https://github.com/VisualComputingInstitute/2D_lidar_person_detection) (GPLv3):  
  Used for LiDAR-based person detection.  
  We adapted parts of the code for real-time filtering and ROS integration.  
  See `third_party/2D_lidar_person_detection/`.