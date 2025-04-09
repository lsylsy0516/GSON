# GSON

## Getting started
### 1. Installation
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

### 2.Quickstart

for dashgo robot,after you launch its navigation submodule,
you can then start GSON module in two terminal:
```
roslaunch perception_module total.launch
roslaunch ourplanner new_planner.launch
```