# ros_yolo
An example ROS package to send image data to a Keras YOLOv2 model.

# Setup
1. Clone this repository into your catkin workspace:
```
git clone --recursive https://github.com/shadySource/ros-yolo.git
```

2. Install ros kinetic

3. Install cv_bridge and usb_cam:
```
sudo apt install ros-kinetic-cv-bridge ros-kinetic-usb-cam
```

4. Install pip for python 2.7:
```
sudo apt install python-pip
```

5. Install python packages:
```
python -m pip install \
    numpy==1.11.0 \
    pillow==4.2.1 \
    h5py==2.7.0 \
    keras==2.0.6 \
    tensorflow-gpu==1.2.1 # for GPU support (highly reccomended)
    # tensorflow==1.2.1 # for CPU only
```
6. [install CUDA](https://gist.github.com/shadySource/c0f1223d653b6488fde748dcac42d232#3-gpu-if-you-want-to-use-gpu) for GPU support (highly reccomended)

7. Create a model:
    1. create a custom classes file in ```YAD2K/model_data```
        1. see ```YAD2K/model_data/aerial_classes.txt``` for inspiration.
    2. download the pretrained YOLO model [here](https://drive.google.com/open?id=0B_fefIm3LDfjOE5ONmlsUE5TMTA).
    3. Use retrain_yolo.py to fine-tune the pretrained model for a new dataset.
        1. ```python retrain_yolo.py --help``` for some info about args
        2. see example dataset [here](https://github.com/shadySource/DATA/tree/092649fd175886ca894630659eb30614f9bf6c26)

8. Ready :D

# Usage
This package is designed to get camera data from a robot as a jpeg buffer in a multiarray then compute box predictions.

1. start encoding and publishing images with ```rosrun ros_yolo encoder```

2. start yolo with ```rosrun ros_yolo yolo.py```

