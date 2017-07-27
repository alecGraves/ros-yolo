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

4. Install pip:
```
sudo apt install python-pip python3-pip
```

5. Install python packages:
```
python -m pip install \
    numpy==1.11.0 \
    pillow==4.2.1 \
    h5py==2.7.0 \
    keras==2.0.6 \
    tensorflow-gpu==1.2.1
```
6. [install CUDA](https://gist.github.com/shadySource/c0f1223d653b6488fde748dcac42d232#3-gpu-if-you-want-to-use-gpu) for GPU support (highly reccomended if you have a gpu)

7. Create a model with YAD2K using python:
    1. create a custom classes file
        1. see ```YAD2K/model_data/aerial_classes.txt``` for inspiration.
    2. Download the .weights and .cfg of the darknet model you want to use into YAD2K
        1. you can find these files at https://pjreddie.com/darknet/yolo/
    3. Make a keras yolo model.
        1. Ex.
        ```
        python yad2k.py -flcl yolo.cfg yolo.weights model_data/yolo.h5
        ```
    4. Use retrain_yolo.py to fine-tune the pretrained model for a new dataset.
        1. ```python retrain_yolo.py --help``` for some info about args
            1. ex. usage: 
            ```
            python retrain_yolo.py -d PATH/TO/my_dataset.npz -c PATH/TO/my_classes.txt
            ```
        2. see example dataset [here](https://github.com/shadySource/DATA/tree/092649fd175886ca894630659eb30614f9bf6c26)

8. Ready :D

# Usage
This package is designed to get camera data from a robot as a jpeg buffer in a multiarray then compute box predictions.

1. start encoding and publishing images with ```rosrun yolo encoder```

2. start yolo with ```rosrun yolo yolo.py -d -c /PATH/TO/my_classes.txt```

