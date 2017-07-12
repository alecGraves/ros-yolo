#####################################################################
# This dockerfile is to set up the environment for this package.
# Ex. usage:
#  bash "docker build -t kinetic-yolo ."
#  cp ../ros_yolo ~/catkin_ws/src
#  docker run --device=/dev/video1:/dev/video0 -itv  ~/catkin_ws/src:/root/catkin_ws -w="/root/catkin_ws" kinetic-yolo
#####################################################################
FROM ros:kinetic-ros-core

# Install python opencv, cv_bridge and usb_cam ros package.
RUN apt-get update && apt-get install --no-install-recommends -y \
    ros-kinetic-cv-bridge \
    ros-kinetic-usb-cam \
    python-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    numpy==1.11.0 \
    pillow==4.2.1 \
    PyYAML==3.12 \
    h5py==2.7.0 \
    keras==2.0.6 \
    tensorflow-gpu==1.2.1




