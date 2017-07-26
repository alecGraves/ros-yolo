/***********************************
Created by:
    https://github.com/Appleman8977
************************************/
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/UInt8MultiArray.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;

Mat Image;

void imageCB(const std_msgs::UInt8MultiArray::ConstPtr& msg)
{
	cv::resize(imdecode(msg->data, 1), Image, cv::Size(640, 480));
	imshow("compressed_feed", Image);
	waitKey(3);
}

int main(int argc, char **argv){
	std::string topicName("/usb_cam/image_raw");

    if (argc == 2) // If argument given.
    {
        topicName = argv[1];
    }
	ros::init(argc, argv, "listener");
	ros::NodeHandle n;
	
	ros::Subscriber sub = n.subscribe(topicName, 1, imageCB);

	ros::spin();

	return 0;
}