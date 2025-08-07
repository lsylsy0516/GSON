#!/home/luo/miniconda3/envs/gson/bin/python3

# -*- coding: utf-8 -*-
# Modified by: Shangyi Luo (lsylsy030516@gmail.com)
# Original project: GSON - Group-based Social Navigation Framework
# License: GNU General Public License v3.0 (see LICENSE file)
 
import cv2
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
from std_msgs.msg import Header

class ImageMergerNode:
    def __init__(self):
        self.left_images = deque(maxlen=5)
        self.right_images = deque(maxlen=5)
        self.bridge = CvBridge()
        self.last_stamp = None  

        self.merged_image_pub = rospy.Publisher('/merged_image', Image, queue_size=10)
        self.left_image_sub = rospy.Subscriber('/usb_cam_l/image_raw', Image, self.left_image_callback)
        self.right_image_sub = rospy.Subscriber('/usb_cam_r/image_raw', Image, self.right_image_callback)
        self.header = Header()
        rospy.loginfo('Image merger node started')

    def left_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.left_images.append(cv_image)
        self.header = msg.header
        self.merge_images()  
        
    def right_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.right_images.append(cv_image)

    def merge_images(self):
        if len(self.left_images) >= 1 and len(self.right_images) >= 1:
            if self.last_stamp == self.header.stamp:
                return  # skip if already published for this stamp

            left_image = self.left_images[-1]
            right_image = self.right_images[-1]
            merged_image = cv2.hconcat([left_image, right_image])
            merged_image = cv2.resize(merged_image, (1280,int(1280 * merged_image.shape[0] / merged_image.shape[1])))   # set width to 1280, height proportional

            merged_image_msg = self.bridge.cv2_to_imgmsg(merged_image, encoding='bgr8')
            merged_image_msg.header.stamp = self.header.stamp
            merged_image_msg.header.frame_id = "merged_frame"
            self.merged_image_pub.publish(merged_image_msg)

            self.last_stamp = self.header.stamp  # update last published time

    def run(self):
        rospy.spin() 

if __name__ == '__main__':
    rospy.init_node('image_merger_node', anonymous=True)
    image_merger = ImageMergerNode()
    image_merger.run()