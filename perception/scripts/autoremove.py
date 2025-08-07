#!/home/luo/miniconda3/envs/gson/bin/python3

# -*- coding: utf-8 -*-
# Author: Shangyi Luo (lsylsy030516@gmail.com)
# Project: GSON - Group-based Social Navigation Framework

import rospy
import cv2
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
import tf.transformations as tf_trans
import numpy as np
import tf2_ros
import rospkg

class DetectionRemover:
    def __init__(self):
        # Initialize map flag and TF listener
        self.map_flag = False
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        # ROS subscribers and publisher
        rospy.Subscriber("/dr_spaam_detections", PoseArray, self.detect_callback)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.detect_pub = rospy.Publisher("/removed_detections", PoseArray, queue_size=10)

        # Load and process the map
        package_path = rospkg.RosPack().get_path('perception')
        self.map_path = f"{package_path}/maps/map.pgm"
        self.origin_map = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        self.origin_map = cv2.threshold(self.origin_map, 220, 255, cv2.THRESH_BINARY)[1]
        self.costmap = cv2.erode(
            cv2.dilate(self.origin_map, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))),
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        )
        rospy.loginfo("Costmap created")

    def map_callback(self, msg: OccupancyGrid):
        # Store map metadata
        self.map_flag = True
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.resolution = msg.info.resolution
        rospy.loginfo("Map info received")

    def detect_callback(self, msg: PoseArray):
        if not self.map_flag:
            return

        pose_array = PoseArray()
        pose_array.header = msg.header
        costmap = self.costmap.copy()

        for pose in msg.poses:
            # Transform pose from laser frame to map frame
            trans = self.buffer.lookup_transform("map", "laser_frame", rospy.Time(0), rospy.Duration(10))
            trans_x, trans_y = trans.transform.translation.x, trans.transform.translation.y
            qtn = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            )
            _, _, trans_yaw = tf_trans.euler_from_quaternion(qtn)

            map_pose = Pose()
            map_pose.position.x = pose.position.x * np.cos(trans_yaw) - pose.position.y * np.sin(trans_yaw) + trans_x
            map_pose.position.y = pose.position.y * np.cos(trans_yaw) + pose.position.x * np.sin(trans_yaw) + trans_y

            # Convert map coordinates to pixel coordinates
            x = int((map_pose.position.x - self.origin_x) / self.resolution)
            y = self.costmap.shape[0] - int((map_pose.position.y - self.origin_y) / self.resolution)

            if x >= costmap.shape[1] or y >= costmap.shape[0]:
                continue

            # Filter detections based on costmap
            if costmap[y, x] > 100:
                cv2.circle(costmap, (x, y), 20, 100, -1)
                pose_array.poses.append(pose)

        self.detect_pub.publish(pose_array)

        # for visualization purposes
        # cv2.imshow("Costmap", costmap)
        # cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("detection_remover")
    DetectionRemover()
    rospy.spin()