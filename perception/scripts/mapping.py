#!/home/luo/miniconda3/envs/gson/bin/python3

# -*- coding: utf-8 -*-
# Modified by: Shangyi Luo (lsylsy030516@gmail.com)
# Original project: GSON - Group-based Social Navigation Framework
# License: GNU General Public License v3.0 (see LICENSE file)

import cv2
import math
import rospy
import numpy as np

from detection_msgs.msg import tracks, mapping
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped

import tf
import tf2_ros
import tf.transformations as tf_trans


class MappingNode:
    def __init__(self, left_flag, camera_info_topic, image_frame, mapping_topic):
        self.listener = tf.TransformListener()
        self.left_flag = left_flag
        self.buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.buffer)

        # ROS I/O topics
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_cb)
        self.tracks_sub = rospy.Subscriber("/tracker", tracks, self.tracks_cb)
        self.mapping_pub = rospy.Publisher(mapping_topic, mapping, queue_size=1)

        self.image_frame = image_frame
        rospy.loginfo(f"Mapping node initialized for {'left' if left_flag else 'right'} camera")

    def camera_info_cb(self, msg: CameraInfo):
        P = np.array(msg.P).reshape(3, 4)
        try:
            camera_matrix, rot_matrix, tvec, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        except cv2.error as e:
            rospy.logerr("Camera projection matrix decomposition error: %s", e)
            return
        tvec = tvec[:3]
        rvec, _ = cv2.Rodrigues(rot_matrix)
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = np.array(msg.D)
        self.rvec = rvec
        self.tvec = tvec
        self.camera_info_sub.unregister()

    def tracks_cb(self, msg: tracks):
        id_list, point_xs, point_ys = [], [], []
        vel_x_list, vel_y_list = [], []
        pose_list = PoseArray()

        for track, id, vel_x, vel_y in zip(msg.track_pose_list.poses, msg.track_id_list, msg.track_vel_x_list, msg.track_vel_y_list):
            try:
                laser_pose = PoseStamped()
                laser_pose.header.frame_id = "laser_frame"
                laser_pose.header.stamp = msg.header.stamp
                trans = self.buffer.lookup_transform("laser_frame", "map", rospy.Time(0), rospy.Duration(1.0))

                trans_x = trans.transform.translation.x
                trans_y = trans.transform.translation.y
                qtn = (
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                )
                _, _, trans_yaw = tf_trans.euler_from_quaternion(qtn)
                laser_pose.pose.position.x = track.position.x * math.cos(trans_yaw) - track.position.y * math.sin(trans_yaw) + trans_x
                laser_pose.pose.position.y = track.position.y * math.cos(trans_yaw) + track.position.x * math.sin(trans_yaw) + trans_y
                laser_pose.pose.position.z = 1

                self.listener.waitForTransform("laser_frame", self.image_frame, rospy.Time(0), rospy.Duration(1.0))
                camera_pose = self.listener.transformPose(self.image_frame, laser_pose)
                cam_coords = np.array([
                    camera_pose.pose.position.x,
                    camera_pose.pose.position.y,
                    camera_pose.pose.position.z
                ])

                if cam_coords[2] < 0 or np.linalg.norm(cam_coords[:2]) > 10:
                    continue

                point, _ = cv2.projectPoints(cam_coords, self.rvec, self.tvec, self.camera_matrix, self.distortion_coefficients)
                point = point[0][0]

                if not (0 <= point[0] <= 1280 and 0 <= point[1] <= 720):
                    continue

                # Adjust based on camera position (left/right)
                point[0] /= 2
                point[1] /= 2
                if not self.left_flag:
                    point[0] += 640 # as merged image is 1280 * 720

                id_list.append(id)
                point_xs.append(int(point[0]))
                point_ys.append(int(point[1]))
                pose_list.poses.append(track)
                vel_x_list.append(vel_x)
                vel_y_list.append(vel_y)

            except Exception as e:
                rospy.logwarn(f"Mapping error: {e}")
                continue

        mapping_msg = mapping()
        mapping_msg.header.stamp = msg.header.stamp
        mapping_msg.id_list = id_list
        mapping_msg.point_xs = point_xs
        mapping_msg.point_ys = point_ys
        mapping_msg.pose_list = pose_list
        mapping_msg.vel_x_list = vel_x_list
        mapping_msg.vel_y_list = vel_y_list

        try:
            self.mapping_pub.publish(mapping_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish mapping: {e}")


if __name__ == "__main__":
    rospy.init_node("mapping_node")

    # Get parameters
    single_camera = rospy.get_param("~single_camera", False)

    if single_camera:
        # Use single camera (default left)
        MappingNode(
            left_flag=True,
            camera_info_topic=rospy.get_param("~camera_info_topic", "/usb_cam_l/camera_info"),
            image_frame=rospy.get_param("~image_frame", "usb_cam_l"),
            mapping_topic=rospy.get_param("~mapping_topic", "/mapping/left")
        )
    else:
        # Dual camera (left and right)
        MappingNode(
            left_flag=True,
            camera_info_topic="/usb_cam_l/camera_info",
            image_frame="usb_cam_l",
            mapping_topic="/mapping/left"
        )
        MappingNode(
            left_flag=False,
            camera_info_topic="/usb_cam_r/camera_info",
            image_frame="usb_cam_r",
            mapping_topic="/mapping/right"
        )

    rospy.spin()
