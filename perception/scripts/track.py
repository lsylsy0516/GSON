#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified by: Shangyi Luo (lsylsy030516@gmail.com)
# Original project: GSON - Group-based Social Navigation Framework
# License: GNU General Public License v3.0 (see LICENSE file)

import os
import sys
import cv2
import rospy
import rospkg
import numpy as np

from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from detection_msgs.msg import tracks
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf.transformations as tf_trans

from utils.tracker import Tracker
from utils.markers_pub import marker_create, marker_array_create

rospack = rospkg.RosPack()
package_path = rospack.get_path('perception')

MAX_TRACK_DISTANCE_SQ = 15 ** 2  # Max distance squared for tracking (15 meters)


class TrackerModule:
    def __init__(self, if_visualization=True, if_publish=True):
        self.if_visualization = if_visualization
        self.if_publish = if_publish

        self.tracker_pub = rospy.Publisher("/tracker", tracks, queue_size=1)
        self.id_marks_pub = rospy.Publisher("id_marks", MarkerArray, queue_size=1)
        self.det_marker_pub = rospy.Publisher("poses_to_box/marker", Marker, queue_size=1)

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.global_map_callback, queue_size=10)

        self.tracker = Tracker(
            dist_thresh=0.7,
            max_frames_to_skip=20,
            max_trace_length=3,
            trackIdCount=0,
            predict_step=6,
            kf_measurement_noise=0.1,
            kf_process_noise=0.3
        )

        rospy.loginfo("Tracker initialization successful.")

    def global_map_callback(self, msg: OccupancyGrid):
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.map_sub.unregister()

    def remove_duplicate_poses(self, poses):
        unique = set()
        filtered = []
        for pose in poses:
            key = (pose.position.x, pose.position.y)
            if key not in unique:
                unique.add(key)
                filtered.append(pose)
            else:
                rospy.loginfo("Removed duplicate pose: %s", key)
        return filtered

    def transform_to_global(self, poses):
        centers = []
        try:
            trans = self.buffer.lookup_transform("map", "laser_frame", rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("TF transform lookup failed: %s", e)
            return centers

        trans_x = trans.transform.translation.x
        trans_y = trans.transform.translation.y
        qtn = (
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        )
        _, _, trans_yaw = tf_trans.euler_from_quaternion(qtn)

        for pose in poses:
            local_x = pose.position.x
            local_y = pose.position.y
            if local_x ** 2 + local_y ** 2 > MAX_TRACK_DISTANCE_SQ:
                continue
            global_x = local_x * np.cos(trans_yaw) - local_y * np.sin(trans_yaw) + trans_x
            global_y = local_y * np.cos(trans_yaw) + local_x * np.sin(trans_yaw) + trans_y
            centers.append(np.array([[global_x], [global_y]]))

        return centers

    def dynamic_obstacles_cb(self, msg: PoseArray):
        if not msg.poses:
            return [], []

        msg.poses = self.remove_duplicate_poses(msg.poses)
        centers = self.transform_to_global(msg.poses)
        self.tracker.Update(centers)

        if not self.tracker.tracks:
            rospy.loginfo("No active tracks.")
            return [], []

        try:
            states = np.array([[[track.KF.state[0], track.KF.state[1], 0]] for track in self.tracker.tracks])
            states = np.expand_dims(states, axis=0)
            for n in range(self.tracker.predict_step):
                k_states = np.array([[[track.KF.future_states[n][0], track.KF.future_states[n][1], 0]] for track in self.tracker.tracks])
                k_states = np.expand_dims(k_states, axis=0)
                states = np.concatenate((states, k_states), axis=0)
            self.other_agents_states = states
        except Exception as e:
            rospy.logwarn(f"[Tracker Exception] Track Count: {len(self.tracker.tracks)}, Error: {e}")

        if self.if_publish:
            tracks_msg = tracks()
            tracks_msg.header.stamp = msg.header.stamp
            tracker_pose_msg = PoseArray()
            tracker_pose_msg.header.stamp = msg.header.stamp
            tracker_pose_msg.header.frame_id = "map"

            for track in self.tracker.tracks:
                pose = Pose()
                pose.position.x = track.KF.state[0]
                pose.position.y = track.KF.state[1]
                pose.position.z = 1
                tracker_pose_msg.poses.append(pose)
                tracks_msg.track_id_list.append(track.track_id % 100)
                tracks_msg.track_vel_x_list.append(track.KF.state[2])
                tracks_msg.track_vel_y_list.append(track.KF.state[3])
            tracks_msg.track_pose_list = tracker_pose_msg

            self.tracker_pub.publish(tracks_msg)
            self.det_marker_pub.publish(marker_create(tracker_pose_msg, "map", 0, [0, 255, 0], 1))

        track_id_list = [track.track_id for track in self.tracker.tracks]
        track_pose_list = [track.KF.state for track in self.tracker.tracks]
        id_marks = marker_array_create(track_id_list, track_pose_list, "id_marks", [0, 255, 0], 1)
        self.id_marks_pub.publish(id_marks)

        return track_id_list, track_pose_list

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node("tracker_node")
    tracker_node = TrackerModule(if_visualization=False, if_publish=True)
    rospy.Subscriber("/removed_detections", PoseArray, tracker_node.dynamic_obstacles_cb)
    tracker_node.run()
    rospy.spin()
