#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Modified by: Shangyi Luo (lsylsy030516@gmail.com)
# Original project: GSON - Group-based Social Navigation Framework
# License: GNU General Public License v3.0 (see LICENSE file)

import cv2
import json
import math
import numpy as np
import os
import rospy
import rospkg
import shutil
import threading

from itertools import combinations
from collections import defaultdict, deque
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from detection_msgs.msg import BoundingBoxes, Groups, mapping, Group
from utils.qwen import group_naive


class ConnectionTracker:
    def __init__(self, if_pub=False, window_size=5, decay=0.9, fusion_weight=0.6, sigma=1):
        self.window_size = window_size
        self.decay = decay
        self.fusion_weight = fusion_weight
        self.sigma = sigma  # Controls proximity decay scale
        self.conn_history = deque(maxlen=window_size)
        self.connection = defaultdict(lambda: defaultdict(float))
        self.observed_ids = set()

        if if_pub:
            self.group_pub = rospy.Publisher('/detection/group', Groups, queue_size=1)
            rospy.loginfo("ConnectionTracker Publisher Initialized")

    def compute_proximity(self, bbox1, bbox2):
        def center(bbox):
            cx, cy, w, h = bbox
            return (cx, cy)

        c1 = center(bbox1)
        c2 = center(bbox2)
        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        proximity = np.exp(-dist / self.sigma)  # Closer means higher proximity (approaching 1.0)
        return proximity

    def update(self, group_list, bbox_dict=None):
        self.conn_history.append(group_list)
        current_ids = set()
        co_occurred = set()

        for group in group_list:
            current_ids.update(group)
            for i, j in combinations(sorted(group), 2):
                co_occurred.add((i, j))

        self.observed_ids.update(current_ids)

        for i in self.observed_ids:
            for j in self.observed_ids:
                if i >= j:
                    continue

                old_val = self.connection[i][j]

                if (i, j) in co_occurred:
                    proximity = 1.0
                    if bbox_dict and i in bbox_dict and j in bbox_dict:
                        proximity = self.compute_proximity(bbox_dict[i], bbox_dict[j])
                    new_val = self.fusion_weight * proximity + (1 - self.fusion_weight) * old_val
                    self.connection[i][j] = new_val
                else:
                    self.connection[i][j] = self.decay * old_val

    def get_connected_groups(self, threshold=0.6):
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        for pid in self.observed_ids:
            parent[pid] = pid

        for i in self.observed_ids:
            for j in self.observed_ids:
                if i < j and self.connection[i][j] >= threshold:
                    union(i, j)

        groups = defaultdict(list)
        for pid in self.observed_ids:
            groups[find(pid)].append(pid)
        return list(groups.values())

    def print_connections(self):
        for i in sorted(self.observed_ids):
            for j in sorted(self.observed_ids):
                if i < j:
                    print(f"Conn({i},{j}) = {self.connection[i][j]:.2f}")
    
    def tracker_callback(self, tracks_msg):
        try:
            track_ids = tracks_msg.track_id_list
            poses = tracks_msg.track_pose_list.poses
            vxs = tracks_msg.track_vel_x_list
            vys = tracks_msg.track_vel_y_list

            if not (len(track_ids) == len(poses) == len(vxs) == len(vys)):
                rospy.logwarn("Mismatched track arrays in tracks_msg")
                return

            id_to_idx = {track_id: i for i, track_id in enumerate(track_ids)}

            # Get connection-based groups (including historical IDs)
            connected_groups = self.get_connected_groups(threshold=0.6)
            group_list_msg = []
            included_ids = set()

            for group in connected_groups:
                filtered_group = [pid for pid in group if pid in id_to_idx]
                if not filtered_group:
                    continue

                group_msg = Group()
                group_msg.header = tracks_msg.header
                group_msg.group_id_list = filtered_group
                group_msg.group_pose_list = PoseArray()
                group_msg.group_pose_list.header = tracks_msg.header
                group_msg.group_vel_x_list = []
                group_msg.group_vel_y_list = []

                for pid in filtered_group:
                    idx = id_to_idx[pid]
                    group_msg.group_pose_list.poses.append(poses[idx])
                    group_msg.group_vel_x_list.append(vxs[idx])
                    group_msg.group_vel_y_list.append(vys[idx])

                included_ids.update(filtered_group)
                group_list_msg.append(group_msg)

            # Add IDs not included in any group as separate groups
            missing_ids = set(track_ids) - included_ids
            for pid in missing_ids:
                idx = id_to_idx[pid]
                group_msg = Group()
                group_msg.header = tracks_msg.header
                group_msg.group_id_list = [pid]
                group_msg.group_pose_list = PoseArray()
                group_msg.group_pose_list.header = tracks_msg.header
                group_msg.group_pose_list.poses = [poses[idx]]
                group_msg.group_vel_x_list = [vxs[idx]]
                group_msg.group_vel_y_list = [vys[idx]]
                group_list_msg.append(group_msg)

            groups_msg = Groups()
            groups_msg.header = tracks_msg.header
            groups_msg.group_list = group_list_msg
            self.group_pub.publish(groups_msg)

        except Exception as e:
            rospy.logerr(f"Error in tracker_callback: {e}")

class KeyframeSaver:
    def __init__(self, connection_tracker, if_publish=True, if_node=True, save_dir=None):
        self.if_publish = if_publish
        self.tracker = connection_tracker
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("image_topic", "/yolov5/image_out")
        self.bbox_topic = rospy.get_param("bbox_topic", "/yolov5/detections")

        if not save_dir:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('perception')  # Update as needed
            self.save_dir = os.path.join(pkg_path, "keyframes")
        else:
            self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            rospy.loginfo(f"Keyframe directory {self.save_dir} already exists. Clearing.")
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

        self.cache = []
        self.cache_limit = 10
        self.cache_cnt = 0
        self.frame_id = 0

        if if_node:
            self.image_pub = rospy.Publisher("/grouped_image", Image, queue_size=1)
            image_sub = Subscriber(self.image_topic, Image)
            bbox_sub = Subscriber(self.bbox_topic, BoundingBoxes)
            self.ts = ApproximateTimeSynchronizer([image_sub, bbox_sub], queue_size=10, slop=0.2)
            self.ts.registerCallback(self.synced_callback)

        self.display_enabled = True
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

        rospy.loginfo("KeyframeSaver initialized.")

    def display_loop(self):
        while self.display_enabled and not rospy.is_shutdown():
            if self.cache:
                try:
                    image, label_data, _, _ = self.cache[-1] if len(self.cache[-1]) == 4 else (*self.cache[-1], {})
                    image = image.copy()
                    bbox_dict = {}
                    for obj in label_data:
                        try:
                            pid = int(obj["Class"])
                            x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            bbox_dict[pid] = (cx, cy)
                        except Exception as e:
                            rospy.logwarn(f"Invalid label in display_loop: {e}")

                    if bbox_dict:
                        connected_groups = self.tracker.get_connected_groups(threshold=0.5)
                        for group in connected_groups:
                            centers = [(pid, bbox_dict[pid]) for pid in group if pid in bbox_dict]
                            for i in range(len(centers)):
                                for j in range(i + 1, len(centers)):
                                    pt_i = centers[i][1]
                                    pt_j = centers[j][1]
                                    cv2.line(image, pt_i, pt_j, (0, 255, 0), 2)

                    cv2.imshow("Latest Cached Frame", image)
                    key = cv2.waitKey(30)
                    if key == ord('q'):
                        rospy.loginfo("Quitting display thread.")
                        self.display_enabled = False
                        break

                except Exception as e:
                    rospy.logwarn(f"Display error: {e}")
            else:
                rospy.sleep(0.1)

        cv2.destroyAllWindows()

    def synced_callback(self, image_msg, bbox_msg):
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            label_data = []
            bbox_dict = {}
            bbox_count = 0

            if bbox_msg is not None:
                for box in bbox_msg.bounding_boxes:
                    try:
                        gt_id = int(box.Class)
                        x1, y1, x2, y2 = box.xmin, box.ymin, box.xmax, box.ymax
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        bbox_dict[gt_id] = (cx, cy)
                        label_data.append({
                            "Class": box.Class,
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2
                        })
                    except Exception as e:
                        rospy.logwarn(f"Invalid bbox format: {e}")

                bbox_count = len(label_data)

            if self.if_publish:
                try:
                    image_msg_out = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                    self.image_pub.publish(image_msg_out)
                except CvBridgeError as e:
                    rospy.logerr(f"Failed to convert/publish image: {e}")

            self.cache.append((image.copy(), label_data, bbox_count))
            self.cache_cnt += 1

            if self.cache_cnt >= self.cache_limit:
                non_empty = [item for item in self.cache if item[2] > 0]
                if non_empty:
                    best_frame = max(non_empty, key=lambda item: item[2])
                    best_image, best_label_data, _ = best_frame

                    frame_path = os.path.join(self.save_dir, f"frame_{self.frame_id:04d}.jpg")
                    label_path = os.path.join(self.save_dir, f"frame_{self.frame_id:04d}.json")

                    cv2.imwrite(frame_path, best_image)
                    with open(label_path, 'w') as f:
                        json.dump(best_label_data, f)

                    rospy.loginfo(f"Saved keyframe: {frame_path}")
                    self.frame_id += 1
                else:
                    rospy.loginfo("No valid bounding boxes in last 10 frames; skipping keyframe.")

                self.cache.clear()
                self.cache_cnt = 0

        except Exception as e:
            rospy.logerr(f"Error in synced_callback: {e}")


class GroupTracker:
    def __init__(self, connection_tracker=None, keyframe_dir=None, save_dir=None):
        self.tracker = connection_tracker if connection_tracker else ConnectionTracker()

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('perception')  

        self.keyframe_dir = keyframe_dir or os.path.join(pkg_path, 'keyframes')
        self.save_dir = save_dir or os.path.join(pkg_path, 'filter_res', 'res')

        if os.path.exists(self.save_dir):
            rospy.loginfo(f"Clearing existing results in {self.save_dir}")
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.frame_cnt = 0

        rospy.Timer(rospy.Duration(1.0), self.schedule_processing)
        rospy.loginfo("GroupTracker initialized.")

    def schedule_processing(self, event):
        frame_id = self.frame_cnt
        frame_name = f"frame_{frame_id:04d}"
        image_path = os.path.join(self.keyframe_dir, frame_name + ".jpg")
        label_path = os.path.join(self.keyframe_dir, frame_name + ".json")

        if os.path.exists(image_path) and os.path.exists(label_path):
            rospy.loginfo(f"[{frame_name}] Keyframe ready for processing.")
            threading.Thread(target=self.process_frame, args=(frame_id,), daemon=True).start()
            self.frame_cnt += 1
        else:
            rospy.loginfo(f"[{frame_name}] Keyframe not ready yet. Waiting...")

    def process_frame(self, frame_id):
        frame_name = f"frame_{frame_id:04d}"
        image_path = os.path.join(self.keyframe_dir, frame_name + ".jpg")
        label_path = os.path.join(self.keyframe_dir, frame_name + ".json")

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            rospy.loginfo(f"[{frame_name}] Keyframe not ready yet.")
            return

        try:
            image = cv2.imread(image_path)
            with open(label_path, 'r') as f:
                boxes = json.load(f)

            bbox_dict = {}
            ids = []
            for box in boxes:
                try:
                    gt_id = int(box["Class"])
                    x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    bbox_dict[gt_id] = (cx, cy, w, h)
                    ids.append(gt_id)
                except Exception as e:
                    rospy.logwarn(f"[{frame_name}] Invalid box format: {e}")
                    continue

            llm_groups = group_naive(image_path, ids)
            rospy.loginfo(f"[{frame_name}] LLM groups computed.")

            self.tracker.update(llm_groups, bbox_dict)
            filtered_groups = self.tracker.get_connected_groups()
            self.tracker.print_connections()

            result = image.copy()
            h, w, _ = result.shape
            cv2.putText(result, f"{frame_name}", (w - 350, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(result, f"LLM: {llm_groups}", (w - 350, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(result, f"Filtered: {filtered_groups}", (w - 350, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            save_path = os.path.join(self.save_dir, frame_name + ".jpg")
            cv2.imwrite(save_path, result)
            rospy.loginfo(f"[{frame_name}] Processed and saved: {save_path}")

        except Exception as e:
            rospy.logwarn(f"[{frame_name}] Error during processing: {e}")


class Mapping_frame:
    def __init__ (self,msg:mapping):
        self.stamp = msg.header.stamp
        self.id_list = []
        self.points = []
        self.pose_array = []
        self.vel_x_list = []
        self.vel_y_list = []

        self.points = [[msg.point_xs[i],msg.point_ys[i]] for i in range(len(msg.id_list))]
        self.pose_array = [msg.pose_list.poses[i] for i in range(len(msg.id_list))]
        self.id_list = [msg.id_list[i] for i in range(len(msg.id_list))]
        self.vel_x_list = [msg.vel_x_list[i] for i in range(len(msg.id_list))]
        self.vel_y_list = [msg.vel_y_list[i] for i in range(len(msg.id_list))]

    def merge(self,frame:"Mapping_frame"):
        self.id_list.extend(frame.id_list)
        self.points.extend(frame.points)
        self.pose_array.extend(frame.pose_array)
        self.vel_x_list.extend(frame.vel_x_list)
        self.vel_y_list.extend(frame.vel_y_list)
        return self

if __name__ == '__main__':
    rospy.init_node('group_tracker_node')

    try:
        tracker = ConnectionTracker(window_size=5, decay=0.75, fusion_weight=0.6, sigma=600)
        keyframe_saver = KeyframeSaver(tracker, if_publish=True)
        group_tracker = GroupTracker(tracker)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
