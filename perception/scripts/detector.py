#!/home/luo/miniconda3/envs/gson/bin/python3

# -*- coding: utf-8 -*-
# Modified by: Shangyi Luo (lsylsy030516@gmail.com)
# Original project: GSON - Group-based Social Navigation Framework
# License: GNU General Public License v3.0 (see LICENSE file)

import cv2
import rospy
import rospkg
import message_filters

from collections import deque
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBoxes, BoundingBox, mapping, tracks

from filter import ConnectionTracker, KeyframeSaver, Mapping_frame, GroupTracker


class DetectorContinuous:

    """
    Synchronizes image, detection, and mapping information, 
    and processes frames to associate detected objects with mapped points.
    """

    def __init__(self):
        
        # Initialize group tracker modules
        # pub Groups msg , and do not pub grouped image
        pkg_path = rospkg.RosPack().get_path('perception')
        
        self.connection_tracker = ConnectionTracker(if_pub=True)    
        
        self.keyframe_saver = KeyframeSaver(
            self.connection_tracker,
            if_publish = False,
            if_node    = False,
            save_dir   = pkg_path + "/keyframes/"
        )
        self.node = GroupTracker(
            self.connection_tracker,
            keyframe_dir= pkg_path + "/keyframes",
            save_dir    = pkg_path + "/keyframes/filter_res"
        )

        # Initialize tool instances
        self.bridge = CvBridge()
        self.image_bbox_buffer = deque(maxlen=10)

        # Initialize ROS subscribers 
        self._initialize_subscribers()
        
        rospy.loginfo("DetectorContinuous Node Initialized Successfully")

    def _initialize_subscribers(self):

        # Subscribe to tracker updates
        rospy.Subscriber("/tracker", tracks, self.connection_tracker.tracker_callback)

        # Subscribe to image and bbox detection topics
        image_sub = message_filters.Subscriber("/yolov5/image_out", Image)
        bbox_sub = message_filters.Subscriber("/yolov5/detections", BoundingBoxes)
        self.image_bbox_sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub], queue_size=10, slop=0.05
        )
        self.image_bbox_sync.registerCallback(self.image_bbox_callback)

        # Subscribe to left/right 2D mapping topics (support single map source if needed)
        if rospy.get_param("~single_camera", False):
            left_map_sub = message_filters.Subscriber("/mapping/left", mapping)
            self.map_sync = message_filters.ApproximateTimeSynchronizer(
                [left_map_sub], queue_size=10, slop=0.05
            )
            self.map_sync.registerCallback(self.mapping_callback_single)
        else:
            left_map_sub = message_filters.Subscriber("/mapping/left", mapping)
            right_map_sub = message_filters.Subscriber("/mapping/right", mapping)
            self.map_sync = message_filters.ApproximateTimeSynchronizer(
                [left_map_sub, right_map_sub], queue_size=10, slop=0.05
            )
            self.map_sync.registerCallback(self.mapping_callback)

    def image_bbox_callback(self, image_msg, bbox_msg):
        """Callback for image + bbox, caches latest synchronized inputs."""
        timestamp = image_msg.header.stamp.to_sec()
        self.image_bbox_buffer.append((timestamp, image_msg, bbox_msg))

    def mapping_callback(self, left_map_msg, right_map_msg):
        """Synchronizes mapping data with cached image/bbox data."""
        if not self.image_bbox_buffer:
            rospy.logwarn("Image/bbox buffer is empty.")
            return

        map_time = left_map_msg.header.stamp.to_sec()
        best_pair = min(
            self.image_bbox_buffer,
            key=lambda x: abs(x[0] - map_time)
        )
        _, image_msg, bbox_msg = best_pair

        curr_map = Mapping_frame(left_map_msg).merge(Mapping_frame(right_map_msg))
        self.process_frame(curr_map, image_msg, bbox_msg)

    def mapping_callback_single(self, map_msg):
        if not self.image_bbox_buffer:
            rospy.logwarn("Image/bbox buffer is empty.")
            return

        map_time = map_msg.header.stamp.to_sec()
        best_pair = min(
            self.image_bbox_buffer,
            key=lambda x: abs(x[0] - map_time)
        )
        _, image_msg, bbox_msg = best_pair

        curr_map = Mapping_frame(map_msg)
        self.process_frame(curr_map, image_msg, bbox_msg)

    def process_frame(self, curr_map, image_msg, bbox_msg):
        rospy.loginfo("Processing synchronized frame...")

        if not bbox_msg.bounding_boxes:
            self.keyframe_saver.synced_callback(image_msg, None)
            rospy.logwarn("No bounding boxes detected. Keyframe saved without annotations.")
            return

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        mapped_id_list, mapped_pose_list = [], []
        mapped_vel_x_list, mapped_vel_y_list = [], []

        # Associate detections with mapped points
        for bbox in bbox_msg.bounding_boxes:
            if bbox.Class != "person":
                continue

            matched_index = -1
            for i in range(len(curr_map.id_list)):
                x_thre = (bbox.xmax - bbox.xmin)
                if bbox.xmin <= curr_map.points[i][0] + x_thre and \
                   bbox.xmax >= curr_map.points[i][0] - x_thre and \
                   bbox.ymin <= curr_map.points[i][1] and \
                   bbox.ymax >= curr_map.points[i][1]:

                    # Optional Draw match
                    # cv2.circle(image, tuple(map(int, curr_map.points[i])), 5, (0, 255, 0), -1)
                    # cv2.putText(
                    #     image, str(curr_map.id_list[i]),
                    #     (int(curr_map.points[i][0]), int(curr_map.points[i][1] - 10)),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    # )
                    
                    # for dashgo system,we prefer to choose the one with the largest x coordinate
                    if matched_index == -1 or curr_map.points[i][0] > curr_map.points[matched_index][0]:    
                        matched_index = i

            if matched_index == -1:
                continue

            mapped_id_list.append(curr_map.id_list[matched_index])
            mapped_pose_list.append(curr_map.points[matched_index])
            mapped_vel_x_list.append(curr_map.vel_x_list[matched_index])
            mapped_vel_y_list.append(curr_map.vel_y_list[matched_index])
            image = self.draw_label(image, bbox, str(curr_map.id_list[matched_index]))

        # Compose new image and bbox message for keyframe
        mapped_img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        mapped_bbox_msg = BoundingBoxes()

        for i in range(len(mapped_id_list)):
            bbox = BoundingBox()
            bbox.Class = str(mapped_id_list[i])
            bbox.xmin = bbox_msg.bounding_boxes[i].xmin
            bbox.xmax = bbox_msg.bounding_boxes[i].xmax
            bbox.ymin = bbox_msg.bounding_boxes[i].ymin
            bbox.ymax = bbox_msg.bounding_boxes[i].ymax
            mapped_bbox_msg.bounding_boxes.append(bbox)

        self.keyframe_saver.synced_callback(mapped_img_msg, mapped_bbox_msg)

    def draw_label(self, image, bbox, text, scale_factor=0.5):
        """Draws bounding box and label text on image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5 * scale_factor
        thickness = int(2 * scale_factor)
        box_thickness = int(2 * scale_factor)

        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), box_thickness)

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 - int(10 * scale_factor) - text_height

        if text_y < 0:
            rect_y1 = y1 + int(10 * scale_factor)
            rect_y2 = rect_y1 + text_height + baseline
            cv2.rectangle(image, (text_x, rect_y1), (text_x + text_width, rect_y2), (255, 0, 0), -1)
            cv2.putText(image, text, (text_x, rect_y2 - baseline), font, font_scale, (255, 255, 255), thickness)
        else:
            cv2.rectangle(image, (text_x, text_y), (text_x + text_width, y1), (255, 0, 0), -1)
            cv2.putText(image, text, (text_x, y1 - int(5 * scale_factor)), font, font_scale, (255, 255, 255), thickness)

        return image


if __name__ == '__main__':
    rospy.init_node('detector_continuous', anonymous=True)
    detector = DetectorContinuous()
    rospy.spin()
