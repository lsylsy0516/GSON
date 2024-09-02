#!/home/sp/miniconda3/envs/percept/bin/python

from detection_msgs.msg import tracks,mapping
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Pose,PoseStamped
import rospy
import numpy as np
import cv2
import tf
import tf2_ros
import tf.transformations as tf_trans
import math

class Mapping_node:
    def __init__(self,left_flag):
        self.listener = tf.TransformListener()
        self.left_flag = left_flag
        self.buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.buffer)
        if left_flag:
            self.camera_info_sub = rospy.Subscriber("/usb_cam_l/camera_info",CameraInfo,self.camera_info_cb)
            self.tracks_sub = rospy.Subscriber("/tracker",tracks,self.tracks_cb)
            self.mapping_pub = rospy.Publisher("/mapping/left",mapping,queue_size=1)
        else:
            self.camera_info_sub = rospy.Subscriber("/usb_cam_r/camera_info",CameraInfo,self.camera_info_cb)
            self.tracks_sub = rospy.Subscriber("/tracker",tracks,self.tracks_cb)
            self.mapping_pub = rospy.Publisher("/mapping/right",mapping,queue_size=1)
        rospy.loginfo("Mapping node init success")



    def camera_info_cb(self,msg:CameraInfo):
        P = np.array(msg.P).reshape(3,4)
        # print("P:", P)
        # print("Shape of P:", P.shape)
        try:
            camera_matrix,rot_matrix,tvec,_, _, _, _ = cv2.decomposeProjectionMatrix(P)
        except cv2.error as e:
            print("Error:", e)
        tvec = tvec[:3] 
        rvec, _ = cv2.Rodrigues(rot_matrix)
        distortion_coefficients = np.array(msg.D)
        self.camera_matrix,self.distortion_coefficients,self.rvec,self.tvec = camera_matrix,distortion_coefficients,rvec,tvec
        self.camera_info_sub.unregister()

    def tracks_cb(self,msg:tracks):
        id_list = []
        point_xs = []
        point_ys = []
        vel_x_list = []
        vel_y_list = []
        pose_list = PoseArray()

        for track,id,vel_x,vel_y in zip(msg.track_pose_list.poses,msg.track_id_list,msg.track_vel_x_list,msg.track_vel_y_list):
            try:
                laser_pose = PoseStamped()
                laser_pose.header.frame_id = "laser_frame"
                laser_pose.header.stamp = msg.header.stamp
                trans = self.buffer.lookup_transform("laser_frame","map",rospy.Time(0),rospy.Duration(1.0))
                trans_x = trans.transform.translation.x
                trans_y = trans.transform.translation.y
                qtn = (
                        trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w
                )
                trans_roll, trans_pitch, trans_yaw = tf_trans.euler_from_quaternion(qtn)
                laser_pose.pose.position.x = track.position.x*np.cos(trans_yaw) - track.position.y*np.sin(trans_yaw) + trans_x
                laser_pose.pose.position.y = track.position.y*np.cos(trans_yaw) + track.position.x*np.sin(trans_yaw) + trans_y
                laser_pose.pose.position.z = 1
                if self.left_flag:
                    self.listener.waitForTransform("laser_frame", "usb_cam_l", rospy.Time(0), rospy.Duration(1.0))
                    left_pose = self.listener.transformPose("usb_cam_l",laser_pose)
                    left_pose = np.array([left_pose.pose.position.x, left_pose.pose.position.y, left_pose.pose.position.z])
                    if left_pose[2] < 0 or math.sqrt(left_pose[0]**2 + left_pose[1]**2) > 10:
                        continue
                    point,_ = cv2.projectPoints(left_pose, self.rvec, self.tvec, self.camera_matrix, self.distortion_coefficients)
                    point = point[0][0]
                    if point[0] < 0 or point[0] > 1280 or point[1] < 0 or point[1] > 720:
                        continue
                    else:
                        point[0] /= 2
                        point[1] /= 2
                        id_list.append(id)
                        point_xs.append(int(point[0]))
                        point_ys.append(int(point[1]))
                        pose_list.poses.append(track)
                        vel_x_list.append(vel_x)
                        vel_y_list.append(vel_y)
                else:
                    self.listener.waitForTransform("laser_frame", "usb_cam_r", rospy.Time(0), rospy.Duration(1.0))
                    right_pose = self.listener.transformPose("usb_cam_r",laser_pose)
                    right_pose = np.array([right_pose.pose.position.x, right_pose.pose.position.y, right_pose.pose.position.z])
                    point,_ = cv2.projectPoints(right_pose, self.rvec, self.tvec, self.camera_matrix, self.distortion_coefficients)
                    point = point[0][0]
                    if point[0] < 0 or point[0] > 1280 or point[1] < 0 or point[1] > 720:
                        continue
                    else:
                        point[0] /= 2
                        point[1] /= 2
                        point[0] += 640
                        id_list.append(id)
                        point_xs.append(int(point[0]))
                        point_ys.append(int(point[1]))
                        pose_list.poses.append(track)
                        vel_x_list.append(vel_x)
                        vel_y_list.append(vel_y)
            except Exception as e:
                rospy.logwarn(e)
                continue
                
                    
        
        mapping_msg = mapping()
        mapping_msg.header.stamp = msg.header.stamp
        mapping_msg.id_list = id_list
        mapping_msg.point_xs = point_xs
        mapping_msg.point_ys = point_ys
        mapping_msg.pose_list = pose_list
        mapping_msg.vel_x_list = vel_x_list
        mapping_msg.vel_y_list = vel_y_list
        try :
            self.mapping_pub.publish(mapping_msg)
        except Exception as e:
            rospy.logwarn(e)
if __name__ == "__main__":
    rospy.init_node("mapping_node")
    left_mapping_node = Mapping_node(left_flag=True)
    right_mapping_node = Mapping_node(left_flag=False)
    rospy.spin()