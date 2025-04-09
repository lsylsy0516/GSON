#!/home/orin/miniconda3/envs/yolo/bin/python
import rospy
from ourplanner.msg import global_path
from detection_msgs.msg import Groups 
from detection_msgs.msg import Group
from detection_msgs.msg import tracks
from ford_msgs.msg import Clusters
import tf.transformations as tf
from std_msgs.msg import Bool
import numpy as np
import tf2_ros
import math

class Trigger:
    def __init__(self):
        # rospy.init_node("trigger_node") 

        self.llm_group_pub = rospy.Publisher("/llm_flag", Bool,queue_size=10)
        self.global_path_sub = rospy.Subscriber("/global_path", global_path, self.global_path_callback)  
        self.global_path = []
        self.group_id_fxxking_history = set()
        self.track_his = {}
        
        self.robo_dis_thre = 10
        self.path_dis_thre = 2.5
        self.track_cnt_thre = 30
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

    def Get_robot_pose(self):
        pose = self.buffer.lookup_transform('map','laser_frame',rospy.Time(0))
        self.robot_x  = pose.transform.translation.x
        self.robot_y  = pose.transform.translation.y

    def tracker_callback(self, msg:tracks):
        if len(msg.track_id_list) <= 2:
            # print("not enough person:{}", len(msg.track_id_list))
            llm_flag = False
            llm_msg = Bool()
            return llm_msg
        
        self.Get_robot_pose()
        
        llm_flag = False

        
        
        # rospy.loginfo(f"{self.group_id_fxxking_history}")
        # print(f"len(msg.track_id_list) : {len(msg.track_id_list)}")
        # print(f"group_id_fxxking_history:{self.group_id_fxxking_history}")
        for i, id in enumerate(msg.track_id_list):
            if id not in self.group_id_fxxking_history:
                    if id in self.track_his.keys():
                        self.track_his[id] += 1
                    else:
                        self.track_his[id] = 1               
                    if self.track_his[id] < self.track_cnt_thre:
                        continue
        
                    # 判断person与机器人pose的距离
                    robot_dis = math.sqrt((self.robot_x - msg.track_pose_list.poses[i].position.x)**2 + (self.robot_y - msg.track_pose_list.poses[i].position.y)**2)
                    if robot_dis < self.robo_dis_thre:
                        closest_point = min(self.global_path, key=lambda point: math.dist(point, [msg.track_pose_list.poses[i].position.x, msg.track_pose_list.poses[i].position.y]))
                        closest_distance = math.dist(closest_point,  [msg.track_pose_list.poses[i].position.x, msg.track_pose_list.poses[i].position.y])
                        # rospy.loginfo(f"human id {msg.track_id_list[i]}")
                        # rospy.loginfo(f"human pose {[msg.track_pose_list.poses[i].position.x, msg.track_pose_list.poses[i].position.y]}")
                        # rospy.loginfo(f"path dis: {closest_distance}")
                        if closest_distance < self.path_dis_thre:
                            rospy.logwarn("need llm group")
                            rospy.logwarn("need llm group")
                            rospy.logwarn("need llm group")
                            llm_flag = True
                            llm_msg = Bool()
                            llm_msg.data = llm_flag
                            self.llm_group_pub.publish(llm_msg)
                            for id in msg.track_id_list:
                                self.group_id_fxxking_history.add(id)
                            return llm_msg                     
        llm_msg = Bool()
        llm_msg.data = llm_flag
        self.llm_group_pub.publish(llm_msg)
        return llm_msg
      
    def group_callback(self, msg:Groups):
        # rospy.loginfo(f"{msg.group_list}")
        for group in msg.group_list:
            group:Group
            for id in group.group_id_list:
                self.group_id_fxxking_history.add(id)

    def global_path_callback(self, msg: global_path):
        rospy.loginfo("get global path")
        global_path = []
        for i in range(msg.length):
            global_path.append([msg.path_x[i], msg.path_y[i]])
        self.global_path = global_path
        self.global_path_sub.unregister()

if __name__ == '__main__':
    rospy.loginfo("start trigger")
    trigger = Trigger()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown:
        rate.sleep()
    rospy.spin()