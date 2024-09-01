#! /usr/bin/env python3

'''
    This is the track module for perception 
    维护一个tracker类，用于追踪动态障碍物
    输入PoseArray,输出一个idlist和新的PoseArray
'''


import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray,Pose
import numpy as np
import tf2_ros
import rospy
import cv2
import sys
from km_tracker.tracker import Tracker
from detection_msgs import tracks
from nav_msgs.msg import OccupancyGrid

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('perception_module')

human_height = 1


class Tracker_Module:
    def __init__(self,if_visualization=True,if_publish=True):# 默认可视化和发布
        self.if_visualization = if_visualization
        self.if_publish = if_publish
        self.tracker_pub = rospy.Publisher("/tracker",tracks,queue_size=1)

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.Global_map_callback,queue_size=10)

        self.tracker = Tracker(
            dist_thresh = 1.0,
            max_frames_to_skip = 30,
            max_trace_length = 3,
            trackIdCount = 0,
            predict_step = 6,
            kf_measurement_noise = 0.1,
            kf_process_noise = 0.5
        )
        rospy.loginfo("Tracker init success")


    def Global_map_callback(self,msg:OccupancyGrid):

        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        rospy.loginfo("origin_x:%f,origin_y:%f",self.origin_x,self.origin_y)
        self.map_sub.unregister()

    def Dynamic_obstacles_cb(self,msg:PoseArray):
        if msg.poses == []: # if there is no dynamic obstacles 
            return [],[]
        
        # 删去重复的Pose
        for i in range(len(msg.poses)):
            for j in range(i+1,len(msg.poses)):
                if msg.poses[i].position.x == msg.poses[j].position.x and msg.poses[i].position.y == msg.poses[j].position.y:
                    msg.poses.pop(j)
                    rospy.loginfo("delete the same pose")
                    break


        try:
            self.test_time1 = rospy.Time.now() # get the time of the callback
        except:
            pass

        self.test_time0 = rospy.Time.now()

        # trans = self.buffer.lookup_transform("map","base_link",rospy.Time(0),rospy.Duration(1.0))
        trans = self.buffer.lookup_transform("map","laser_frame",rospy.Time(0),rospy.Duration(1.0))
        trans_x = trans.transform.translation.x
        trans_y = trans.transform.translation.y
        qtn = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
        )
        trans_roll, trans_pitch, trans_yaw = tf_trans.euler_from_quaternion(qtn)

        # visualization
        map_path = package_path + "/origin_map.png"
        map = cv2.imread(map_path)

        n = len(msg.poses)
        centers = []
        # 接下来将local坐标转换为global坐标
        for i in range(len(msg.poses)):
            local_x = msg.poses[i].position.x
            local_y = msg.poses[i].position.y 

            global_x = local_x*np.cos(trans_yaw) - local_y*np.sin(trans_yaw) + trans_x
            global_y = local_y*np.cos(trans_yaw) + local_x*np.sin(trans_yaw) + trans_y


            # global_x = local_x
            # global_y = local_y

            # rospy.loginfo("global_x:%f,global_y:%f",global_x,global_y)
            # rospy.loginfo("local_x:%f,local_y:%f",local_x,local_y)

            # global_x = -global_x + self.origin_x
            # global_y = global_y + self.origin_y

            centers.append(np.array([[global_x],[global_y]]))
            
            cv2.circle(map,(int(global_y*20),int(global_x*20)),int(5),(0,0,255))
        
        # rospy.loginfo("--------------------")
        self.tracker.Update(centers)
        if len(self.tracker.tracks) == 0: # if there is no track
            return


        # 接下来
        try :
            states = np.array([[[self.tracker.tracks[0].KF.state[0],self.tracker.tracks[0].KF.state[1],0]]])
            for k in range(1,len(self.tracker.tracks)):
                states = np.concatenate((states,np.array([[[self.tracker.tracks[k].KF.state[0],self.tracker.tracks[k].KF.state[1],0]]])),axis=1)

            for n in range(self.tracker.predict_step):
                k_states = np.array([[[self.tracker.tracks[0].KF.future_states[n][0],self.tracker.tracks[0].KF.future_states[n][1],0]]])
                for k in range(1,len(self.tracker.tracks)):

                    k_states = np.concatenate((k_states,np.array([[[self.tracker.tracks[k].KF.future_states[n][0],self.tracker.tracks[k].KF.future_states[n][1],0]]])),axis=1)
                states = np.concatenate((states,k_states),axis=0)
            self.other_agents_states = states

        except Exception as e:
            rospy.logwarn("tracker:")
            rospy.logwarn(len(self.tracker.tracks))
            rospy.logwarn(k)
            rospy.logwarn(self.tracker.tracks[k].KF.future_states)
            rospy.logwarn(e)
            pass

        if self.if_publish:
            tracker_msg = PoseArray()
            tracker_msg.header.stamp = msg.header.stamp
            tracker_msg.header.frame_id = "map"
            for track in self.tracker.tracks:
                pose = Pose()
                pose.position.x = track.KF.state[0]
                pose.position.y = track.KF.state[1]
                # pose.position.z = human_height
                pose.position.z = 0.5
                tracker_msg.poses.append(pose)
            # for pose in tracker_msg.poses:
                # rospy.loginfo("pose:%f",pose.position.z)
            self.tracker_pub.publish(tracker_msg)

        if self.if_visualization:
            # print("track:",len(self.tracker.tracks))

            for i in range(len(self.tracker.tracks)):
                est_x = self.tracker.tracks[i].KF.state[0]
                est_y = self.tracker.tracks[i].KF.state[1]
                cv2.circle(map,(int(est_y*20),int(est_x*20)),int(7),(255,0,0)) # 蓝色代表预测的位置

                for k in range(len(self.tracker.tracks[i].KF.future_states)):
                    x = self.tracker.tracks[i].KF.future_states[k][0]
                    y = self.tracker.tracks[i].KF.future_states[k][1]
                    cv2.circle(map,(int(y*20),int(x*20)),int(7),(255,0,0))  # 蓝色代表预测的位置


            
                if (len(self.tracker.tracks[i].trace) > 1): # 若轨迹长度大于1，则画出轨迹
                    for j in range(len(self.tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = self.tracker.tracks[i].trace[j][0][0]
                        y1 = self.tracker.tracmarker_msgks[i].trace[j][1][0]
                        x2 = self.tracker.tracks[i].trace[j+1][0][0]
                        y2 = self.tracker.tracks[i].trace[j+1][1][0]
                        cv2.line(map, (int(y1*20), int(x1*20)), (int(y2*20), int(x2*20)),(255,0,255), 2)

            cv2.imshow("map.png",map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("shutdown")
                return   

        track_id_list = []
        track_pose_list = []
        
        if len(self.tracker.tracks) == 0:
            rospy.loginfor("No track")
            return track_id_list,track_pose_list
        
        for track in self.tracker.tracks:
            track_id_list.append(track.track_id)
            track_pose_list.append(track.KF.state)
        
        # rospy.loginfo("len(track_id_list):%d",len(track_id_list))
        
        return track_id_list,track_pose_list
        

    def run(self):
        rate = rospy.Rate(10)  
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node("tracker_test")
    tracker_test = Tracker_Module(if_visualization=False,if_publish=True)
    rospy.Subscriber("/removed_detections",PoseArray,tracker_test.Dynamic_obstacles_cb)
    tracker_test.run()
    rospy.spin()

