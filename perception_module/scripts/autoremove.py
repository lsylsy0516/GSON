#!/home/luo/miniconda3/envs/percept/bin/python

import rospy 
import cv2
from geometry_msgs.msg import PoseStamped,PoseArray,Pose
from nav_msgs.msg import OccupancyGrid
import tf.transformations as tf_trans

import numpy as np
import tf2_ros

import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('perception_module')

class Detection_Remover:
    def __init__(self) -> None:
        self.map_flag = False
        self.detect_sub = rospy.Subscriber("/dr_spaam_detections",PoseArray, self.Detect_cb)
        self.map_sub = rospy.Subscriber("/map",OccupancyGrid,self.Map_cb)
        self.detect_pub = rospy.Publisher("/removed_detections",PoseArray,queue_size=10)
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.map_path = package_path + "/maps/map.pgm"
        self.origin_map = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        # 转为2值图
        self.origin_map = cv2.threshold(self.origin_map, 220, 255, cv2.THRESH_BINARY)[1]
        # 左右翻转
        # self.origin_map = cv2.flip(self.origin_map, 1)
        # 上下翻转
        self.origin_map = cv2.flip(self.origin_map, 0) 
        # 腐蚀 使得障碍物更大 

        # 进行先膨胀后腐蚀,即开运算
        self.costmap = cv2.dilate(self.origin_map, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        self.costmap = cv2.erode(self.costmap, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    def Map_cb(self,msg:OccupancyGrid):
        self.map_flag = True
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.resolution = msg.info.resolution

        self.grid_map = msg.data
        self.grid_map_width = msg.info.width
        self.grid_map_height = msg.info.height

        rospy.loginfo("Get map info")
        self.map_sub.unregister()

    def Detect_cb(self,msg:PoseArray):
        '''
        检测结果回调函数
        '''
        # rospy.loginfo("Get detection")
        if self.map_flag:
            # rospy.loginfo("Get detection")
            pose_array = PoseArray()
            pose_array.header = msg.header
            cnt = 0
            costmap = self.costmap.copy()
            for pose in msg.poses:
                trans = self.buffer.lookup_transform("map","laser_frame",rospy.Time(0),rospy.Duration(10))
                trans_x = trans.transform.translation.x
                trans_y = trans.transform.translation.y
                qtn = (
                            trans.transform.rotation.x,
                            trans.transform.rotation.y,
                            trans.transform.rotation.z,
                            trans.transform.rotation.w
                    )
                trans_roll, trans_pitch, trans_yaw = tf_trans.euler_from_quaternion(qtn)
                map_pose = Pose()
                map_pose.position.x = pose.position.x*np.cos(trans_yaw) - pose.position.y*np.sin(trans_yaw) + trans_x
                map_pose.position.y = pose.position.y*np.cos(trans_yaw) + pose.position.x*np.sin(trans_yaw) + trans_y

                x = int((map_pose.position.x -self.origin_x)/self.resolution)
                y = int((map_pose.position.y - self.origin_y)/self.resolution)
                if costmap[x,y] == 0:
                    cv2.circle(costmap,(x,y),6,200,-1)  # 说明这个边缘误识别
                else:
                    cv2.circle(costmap,(x,y),10,0,-1)
                    pose_array.poses.append(pose)
                    cnt += 1

            # rospy.loginfo("Remove %d detections",cnt)
            costmap = cv2.resize(costmap,(600,600))
            # cv2.imshow("costmap",costmap)
            # cv2.waitKey(1)
            self.detect_pub.publish(pose_array)


if __name__ == "__main__":
    rospy.init_node("detection_remover")
    remover = Detection_Remover()
    while not rospy.is_shutdown():
        rospy.spin()