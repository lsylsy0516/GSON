from geometry_msgs.msg import PoseArray,Point, Pose
from visualization_msgs.msg import  Marker, MarkerArray
import rospy
import numpy as np

def marker_create(posearray:PoseArray, marker_topic:str,msg_id:int,color:list,height:float)-> Marker:
    
    marker_msg = Marker()
    marker_msg.header.frame_id = marker_topic
    marker_msg.action = Marker.ADD
    marker_msg.type = Marker.LINE_LIST
    marker_msg.ns = marker_topic # namespace
    marker_msg.id = msg_id

    marker_msg.pose.orientation.x = 0.0
    marker_msg.pose.orientation.y = 0.0
    marker_msg.pose.orientation.z = 0.0
    marker_msg.pose.orientation.w = 1.0

    marker_msg.scale.x = 0.03  # line width
    marker_msg.color.r = color[0]
    marker_msg.color.g = color[1]
    marker_msg.color.b = color[2]
    marker_msg.color.a = 1

    # circle 
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)
    for pose in posearray.poses:
        for i in range(len(xy_offsets) - 1):
            p0 = Point()
            p0.x = pose.position.x + xy_offsets[i, 0]
            p0.y = pose.position.y + xy_offsets[i, 1]
            p0.z = height
            marker_msg.points.append(p0)

            p1 = Point()
            p1.x = pose.position.x + xy_offsets[i + 1, 0]
            p1.y = pose.position.y + xy_offsets[i + 1, 1]
            p1.z = height
            marker_msg.points.append(p1)
    
    return marker_msg

def marker_array_create(ID_list:list, Pose_list:list, marker_topic:str,color:list,height:float)-> MarkerArray:
    # start = rospy.Time.now()
    marker_array = MarkerArray()

    for i in range(len(ID_list)):
        marker_msg = Marker()
        marker_msg.header.frame_id = "map"
        # marker_msg.header.frame_id = "laser_frame"
        marker_msg.header.stamp = rospy.Time.now()

        marker_msg.ns = marker_topic
        marker_msg.id = i

        marker_msg.type = Marker.TEXT_VIEW_FACING
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = Pose_list[i][0]
        marker_msg.pose.position.y = Pose_list[i][1]
        marker_msg.pose.position.z = height

        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

        marker_msg.scale.z = 0.4  # 设置文本高度
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]
        marker_msg.color.a = 1

        marker_msg.text = str(ID_list[i])
        marker_array.markers.append(marker_msg)
    # end = rospy.Time.now()
    # rospy.loginfo("marker time cost:%f",end.to_sec()-start.to_sec())
    
    return marker_array
