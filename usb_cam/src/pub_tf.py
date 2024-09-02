#!/home/sp/miniconda3/envs/percept/bin/python

import rospy
import tf
import math
from tf2_msgs.msg import TFMessage
import tf2_ros
import geometry_msgs.msg

def static_tf_publish():
    br = tf2_ros.StaticTransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "laser_frame"
    t.child_frame_id = "camera_link" 
    t.transform.translation.x = -0.0025
    t.transform.translation.y = -0.001
    t.transform.translation.z = 0.111
    q = tf.transformations.quaternion_from_euler(0, math.radians(-6), 0)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    br.sendTransform(t)
    # rospy.loginfo("Published static transform from laser_frame to camera_link")

def left_camera_to_laser():
    br = tf2_ros.StaticTransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "laser_frame"
    t.child_frame_id = "usb_cam_l" 
    t.transform.translation.x = 0.0211
    t.transform.translation.y = -0.034
    t.transform.translation.z = 0.306
    q = tf.transformations.quaternion_from_euler(math.radians(90),math.radians(180), math.radians(126))
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    br.sendTransform(t)


def right_camera_to_laser():
    br = tf2_ros.StaticTransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "laser_frame"
    t.child_frame_id = "usb_cam_r" 
    t.transform.translation.x = -0.0211
    t.transform.translation.y = -0.034
    t.transform.translation.z = 0.306
    q = tf.transformations.quaternion_from_euler(math.radians(90),math.radians(180), math.radians(60))
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    br.sendTransform(t)


if __name__ == '__main__':
    rospy.init_node('my_tf_publisher')
    # rospy.loginfo("Published static transform from laser_frame to camera_link")
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        static_tf_publish()
        left_camera_to_laser()
        right_camera_to_laser()
        rate.sleep()