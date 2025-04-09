#!/home/orin/miniconda3/envs/yolo/bin/python


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Header

def publish_image():
    # 初始化ROS节点
    rospy.init_node('image_publisher', anonymous=True)
    
    # 创建图像发布者
    image_pub = rospy.Publisher('/goal_image', Image, queue_size=10)
    
    # 使用CvBridge进行OpenCV图像和ROS图像的转换
    bridge = CvBridge()
    
    # 读取图像
    image_path = '/home/orin/planner_ws/src/perception_module/scripts/goal_image.jpg'
    cv_image = cv2.imread(image_path)
    
    if cv_image is None:
        rospy.logerr("Failed to load image from path: %s" % image_path)
        return
    
    # 设置发布频率
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        # 创建ROS消息头部
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_frame"

        # 将OpenCV图像转换为ROS的Image消息
        ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        ros_image.header = header  # 给消息添加头部信息

        # 发布图像
        image_pub.publish(ros_image)

        # 按照设定的频率发布图像
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass
