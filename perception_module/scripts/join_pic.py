#!/home/sp/miniconda3/envs/percept/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from collections import deque
from std_msgs.msg import Header

class ImageMergerNode:
    def __init__(self):
        self.left_images = deque(maxlen=5)
        self.right_images = deque(maxlen=5)
        self.bridge = CvBridge()

        self.merged_image_pub = rospy.Publisher('/merged_image', Image, queue_size=10)
        self.left_image_sub = rospy.Subscriber('/usb_cam_l/image_raw', Image, self.left_image_callback)
        self.right_image_sub = rospy.Subscriber('/usb_cam_r/image_raw', Image, self.right_image_callback)
        self.header = Header()
        rospy.loginfo('Image merger node started')

    def left_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.left_images.append(cv_image)
        self.header = msg.header
    def right_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.right_images.append(cv_image)

    def merge_images(self):
        if len(self.left_images) >= 2 and len(self.right_images) >= 2:
            left_image = self.left_images[-1]
            right_image = self.right_images[-1]
            merged_image = cv2.hconcat([left_image, right_image])
            merged_image = cv2.resize(merged_image, (int(merged_image.shape[1] * 360 / merged_image.shape[0]), 360))

            merged_image_msg = self.bridge.cv2_to_imgmsg(merged_image, encoding='bgr8')
            merged_image_msg.header = self.header
            self.merged_image_pub.publish(merged_image_msg)
            # cv2.imshow('merged_image', merged_image)
            # cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            self.merge_images()
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('image_merger_node')
    image_merger_node = ImageMergerNode()
    image_merger_node.run()