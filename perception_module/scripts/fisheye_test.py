#!/home/sp/miniconda3/envs/percept/bin/python

import cv2
import numpy as np

FOVD = 210

class FisheyeSticher:
    def __init__(self) -> None:
        # if debug or not
        self.debug = True
        # hard code for now
        self.fovd = 210
        self.inner_fovd = 180
        self.ws = 720
        self.hs = 680
        self.wd = int(self.ws * 360 / FOVD)
        self.hd = int(self.wd / 2)

        self.fish2Map()
        self.createMask()

    def fish2Eqt(self, x_d, y_d, w_rad) -> tuple:
        # 球面坐标
        phi = x_d / w_rad
        theta = -y_d / w_rad + np.pi / 2
        # 球面坐标转换为直角坐标
        if theta < 0:   # 若theta小于0，说明在下半球，需要转换
            theta = -theta
            phi += np.pi    # phi加上pi，表示在下半球

        if theta > np.pi:   # 若theta大于pi，说明在下半球，需要转换
            theta = np.pi - (theta - np.pi)
            phi += np.pi
        # 计算球面坐标的直角坐标
        s = np.sin(theta)
        v0 = s * np.sin(phi) 
        v1 = np.cos(theta)
        r = np.sqrt(v0 * v0 + v1 * v1)
        theta = w_rad * np.arctan2(r, s * np.cos(phi))

        x_src = theta * v0 / r
        y_src = theta * v1 / r
        return x_src, y_src

    def fish2Map(self) -> None:
        mapx = np.zeros((self.hd, self.wd), dtype=np.float32)
        mapy = np.zeros((self.hd, self.wd), dtype=np.float32)
        w_rad = self.wd /(2 * np.pi)     # 宽度的弧度
        
        w2 = self.wd/2 - 0.5 # 宽度的一半
        h2 = self.hd/2 - 0.5
        ws2 = self.ws/2 - 0.5
        hs2 = self.hs/2 - 0.5

        for y in range(self.hd):
            y_d = y - h2 # y相对中心的偏移
            for x in range(self.wd):
                x_d = x - w2 # x的偏移

                x_src, y_src = self.fish2Eqt(x_d, y_d, w_rad)

                x_src += ws2
                y_src += hs2

                mapx[y, x] = x_src
                mapy[y, x] = y_src
        
        self.mapx = mapx
        self.mapy = mapy

    def createMask(self) -> None:
        cir_mask = np.zeros((self.hs, self.ws), dtype=np.uint8)
        inner_cir_mask = np.zeros((self.hs, self.ws), dtype=np.uint8)
        wShift = int((self.ws*(self.fovd - self.inner_fovd)/self.fovd)/2)
        r1 = self.ws//2
        r2 = r1- wShift*2
        cv2.circle(cir_mask, (self.ws//2, self.hs//2), int(r1), 255, -1)
        cv2.circle(inner_cir_mask, (self.ws//2, self.hs//2), int(r2), 255, -1)
        self.circle_mask = cir_mask
        self.inner_circle_mask = inner_cir_mask

        # if self.debug:
        #     cv2.imwrite("circle_mask.jpg", cir_mask)
        #     cv2.imwrite("inner_circle_mask.jpg", inner_cir_mask)
        #     self.test_unwarp()

    def unwarp(self, src:cv2.Mat) -> np.ndarray:
        equi = cv2.remap(src, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0))
        return equi

    def test_unwarp(self):
        test_img = self.unwarp(self.circle_mask)
        cv2.imwrite("test_unwarp.jpg", test_img)

    def stitch_test(self,img:cv2.Mat) -> cv2.Mat:
        # 进行mask操作
        img = cv2.bitwise_and(img,img, mask=self.circle_mask)
        # 假设会做光照补偿
        img = img
        # fisheye unwarp
        img_unwarp = self.unwarp(img)
        # cv2.Rect(img_unwarp, (int(self.wd/2 - self.ws/2), 0), (self.ws, self.hd), (0, 0, 0), -1)
        img_crop = img_unwarp[:, int(self.wd/2 - self.ws/2):(self.ws+int(self.wd/2 - self.ws/2))]
        if self.debug:
            cv2.imwrite("img_crop.jpg", img_crop)
            cv2.imwrite("img_unwarp.jpg", img_unwarp)
            cv2.imwrite("img.jpg", img)
    
    def stitch(self, l_ori:cv2.Mat,r_ori:cv2.Mat) -> cv2.Mat:
        # 进行mask操作
        # print(self.circle_mask.shape)
        l_ori = cv2.bitwise_and(l_ori,l_ori, mask=self.circle_mask)
        r_ori = cv2.bitwise_and(r_ori,r_ori, mask=self.circle_mask)
        # 假设会做光照补偿
        l_ori = l_ori
        r_ori = r_ori
        # fisheye unwarp
        l_unwarp = self.unwarp(l_ori)
        r_unwarp = self.unwarp(r_ori)
        # cv2.Rect(img_unwarp, (int(self.wd/2 - self.ws/2), 0), (self.ws, self.hd), (0, 0, 0), -1)
        l_crop = l_unwarp[:, int(self.wd/2 - self.ws/2):(self.ws+int(self.wd/2 - self.ws/2))]
        r_crop = r_unwarp[:, int(self.wd/2 - self.ws/2):(self.ws+int(self.wd/2 - self.ws/2))]        
        # 进行拼接
        stitch = cv2.hconcat([l_crop, r_crop])
        
        if self.debug:
            cv2.imwrite("l_crop.jpg", l_crop)
            cv2.imwrite("r_crop.jpg", r_crop)
            cv2.imwrite("l_unwarp.jpg", l_unwarp)
            cv2.imwrite("r_unwarp.jpg", r_unwarp)
            cv2.imwrite("l_ori.jpg", l_ori)
            cv2.imwrite("r_ori.jpg", r_ori)
            cv2.imwrite("stitch.jpg", stitch)
        return stitch


    # fisheyesticher = FisheyeSticher()
    # video_pkg_path ="../videos/"
    # img_pkg_path = "../images/"
    # img_path = img_pkg_path + "test.jpg"
    # img = cv2.imread(img_path)
    
    # l_ori = img[:, 0:720]
    # r_ori = img[:, 720:]
    # stitched = fisheyesticher.stitch(l_ori, r_ori) 
    # cv2.imshow("stitched", stitched)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
        self.fisheyesticher = FisheyeSticher()

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
            # 提取左右图像中间720像素 从左到右(280，960)
            l_ori = left_image[:, 280:960]
            r_ori = right_image[:, 280:960]
            # print(l_ori.shape)
            r_rotated = cv2.rotate(l_ori, cv2.ROTATE_90_COUNTERCLOCKWISE)
            l_rotated = cv2.rotate(r_ori, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 720*720 ，向右旋转90度
            # l_rotated = cv2.rotate(l_ori, cv2.ROTATE_90_CLOCKWISE)
            # r_rotated = cv2.rotate(r_ori, cv2.ROTATE_90_CLOCKWISE)
            # print(l_rotated.shape)
            # print(r_rotated.shape)
            # merged_image = cv2.hconcat([l_rotated, r_rotated])
            # merged_image = cv2.resize(merged_image, (int(merged_image.shape[1] * 360 / merged_image.shape[0]), 360))
            merged_image = self.fisheyesticher.stitch(l_rotated, r_rotated)

            merged_image_msg = self.bridge.cv2_to_imgmsg(merged_image, encoding='bgr8')
            merged_image_msg.header = self.header
            self.merged_image_pub.publish(merged_image_msg)
            cv2.imshow('merged_image', merged_image)
            a = cv2.waitKey(1)
            if a  == 115:
                print("sss")
                cv2.imwrite("/home/sp/planner_ws/src/perception_module/scripts/goal_image.jpg",merged_image)
            else:
                print(a)
    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            self.merge_images()
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('image_merger_node')
    image_merger_node = ImageMergerNode()
    image_merger_node.run()