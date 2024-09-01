import sys
import rospy
from detection_msgs.msg import tracks
from nav_msgs.msg import Odometry
import time
import numpy as np
import warnings

# 全局变量用于存储位置和速度数据
pose = []
vel = []
flag_cal = False

# 里程计回调函数，用于接收里程计数据
def odom_callback(msg):
    global flag_cal
    # 从里程计消息中提取位置和速度数据
    pos_x = msg.pose.pose.position.x
    pos_y = msg.pose.pose.position.y
    pos_z = msg.pose.pose.orientation.z
    vel_x = msg.twist.twist.linear.x
    vel_y = msg.twist.twist.linear.y
    # 将位置和速度数据存入全局列表中
    pose.append([pos_x, pos_y, pos_z])
    vel.append([vel_x, vel_y])
    # 当位置达到一定阈值且还未开始计算时，设置标志位为True
    if pos_x > 27 and not flag_cal:
        flag_cal = True
        # 开始计算曲率、粗糙度和加速度
        get_curvature(pose)
        get_roughness(pose)
        get_jerk(vel)

# 计算加速度变化
def calc_jerk(v1, v2, v3):
    # 计算速度模值
    v1 = (v1[0]**2 + v1[1]**2)**0.5
    v2 = (v2[0]**2 + v2[1]**2)**0.5
    v3 = (v3[0]**2 + v3[1]**2)**0.5
    # 计算加速度
    a1 = v2 - v1
    a2 = v3 - v2
    # 计算加速度变化（即 jerk）
    jerk = np.abs(a2 - a1)
    acc = a1
    return jerk, acc

# 计算曲率
def calc_curvature(x, y, z):
    # 计算三角形面积
    triangle_area = 0.5 * np.abs(x[0]*(y[1]-z[1]) + y[0]*(z[1]-x[1]) + z[0]*(x[1]-y[1]))
    # 计算曲率
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        curvature = 4 * triangle_area / (np.abs(np.linalg.norm(x-y)) * np.abs(np.linalg.norm(y-z)) * np.abs(np.linalg.norm(z-x)))
        # 计算标准化曲率
        normalized_curvature = curvature * (np.abs(np.linalg.norm(x-y)) + np.abs(np.linalg.norm(y-z)))
    return [curvature, normalized_curvature]

# 计算粗糙度
def calc_roughness(x, y, z):
    # 计算三角形面积
    triangle_area = 0.5 * abs(x[0]*(y[1]-z[1]) + y[0]*(z[1]-x[1]) + z[0]*(x[1]-y[1]))
    # 计算粗糙度
    roughness = 2 * triangle_area / np.abs((z[0]-x[0])**2 + (z[1]-x[1])**2)
    return roughness

# 计算并打印粗糙度
def get_roughness(pose):
    roughness_list = []
    for i in range(1, len(pose)):
        try:
            x = np.array(pose[i])
            y = np.array(pose[i + 1])
            z = np.array(pose[i + 2])
            roughness_list.append(calc_roughness(x, y, z))
            continue
        except:
            continue
    print("roughness:", sum(roughness_list) / len(roughness_list))
    return roughness_list

# 计算并打印曲率和标准化曲率
def get_curvature(pose):
    curvature_list = []
    normalized_curvature_list = []
    for i in range(1, len(pose)):
        try:
            x = np.array(pose[i])
            y = np.array(pose[i + 1])
            z = np.array(pose[i + 2])
            curvature_list.append(calc_curvature(x, y, z)[0])
            normalized_curvature_list.append(calc_curvature(x, y, z)[1])
            continue
        except:
            continue
    print("curvature:", sum(curvature_list) / len(curvature_list))
    print("normalized_curvature:", sum(normalized_curvature_list) / len(normalized_curvature_list))
    return curvature_list, normalized_curvature_list

# 计算并打印加速度和速度模值
def get_jerk(vel):
    jerk_list, acc_list, vel_list = [], [], []
    for i in range(1, len(vel)):
        try:
            v1 = np.array(vel[i])
            v2 = np.array(vel[i + 1])
            v3 = np.array(vel[i + 2])
            jerk, acc = calc_jerk(v1, v2, v3)
            jerk_list.append(jerk)
            acc_list.append(acc)
            continue
        except:
            continue
    # 计算速度模值
    for i, j in vel:
        vel_list.append((i**2 + j**2)**(1/2))
    # 打印加速度和速度模值
    print("jerk:", sum(jerk_list) / len(jerk_list))
    return jerk_list, acc_list, vel_list

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node('find_pose_node')
    # 订阅里程计消息
    rospy.Subscriber('/odom', Odometry, odom_callback)
    # 进入消息处理循环
    while not rospy.is_shutdown():
        rospy.spin()
