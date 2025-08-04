import cv2

# 遍历设备索引0到10
for i in range(11):
    # 尝试打开相机
    cap = cv2.VideoCapture(i)
    
    # 检查相机是否打开成功
    if cap.isOpened():
        print(f"Camera {i} is available")
        
        # 读取帧并显示
        ret, frame = cap.read()
        if ret:
            window_name = f"cap_{i}"
            cv2.imshow(window_name, frame)
            cv2.waitKey(1000)  # 显示1秒钟
            
        # 释放相机
        cap.release()
    else:
        print(f"Camera {i} is not available")

# 关闭所有窗口
cv2.destroyAllWindows()
