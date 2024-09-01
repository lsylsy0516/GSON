from tf2_ros import buffer, TransformListener
from geometry_msgs.msg import PoseStamped
import rospy

rospy.init_node('receive_tf')
tf_buffer = buffer.Buffer()
tf_listener = TransformListener(tf_buffer)
while not rospy.is_shutdown():
    try:
        pose = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
        print(pose.transform.translation.x, pose.transform.translation.y)
    except Exception as e:
        print(e)
    rospy.sleep(1.0)