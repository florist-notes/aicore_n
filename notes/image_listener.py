#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageListener(Node):

    def __init__(self):
        super().__init__('image_listener')
        self.subscription = self.create_subscription(
            Image,
            '/overhead_camera/overhead_camera3/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('Receiving image')
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow("Camera Image", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
