import torch
import cv2
import numpy as np

import sys
import os
import threading

sys.path.append(
    os.path.join("/home/gw/Posture-Gesture-Recognition")
)


from models.tcn import TCN
from runtime.frame_inferencer import TCNFrameInferencer

from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from inha_interfaces.msg import ImageBbox
from inha_interfaces.srv import GestureDetection
from vision_msgs.msg import BoundingBox2D
from ament_index_python.packages import get_package_share_directory

class GestureDetectNode(Node):
    def __init__(self):
        super().__init__('gesture_detection_node')

        # ROS QoS Profiles
        self.qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Parameters
        self.declare_parameter('device','cuda')
        self.declare_parameter('det_frequency',1)
        self.declare_parameter('buffer_size', 60)
        self.declare_parameter('track_ttl', 45)
        self.declare_parameter('track_match_distance', 120)
        self.declare_parameter('mode', 'balanced', choices=['performance', 'balanced', 'lightweight'])
        self.declare_parameter('max_persons_inference', 8)

        # ROS2 Parameters
        self.declare_parameter('package_name', 'posture_and_gesture')
        self.declare_parameter('input_topic','/camera/camera_head/color/image_raw/compressed')
        self.declare_parameter('image_transport','compressed')
        self.declare_parameter('output_topic', '/gesture_detection')

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.det_frequency = self.get_parameter('det_frequency').get_parameter_value().integer_value
        self.track_ttl = self.get_parameter('track_ttl').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.track_match_distance = self.get_parameter('track_match_distance').get_parameter_value().integer_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.max_persons_inference = self.get_parameter('max_persons_inference').get_parameter_value().integer_value

        self.package_name = self.get_parameter('package_name').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.image_transport = self.get_parameter('image_transport').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.get_logger().info(
            f"""PostureDetectNode initialized with device={self.device},
            det_frequency={self.det_frequency}, 
            track_ttl={self.track_ttl}, 
            track_match_distance={self.track_match_distance}, 
            mode={self.mode}, 
            max_persons_inference={self.max_persons_inference}"""
        )

        self.pub = self.create_publisher(
            ImageBbox, 
            self.output_topic,
            self.qos_best_effort
        )

        self.gesture_service = self.create_service(
            GestureDetection,
            'gesture_detection_service',
            self.gesture_detection_callback
        )

        self.get_logger().info(
            f"""Subscribing to {self.input_topic} 
            with transport {self.image_transport}.
            Publishing results to {self.output_topic}."""
        )

        # Initialize model
        package_name = get_package_share_directory(self.package_name)
        model_path = os.path.join(package_name, 'models', 'best_tcn_xsub.pth')

        if torch.cuda.is_available() and self.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.get_logger().warn("CUDA not available, using CPU instead.")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = TCN(
            input_dim=checkpoint.get("input_dim", 34),
            num_classes=checkpoint.get("num_classes", 5),
            hidden_dims=checkpoint.get("hidden_dims", [64, 128, 128, 256]),
            kernel_size=checkpoint.get("kernel_size", 5),
            dropout=checkpoint.get("dropout", 0.3),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.inferencer = TCNFrameInferencer(
            model=self.model,
            device=self.device,
            num_classes=checkpoint.get("num_classes", 5),
            max_frames=checkpoint.get("max_frames", 120),
            num_joints=checkpoint.get("num_joints", 65),
            pose_device=str(self.device),
        )
        
        # Event Trigger
        self._action_active = False
        self.target_class = []
        
        
    def start_subscription(self):
        self.sub = self.create_subscription(
            CompressedImage if self.image_transport == 'compressed' else Image,
            self.input_topic,
            self.image_callback,
            self.qos_best_effort
        )

    def image_callback(self, msg):
        """
        ImageBbox Message Structure:
        
        Header header
        string frame_id
        time stamp

        Image image
        BoundingBox2D[] bboxes
        string[] track_id
        string[] det_index
        string[] class_name
        """
        if not self._action_active:
            return
        
        if self.image_transport == 'compressed':
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image frame.")
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # Inference Posture with MLP
        result = self.inferencer.infer(frame, return_debug_image=True)

        ### result -> detections, debug_image, keypoints, scores
        ### result.detections -> track_id, det_index, class_name, confidence, keypoint_scores
        ###                      probabilities, center_xy, bbox_xyxy, skeleton_xy

        out = ImageBbox()
        out.header = msg.header
        out.header.frame_id = "camera_head_color_optical_frame"
        out.header.stamp = self.get_clock().now().to_msg()

        dbg_bgr = cv2.cvtColor(result.debug_image, cv2.COLOR_RGB2BGR)
        out.image.height = int(dbg_bgr.shape[0])
        out.image.width = int(dbg_bgr.shape[1])
        out.image.encoding = 'bgr8'
        out.image.step = int(dbg_bgr.shape[1] * 3)
        out.image.data = dbg_bgr.tobytes()

        for det in result.detections:
            track_id, det_index, class_name, confidence, keypoint_scores, probs, center_xy, bbox_xyxy, skeleton_xy = det

            if self.target_class and class_name not in self.target_class:
                continue

            bb = BoundingBox2D()
            bb.center.x = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)
            bb.center.y = float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)
            bb.size_x = float(bbox_xyxy[2] - bbox_xyxy[0])
            bb.size_y = float(bbox_xyxy[3] - bbox_xyxy[1])

            out.bboxes.append(bb)
            out.track_id.append(str(track_id))
            out.det_index.append(str(det_index))
            out.class_name.append(str(class_name))

        self.pub.publish(out)


    

    def gesture_detection_callback(self, request, response):
        """
        GestureDetection Service:
        - Request:
            bool start
            string[] target_class
        - Response:
            bool success
            string message
        """
        if request.start:
            self.target_class = list(request.target_class)
            self.get_logger().info(f"Received gesture detection request for class: {self.target_class}")
            self._start_action() 
        else:
            self.get_logger().info("Received gesture detection stop request.")
            self._stop_action()
        
        response.success = True
        response.message = "Gesture detection started." if request.start else "Gesture detection stopped."
        return response

    def _start_action(self):
        if not self._action_active:
            self.get_logger().info("Starting gesture detection service.")
            self._action_active = True
            self.start_subscription()

    def _stop_action(self):
        if self._action_active:
            self.get_logger().info("Stopping gesture detection service.")
            self._action_active = False
            self.target_class = []
            self.destroy_subscription(self.sub)

def main(args=None):
    rclpy.init(args=args)
    node = GestureDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
