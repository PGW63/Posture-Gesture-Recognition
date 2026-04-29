import torch
import cv2
import numpy as np

import sys
import os
import threading
from pathlib import Path

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (
            (parent / 'models' / 'tcn.py').exists()
            and (parent / 'runtime' / 'frame_inferencer.py').exists()
            and (parent / 'data').exists()
        ):
            return parent
    raise RuntimeError(f"Could not locate repository root from {current}")


REPO_ROOT = _find_repo_root()
VENDORED_RTMLIB_ROOT = REPO_ROOT / 'rtmlib'
for path in (REPO_ROOT, VENDORED_RTMLIB_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from models.tcn import TCN
from runtime.frame_inferencer import TCNFrameInferencer

from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from inha_interfaces.srv import GestureDetection
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose
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
        self.declare_parameter('mode', 'balanced')
        self.declare_parameter('max_persons_inference', 8)

        # ROS2 Parameters
        self.declare_parameter('package_name', 'Posture_and_Gesture')
        self.declare_parameter('input_topic','/camera/camera_head/color/image_raw/compressed')
        self.declare_parameter('image_transport','compressed')
        self.declare_parameter('output_topic', '/gesture_detection')
        self.declare_parameter('output_image_topic', '')
        self.declare_parameter('output_detection_topic', '')

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.det_frequency = self.get_parameter('det_frequency').get_parameter_value().integer_value
        self.track_ttl = self.get_parameter('track_ttl').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.track_match_distance = self.get_parameter('track_match_distance').get_parameter_value().integer_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.max_persons_inference = self.get_parameter('max_persons_inference').get_parameter_value().integer_value

        valid_modes = {'performance', 'balanced', 'lightweight'}
        if self.mode not in valid_modes:
            self.get_logger().warn(
                f"Unsupported mode '{self.mode}', falling back to 'balanced'. "
                f"Valid values: {sorted(valid_modes)}"
            )
            self.mode = 'balanced'

        self.package_name = self.get_parameter('package_name').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.image_transport = self.get_parameter('image_transport').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        output_detection_topic = self.get_parameter('output_detection_topic').get_parameter_value().string_value
        self.output_image_topic = output_image_topic or f'{self.output_topic}/image'
        self.output_detection_topic = output_detection_topic or f'{self.output_topic}/detections'

        self.get_logger().info(
            f"""PostureDetectNode initialized with device={self.device},
            det_frequency={self.det_frequency}, 
            track_ttl={self.track_ttl}, 
            track_match_distance={self.track_match_distance}, 
            mode={self.mode}, 
            max_persons_inference={self.max_persons_inference}"""
        )

        self.image_pub = self.create_publisher(
            Image,
            self.output_image_topic,
            self.qos_best_effort
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.output_detection_topic,
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
            Publishing image to {self.output_image_topic}.
            Publishing detections to {self.output_detection_topic}."""
        )

        # Initialize model
        model_path = REPO_ROOT / 'models' / 'best_tcn_xsub.pth'

        if torch.cuda.is_available() and self.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.get_logger().warn("CUDA not available, using CPU instead.")

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)
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
        if not self._action_active:
            return
        
        if self.image_transport == 'compressed':
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image frame.")
                return
        else:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        

        # Inference Posture with MLP
        result = self.inferencer.infer(frame, return_debug_image=True)

        ### result -> detections, debug_image, keypoints, scores
        ### result.detections -> track_id, det_index, class_name, confidence, keypoint_scores
        ###                      probabilities, center_xy, bbox_xyxy, skeleton_xy

        header = msg.header
        header.frame_id = "camera_head_color_optical_frame"
        header.stamp = self.get_clock().now().to_msg()

        dbg_rgb = cv2.cvtColor(result.debug_image, cv2.COLOR_BGR2RGB)
        image_msg = Image()
        image_msg.header = header
        image_msg.height = int(dbg_rgb.shape[0])
        image_msg.width = int(dbg_rgb.shape[1])
        image_msg.encoding = 'rgb8'
        image_msg.is_bigendian = False
        image_msg.step = int(dbg_rgb.shape[1] * 3)
        image_msg.data = dbg_rgb.tobytes()

        detection_array = Detection2DArray()
        detection_array.header = header

        for det in result.detections:
            if self.target_class and det.class_name not in self.target_class:
                continue

            if det.bbox_xyxy is None:
                continue

            bb = BoundingBox2D()
            bb.center.position.x = float((det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0)
            bb.center.position.y = float((det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2.0)
            bb.size_x = float(det.bbox_xyxy[2] - det.bbox_xyxy[0])
            bb.size_y = float(det.bbox_xyxy[3] - det.bbox_xyxy[1])

            detection = Detection2D()
            detection.header = header
            detection.bbox = bb
            detection.id = f"track:{det.track_id};det:{det.det_index}"

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det.class_name)
            hypothesis.hypothesis.score = float(det.confidence)
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

        self.image_pub.publish(image_msg)
        self.detection_pub.publish(detection_array)


    

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
