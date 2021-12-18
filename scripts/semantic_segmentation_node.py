#!/usr/bin/python3

# ROS
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Image processing
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import rescale

from modules import detectron_semantic_segmentation as detectron

class SemanticSegmentationNode:
    
    def __init__(self):
        confidence_threshold = rospy.get_param("~confidence_threshold", 0.8)
        self.detectron_cfg, self.detectron_model = detectron.load_model(confidence_threshold)
        self.image_subscriber = rospy.Subscriber("/camera/image", Image, self.image_callback, queue_size=1)
        self.segmentation_mask_publisher = rospy.Publisher("/detectron2/mask/semantic", Image, queue_size=10)
        self.bridge = CvBridge()
        
    def image_callback(self, msg : Image) -> None:
        # Ignore old messages
        stamp = msg.header.stamp
        duration = (rospy.Time.now() - stamp).to_sec()
        if duration >= 0.2:
            return
        try:
            image_array_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # Preprocessing
        image_array_bgr = (
            rescale(
                equalize_hist(image_array_bgr) * 255, scale=0.2, preserve_range=True, multichannel=True
            ) 
        ).astype(np.uint8)
        image_array_rgb = image_array_bgr[...,::-1]

        # Prediction and decoding
        prediction_results = detectron.predict(image_array_rgb, self.detectron_model)
        mask = detectron.get_segmentation_mask(prediction_results)
        
        # Publish
        self.segmentation_mask_publisher.publish(
            self.bridge.cv2_to_imgmsg(mask, encoding="mono16")
        )
        
if __name__ == "__main__":
    rospy.init_node("semantic_segmentation_node")
    node = SemanticSegmentationNode()
    rospy.spin()