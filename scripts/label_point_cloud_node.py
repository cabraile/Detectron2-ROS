#!/usr/bin/python3
import numpy as np
import rospy
import ros_numpy
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

class LabelPointCloudNode:

    def __init__(self):
        # Subscribers
        self.pointcloud_subscriber = rospy.Subscriber("/pointcloud", PointCloud2, callback=self.pointcloud_callback)
        self.mask_subscriber = rospy.Subscriber("/mask/semantic", Image, callback=self.mask_callback)
        self.camera_info_subscriber = rospy.Subscriber("/camera/camera_info", CameraInfo, callback=self.camera_info_callback)
        
        # Publishers
        self.pointcloud_publisher = rospy.Publisher("/semantic/pointcloud", PointCloud2, queue_size=10)

        self.cv_bridge = cv_bridge.CvBridge()
        self.mask_msg = None
        self.pointcloud_msg = None

        # Flags 
        self.flags = {"can_project" : False}
    
    def mask_callback(self, mask_msg : Image) -> None:
        self.mask_msg = mask_msg
        if self.pointcloud_msg is None:
            return
        self.project_and_label()
    
    def pointcloud_callback(self, pointcloud_msg : PointCloud2) -> None:
        self.pointcloud_msg = pointcloud_msg
        if self.mask_msg is None:
            return
        self.project_and_label()

    def project_and_label(self) -> None:
        if not self.flags["can_project"]:
            return
        
        mask_msg = self.mask_msg
        pointcloud_msg = self.pointcloud_msg
            

        # Load cloud data
        cloud_sarray = ros_numpy.numpify(pointcloud_msg) # as structured array
        cloud_array = np.column_stack([
            cloud_sarray["x"].flatten(),
            cloud_sarray["y"].flatten(),
            cloud_sarray["z"].flatten(),
        ])
        cloud_array = np.column_stack([
            cloud_array, 
            np.ones(len(cloud_array))
        ])
        # Project points to mask
        P = self.calibration_parameters["P"]
        K = self.calibration_parameters["K"]
        nrows = self.calibration_parameters["height"]
        ncols = self.calibration_parameters["width"]

        projected_points = (K @ P @ cloud_array.T).T
        projected_points_cols_rows = (projected_points[:,:2]/projected_points[:,2].reshape(-1,1)).astype(int)
        rows, cols = (projected_points_cols_rows[:,1], projected_points_cols_rows[:,0])
        
        # Filter out of bounds points
        ids = np.where(
            (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
        )
        rows = rows[ids]
        cols = cols[ids]

        # Label cloud points
        mask_img = self.cv_bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono16")
        class_ids = mask_img[rows, cols].flatten()

        # Create labeled cloud
        label_cloud_sarray = np.zeros(len(rows), dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("label", np.uint16)
        ])
        for field in ["x", "y", "z"]:
            label_cloud_sarray[field] = cloud_sarray[field].flatten()[ids]
        label_cloud_sarray["label"] = class_ids

        # Publish labeled cloud
        label_pointcloud_msg = ros_numpy.msgify(PointCloud2, label_cloud_sarray)
        label_pointcloud_msg.header.frame_id = pointcloud_msg.header.frame_id
        label_pointcloud_msg.header.stamp = pointcloud_msg.header.stamp
        label_pointcloud_msg.header.seq = pointcloud_msg.header.seq
        self.pointcloud_publisher.publish(label_pointcloud_msg)
        
        self.mask_msg = None
        self.pointcloud_msg = None

    def camera_info_callback(self, msg : CameraInfo) -> None:
        self.calibration_parameters = { 
            "K" : np.array(msg.K).reshape(3,3),
            "P" : np.array(msg.P).reshape(3,4),
            "height" : msg.height,
            "width" : msg.width
        }
        self.flags["can_project"] = True
        self.camera_info_subscriber.unregister()

def main() -> int:
    rospy.init_node("label_point_cloud_node")
    lpc_node = LabelPointCloudNode()
    rospy.spin()
    return 0 

if __name__ == "__main__":
    exit(main())