# 1. About
This project aims to wrap the [Detectron2](https://github.com/facebookresearch/detectron2) implementation  for using it with ROS. Currently, only a node for semantic segmentation is implemented, but in later versions I aim to release the object detection node as well.


# 2. Setup
## 2.1 ROS
* Make sure ROS1 is installed in your machine. This project was tested using the ROS Noetic version. Installation instructions can be found [here](http://wiki.ros.org/noetic/Installation).

## 2.2 Setup your GPU (Optional)
* Even though this framework can be run without GPU, proper GPU support will enhance the performance of your application.
* In this project, the NVIDIA-toolkit v11.3 was used in a Ubuntu 20.04 machine. The toolkit and drivers can be downloaded [here](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local). I suggest running the runfile instead of the `deb` installation.

## 2.3 Setup the correct pytorch version
* In order to install the Detectron2 module, it is necessary to have installed the pytorch module.
  * In their installation tutorial they provide the compatible CUDA-torch versions. 
  * If you have installed the NVIDIA-toolkit 11.3, run `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` (you will probably need admistrative privileges for that - include `sudo` if required). 
  * Otherwise check out their compatibility list in [this tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [pytorch installation instructions](https://pytorch.org/get-started/locally/).

## 2.4 Detectron2
* This project's based upon the [Detectron2](https://github.com/facebookresearch/detectron2) framework developped by Facebook. Follow their installation and configuration instructions from [this link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## 2.5 Install dependencies
* In the root directory of this project, run `pip install -r requirements.txt`

## 2.6 Make the scripts executable
* In the root directory of this project, run `chmod +x scripts/*.py`.

# 3. Nodes

## 3.1. Semantic segmentation node
Implemented in `scripts/semantic_segmentation_node.py`.

**Arguments**
* `confidence_threshold` (Optional): How certain must a prediction be so it is labelled in the mask. Ranges between 0 and 1.
* `process_scale` (Optional): The scale in which the image will be processed. This value can be any value starting from 0, in which `process_scale = 0.5` means that the image to be processed will be half of the size of the input image. The output dimensions will be equals to the input's dimensions regardless the value.

**Subscribes** to the topic `/camera/image` (`sensor_msgs.Image`). You might want to remap this topic.

**Publishes** the segmentation mask (`sensor_msgs.Image`) in `/detectron2/mask/semantic`. The published encoding is `mono16`.

## 3.2 Point cloud labelling node
Implemented in `scripts/label_point_cloud_node.py` and used with the semantic segmentation node.

**Subscribes to**
* `/pointcloud` (`sensor_msgs.PointCloud2`). The point cloud points to be labelled.
* `/camera/camera_info` (`sensor_msgs.CameraInfo`). The intrinsic parameters of the camera from which the images were labelled.
* `/mask/semantic` (`sensor_msgs.Image`). The semantic mask (encoded as a `mono16` data type) from the predicted input. Tipically this topic **correponds to the output of the semantic segmentation node**.

**Publishes**
* `/semantic/pointcloud` (`sensor_msgs.PointCloud2`). The point cloud in which points that belong into the semantic mask are labelled. The data field that corresponds to the label is `label`.

# 4. Launch files

## 4.1 `semantic_segmentation.launch`
Wraps both the image semantic segmentation and point cloud labelling.

**Arguments**
* `image_topic`: The image to be segmented. Refer to the semantic segmentation node for more information.
* `camera_info_topic`: The calibration parameters of the input image.
* `point_cloud_topic`: The topic of the point clouds to be labeled.
* `confidence_threshold`: The threshold of the model's confidence for accepting a label.
* `process_scale`: The scale of the image to be processed.