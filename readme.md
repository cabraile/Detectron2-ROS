# About
This project aims to wrap the Detectron2 implementation  for using it with ROS. Currently, only a node for semantic segmentation is implemented, but in later versions I aim to release the object detection node as well.

# Usage

## Semantic segmentation node
Implemented in `scripts/semantic_segmentation_node.py`, it requires the **argument**:
* `confidence_threshold`: How certain must a prediction be so it is labelled in the mask. Ranges between 0 and 1.

**Publishes** the segmentation mask in `/detectron2/mask/semantic`

**Subscribes** to the topic `/camera/image`. You might want to remap this topic. 

A launch file wraps this node's call in `launch/semantic_segmentation.launch`, in which both the camera topic and confidence threshold can be set.

# Setup
**ROS**
* Make sure ROS1 is installed in your machine. This project was tested using the ROS Noetic version. Installation instructions can be found [here](http://wiki.ros.org/noetic/Installation).

**Setup your GPU (GPU)**
* Even though this framework can be run without GPU, proper GPU support will enhance the performance of your application.
* In this project, the NVIDIA-toolkit v11.3 was used in a Ubuntu 20.04 machine. The toolkit and drivers can be downloaded [here](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local). I suggest running the runfile instead of the `deb` installation.

**Setup the correct pytorch version**
* In order to install the Detectron2 module, it is necessary to have installed the pytorch module.
  * In their installation tutorial they provide the compatible CUDA-torch versions. 
  * If you have installed the NVIDIA-toolkit 11.3, run `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html` (you will probably need admistrative privileges for that - include `sudo` if required). 
  * Otherwise check out their compatibility list in [this tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [pytorch installation instructions](https://pytorch.org/get-started/locally/).

**Detectron2**:
* This project's based upon the [Detectron2](https://github.com/facebookresearch/detectron2) framework developped by Facebook. Follow their installation and configuration instructions from [this link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

**Install dependences**
* In the root directory of this project, run `pip install -r requirements.txt`

**Make the scripts executable**
* In the root directory of this project, run `chmod +x scripts/*.py`.
