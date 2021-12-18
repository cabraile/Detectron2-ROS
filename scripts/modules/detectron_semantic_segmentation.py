#!/usr/bin/python3
from typing import Dict, Any, Tuple
import numpy as np

from detectron2.config import get_cfg, CfgNode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def load_model(
    confidence_threshold : float,
    force_cpu : bool = False
) -> Tuple[CfgNode, DefaultPredictor]:
    cfg = get_cfg()
    dataset_name = "COCO-PanopticSegmentation"
    model_name = "panoptic_fpn_R_50_3x"
    model_type = f"{dataset_name}/{model_name}"
    cfg.merge_from_file(model_zoo.get_config_file(f"{model_type}.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model_type}.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # set threshold for this model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1
    if force_cpu:
        cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def predict(
    image_rgb : np.ndarray,
    predictor : DefaultPredictor,
) -> Dict[str, Any]:
    image_bgr = image_rgb[...,::-1]
    outputs = predictor(image_bgr)
    return outputs

def get_segmentation_mask(results_dict : Dict[str, Any]) -> np.ndarray:
    """https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BitMasks
    """
    mask_channels_first = results_dict["sem_seg"].cpu().detach().numpy()
    mask_channels_last = np.swapaxes(mask_channels_first[...,np.newaxis], 0, 3)[0]
    mask = np.argmax(mask_channels_last, axis=2).astype(np.uint16)
    return mask