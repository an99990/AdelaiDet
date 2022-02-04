# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import multiprocessing as mp
from halodi_segmentation.config import settings
from pathlib import Path

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from adet.config import get_cfg

import Fire
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args[0])
    cfg.merge_from_list(args[-1])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args[2]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args[2]
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args[2]
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args[2]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args[2]
    cfg.freeze()
    return cfg


def main():
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    root_dir = Path(__file__).parents[0]
    model_weigths = Path(__file__).parents[2]
    args = [f'{root_dir}/configs/BlendMask/R_101_dcni3_5x.yaml',f'{root_dir}/{settings.paths[0].image}',0.35,['MODEL.WEIGHTS',f'{model_weigths}/models_weights/R_101_dcni3_5x.pth']]

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    root_dir = Path(__file__).parents[3]
    path = root_dir / settings.paths[0].image
    logger.info(f"image: {path}")
    img = read_image(path, format="BGR")
    predictions, _ = demo.run_on_image(img)

if __name__ == "__main__":
    Fire.fire(main)


