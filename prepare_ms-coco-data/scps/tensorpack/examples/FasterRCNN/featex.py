#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
from tqdm import tqdm
import pickle

import tensorpack.utils.viz as tpviz
from tensorpack.predict import (
    MultiTowerOfflinePredictor,
    OfflinePredictor,
    PredictConfig,
)
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_balloon
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import (
    DetectionResult,
    multithread_predict_dataflow,
    predict_image,
    extract_features,
)
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load a model for evaluation.", required=True)
    parser.add_argument("--predict", help="Run prediction on given images. ")
    parser.add_argument(
        "--config",
        help="A list of KEY=VALUE to overwrite those defined in config.py",
        nargs="+",
    )
    parser.add_argument("--output-filename", help="name of featex files")
    parser.add_argument("--output-dir", help="Save featex to dir")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    register_balloon(cfg.DATA.BASEDIR)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util

        assert (
            get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled()
        ), "Inference requires either GPU support or MKL support!"
    assert args.load
    finalize_configs(is_training=False)

    if args.predict:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1],
    )

    with open(args.predict, "rb") as fh:
        image_paths = pickle.load(fh)

    detect_res = {}
    predictor = OfflinePredictor(predcfg)

    pbar = tqdm(image_paths)
    for image_file in pbar:
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        detect_res[image_file] = extract_features(img, predictor)

    with open(os.path.join(args.output_dir, args.output_filename), "wb") as fh:
        pickle.dump(detect_res, fh, protocol=pickle.HIGHEST_PROTOCOL)
