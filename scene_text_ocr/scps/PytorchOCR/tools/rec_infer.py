# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import numpy as np
import pathlib
import tqdm 

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter
from wer import compute_scores

class RecInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt["cfg"]
        self.model = build_model(cfg["model"])
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg["dataset"]["train"]["dataset"])
        self.converter = CTCLabelConverter(cfg["dataset"]["alphabet"])

    def predict(self, img):
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt


def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="PytorchOCR infer")
    parser.add_argument("--model_path", required=True, type=str, help="rec model path")
    parser.add_argument(
        "--img_path", required=True, type=str, help="img path for predict"
    )
    parser.add_argument("--output_file", required=True, type=str, help="output OCR results")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import cv2

    args = init_args()
    model = RecInfer(args.model_path)

    images, words = [], []
    with open(args.img_path, "r") as fh:
        for line in fh:
            images.append(line.strip().split()[0])

    trans = {}
    conf = {}
    for image in tqdm.tqdm(images):
        img = cv2.imread(image)
        if img.shape[0] < 10 or img.shape[1] < 10:
            continue

        hypothesis = model.predict(img)

        words = []
        scores = []
        for hyp in hypothesis:
            word = hyp[0]
            score = np.mean(hyp[1])
            words.append(word)
            scores.append(score)

        trans[image] = words
        conf[image] = scores

    with open(args.output_file, "w") as fh:
        for image in trans:
            fh.write(image + " " + " ".join(trans[image]) + "\n")
            fh.write(image + " " + " ".join(str(i) for i in conf[image]) + "\n")
