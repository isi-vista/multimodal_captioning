# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import lmdb
import mmap
import pathlib
from tqdm import tqdm

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeShortSize
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.postprocess import build_post_process
from torchocr.utils import CTCLabelConverter


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
        out, feat = self.model(tensor)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt, feat


class DetInfer:
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

        self.resize = ResizeShortSize(736, False)
        self.post_proess = build_post_process(cfg["post_process"])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg["dataset"]["train"]["dataset"]["mean"],
                    std=cfg["dataset"]["train"]["dataset"]["std"],
                ),
            ]
        )

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {"img": img, "shape": [img.shape[:2]], "text_polys": []}
        data = self.resize(data)
        tensor = self.transform(data["img"])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        box_list, score_list = self.post_proess(
            out, data["shape"], is_output_polygon=is_output_polygon
        )
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [
        (min(x_coordinates), min(y_coordinates)),
        (max(x_coordinates), max(y_coordinates)),
    ]


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="PytorchOCR infer")
    parser.add_argument(
        "--det_model_path", required=True, type=str, help="rec model path"
    )
    parser.add_argument(
        "--rec_model_path", required=True, type=str, help="rec model path"
    )
    parser.add_argument("--img_dir", required=True, type=str, help="image folder")
    parser.add_argument(
        "--img_list_file",
        required=True,
        type=str,
        help="path for image list file for predict",
    )
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        help="path for predictedscene ocr outputs",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils import draw_ocr_box_txt, draw_bbox

    args = init_args()

    det_model = DetInfer(args.det_model_path)
    rec_model = RecInfer(args.rec_model_path)

    env = lmdb.open(args.out_file, map_size=1000000000)

    with open(args.img_list_file, "r") as file, env.begin(write=True) as txn:
        for line in tqdm(file, total=get_num_lines(args.img_list_file)):
            img_path = os.path.join(args.img_dir, line.strip())

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            box_list, score_list = det_model.predict(img, is_output_polygon=False)
            if box_list:
                feats = []
                for box in box_list:
                    bbox = bounding_box(box)
                    cropped_img = img[
                        int(bbox[0][1]) : int(bbox[1][1]),
                        int(bbox[0][0]) : int(bbox[1][0]),
                    ]
                    out, feat = rec_model.predict(cropped_img)
                    feats.append(feat)
                feats = torch.cat(feats, dim=1)
                txn.put(line.strip().encode("ascii"), feats.cpu().detach().numpy())
