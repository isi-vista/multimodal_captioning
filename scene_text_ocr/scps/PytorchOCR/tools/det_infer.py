# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib
import numpy as np
import tqdm
from PIL import Image
from os.path import basename, splitext

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeShortSize
from torchocr.postprocess import build_post_process


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


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def get_image_list(horizontal_list, img, model_height=64, sort_output = True):
    image_list = []

    for box in box_list:
        rect = np.array(box, dtype = "float32")
        transformed_img = four_point_transform(img, rect)
        ratio = transformed_img.shape[1]/transformed_img.shape[0]
        crop_img = cv2.resize(transformed_img, (int(model_height*ratio), model_height), interpolation =  Image.ANTIALIAS)
        image_list.append( crop_img ) # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    """
    maximum_y,maximum_x, _ = img.shape
    for box in horizontal_list:
        x_min = max(0,min(list(list(zip(*box))[0])))
        x_max = min(max(list(list(zip(*box))[0])),maximum_x)
        y_min = max(0, min(list(list(zip(*box))[1])))
        y_max = min(max(list(list(zip(*box))[1])),maximum_y)
        crop_img = img[int(y_min) : int(y_max), int(x_min):int(x_max), :]
        image_list.append( crop_img )
    """

    return image_list

def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="PytorchOCR infer")
    parser.add_argument("--model_path", required=True, type=str, help="rec model path")
    parser.add_argument(
        "--file_path", required=True, type=str, help="img list file path for predict"
    )
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import cv2
    from torchocr.utils import draw_ocr_box_txt, draw_bbox

    args = init_args()
    model = DetInfer(args.model_path)

    with open(args.file_path, "r") as fh:
        lines = fh.readlines()
        for line in tqdm.tqdm(lines):
            img_path = line.strip()
            bname = splitext(basename(img_path))[0]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            box_list, score_list = model.predict(img, is_output_polygon=False)
            image_list = get_image_list( box_list, img )

            for idx, image in enumerate(image_list):
                cv2.imwrite(args.output_path + "/" + bname +"_" + str(idx) + ".png", image)

    """
    img = cv2.imread(args.file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box_list, score_list = model.predict(img, is_output_polygon=False)
    # img = draw_ocr_box_txt(img, box_list)
    img = draw_bbox(img, box_list)
    cv2.imwrite("./tmp.png", img)
    # plt.imshow(img)
    # plt.show()
    """
